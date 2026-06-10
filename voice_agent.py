#!/usr/bin/env python3
"""WebSocket voice agent demo: browser mic -> STT -> LLM -> megakernel TTS -> browser.

Pipeline (all in this process):
    browser ⇄ WebSocket (protobuf frames) ⇄ FastAPIWebsocketTransport
        -> OpenAI STT (gpt-4o-mini-transcribe)
        -> OpenAI LLM (gpt-4o-mini)
        -> MegakernelTTSService (in-process, streams PCM as it decodes)

Run on the GPU server (needs OPENAI_API_KEY in .env):
    python3 voice_agent.py        # listens on 127.0.0.1:8001

Access from your machine via SSH tunnel:
    ssh -L 8001:localhost:8001 -p <port> root@<server>
    -> open http://localhost:8001, click Start, talk.

WebSocket (TCP) is used instead of WebRTC because the demo rides an SSH
local-forward, which cannot carry WebRTC's UDP media. WebRTC is the
production path for clients on the open internet.
"""

import asyncio
import datetime
import json
import os
import sys
import threading
import time
import wave

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMRunFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.getenv("VOICE_AGENT_PORT", "8001"))
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-realtime-whisper")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "ryan")

SPEAKERS = ["ryan", "serena", "vivian", "uncle_fu", "aiden",
            "ono_anna", "sohee", "eric", "dylan"]

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your responses are converted to "
    "speech, so answer in short conversational sentences with no markdown, "
    "lists, emojis, or special characters. Keep replies to one or two "
    "sentences unless asked for detail."
)

app = FastAPI(title="Megakernel Voice Agent")

engine = None
engine_lock = threading.Lock()  # one TTS generation at a time (single KV cache)


def get_engine():
    global engine
    if engine is None:
        from qwen_tts_megakernel.engine import MegakernelTTS

        engine = MegakernelTTS()
        for _ in engine.stream("Warm up.", speaker=TTS_SPEAKER, max_new_tokens=30):
            pass
    return engine


class ControlChannel(FrameProcessor):
    """Handles JSON control messages from the browser (e.g. speaker switch)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            msg = frame.message if isinstance(frame.message, dict) else {}
            if msg.get("type") == "set_speaker" and msg.get("speaker") in SPEAKERS:
                from qwen_tts_megakernel.pipecat_tts import MegakernelTTSSettings

                logger.info(f"Switching TTS speaker to {msg['speaker']}")
                await self.push_frame(
                    TTSUpdateSettingsFrame(
                        delta=MegakernelTTSSettings(voice=msg["speaker"])
                    )
                )
            return  # control messages are not pipeline data
        await self.push_frame(frame, direction)


class TurnRecorder(FrameProcessor):
    """Collects each bot turn's audio + metrics; saves a wav and notifies the UI.

    Sits between the TTS service and the transport output, so it sees the
    TTS audio frames, the per-service MetricsFrames, and the turn-boundary
    frames. At each turn end it writes voices/agent_*.wav (+ .json meta) and
    pushes a transport message the browser uses to render the turn row.
    """

    def __init__(self, speaker: str, session: str):
        super().__init__()
        self._speaker = speaker
        self._session = session
        self._pcm = bytearray()
        self._sample_rate = 24000
        self._ttfb: dict[str, float] = {}
        self._turn = 0
        self._t_user_stop: float | None = None
        self._response_ms: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData) and d.value > 0:
                    self._ttfb[d.processor.split("#")[0]] = round(d.value * 1000)
        elif isinstance(frame, TTSUpdateSettingsFrame):
            if frame.delta is not None and getattr(frame.delta, "voice", None):
                self._speaker = frame.delta.voice
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._t_user_stop = time.perf_counter()
        elif isinstance(frame, TTSAudioRawFrame):
            if not self._pcm and self._t_user_stop is not None:
                self._response_ms = round(
                    (time.perf_counter() - self._t_user_stop) * 1000)
            self._sample_rate = frame.sample_rate
            self._pcm.extend(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            await self._finalize(interrupted=False)
        elif isinstance(frame, InterruptionFrame):
            await self._finalize(interrupted=True)
        await self.push_frame(frame, direction)

    async def _finalize(self, interrupted: bool):
        if not self._pcm:
            return
        self._turn += 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{stamp}_agent_{self._session}_turn{self._turn}_{self._speaker}.wav"
        meta = {
            "engine": "voice_agent",
            "session": self._session,
            "turn": self._turn,
            "file": fname,
            "speaker": self._speaker,
            "audio_seconds": round(len(self._pcm) / 2 / self._sample_rate, 3),
            "response_ms": self._response_ms,
            "ttfb_ms": dict(self._ttfb),
            "interrupted": interrupted,
            "created": stamp,
        }
        pcm, sr = bytes(self._pcm), self._sample_rate
        self._pcm = bytearray()
        self._ttfb = {}
        self._response_ms = None
        self._t_user_stop = None

        def write_files():
            with wave.open(os.path.join(VOICES_DIR, fname), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
            with open(os.path.join(VOICES_DIR,
                                   fname.removesuffix(".wav") + ".json"), "w") as f:
                json.dump(meta, f)

        # Fire-and-forget: disk I/O must not stall the audio pipeline.
        asyncio.create_task(asyncio.to_thread(write_files))


async def run_pipeline(websocket: WebSocket, speaker: str, session: str):
    from qwen_tts_megakernel.pipecat_tts import MegakernelTTSService

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    # Realtime STT streams transcription while the user is still speaking
    # (local-VAD mode: our Silero VAD commits the audio buffer on speech end).
    stt = OpenAIRealtimeSTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(model=STT_MODEL),
    )
    llm = OpenAILLMService(settings=OpenAILLMService.Settings(model=LLM_MODEL))
    tts = MegakernelTTSService(
        engine=get_engine(), engine_lock=engine_lock, speaker=speaker,
        stop_frame_timeout_s=1.0,  # turn finalizes 1s after last audio
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            ControlChannel(),
            stt,
            user_agg,
            llm,
            tts,
            TurnRecorder(speaker=speaker, session=session),
            transport.output(),
            assistant_agg,
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        logger.info("Client disconnected, stopping pipeline")
        await task.cancel()

    # Have the bot greet as soon as the connection is up.
    await task.queue_frames([LLMRunFrame()])
    await PipelineRunner(handle_sigint=False).run(task)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    speaker = websocket.query_params.get("speaker", TTS_SPEAKER)
    if speaker not in SPEAKERS:
        speaker = TTS_SPEAKER
    session = "".join(c for c in websocket.query_params.get("session", "")
                      if c.isalnum())[:16] or "nosession"
    await websocket.accept()
    await run_pipeline(websocket, speaker, session)


@app.get("/turns")
def list_turns(session: str):
    turns = []
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".json") and "_agent_" in f:
            try:
                with open(os.path.join(VOICES_DIR, f)) as mf:
                    meta = json.load(mf)
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("session") == session:
                turns.append(meta)
    return sorted(turns, key=lambda m: m.get("turn", 0))


@app.get("/turns/{name}")
def get_turn_audio(name: str):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(400, "bad name")
    path = os.path.join(VOICES_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="audio/wav")


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Megakernel Voice Agent</title>
<script src="https://cdn.jsdelivr.net/npm/protobufjs@7.4.0/dist/protobuf.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
 h1{font-size:1.3rem}
 button{cursor:pointer;background:#2563eb;color:#eee;border:none;border-radius:8px;padding:.6rem 1.2rem;font-size:1rem}
 button.stop{background:#dc2626}
 #status{margin:.8rem 0;color:#9ca3af;min-height:1.2rem}
 #log{margin-top:1rem;border-top:1px solid #2a2a2a;padding-top:.6rem;font-size:.95rem;line-height:1.5}
 .you{color:#93c5fd}.bot{color:#86efac}
 select{background:#1c1c1c;color:#eee;border:1px solid #444;border-radius:8px;padding:.5rem .9rem;font-size:1rem;margin-right:.5rem}
 table{width:100%;border-collapse:collapse;margin-top:1rem}
 td,th{padding:.4rem;border-bottom:1px solid #2a2a2a;text-align:left;font-size:.85rem}
 audio{height:2rem;vertical-align:middle}
 a.dl{color:#93c5fd}
</style></head><body>
<h1>🎙️ Megakernel Voice Agent</h1>
<p>STT: OpenAI · LLM: OpenAI · TTS: Qwen3-TTS megakernel (local 5090)</p>
<select id="speaker">
 <option>ryan</option><option>serena</option><option>vivian</option><option>uncle_fu</option>
 <option>aiden</option><option>ono_anna</option><option>sohee</option><option>eric</option><option>dylan</option>
</select>
<button id="btn">Start</button>
<button id="load" style="background:#16a34a">Get session recordings</button>
<div id="status">idle</div>
<div id="log"></div>
<h2 style="font-size:1.05rem">Recorded bot turns (this session)</h2>
<table><thead><tr><th>#</th><th>speaker</th><th>audio</th><th>response</th>
<th>STT</th><th>LLM</th><th>TTS</th><th></th><th></th></tr></thead>
<tbody id="turns"></tbody></table>
<script>
// Mirrors pipecat's frames.proto (only the messages the serializer uses).
const PROTO = `syntax = "proto3"; package pipecat;
message TextFrame { uint64 id=1; string name=2; string text=3; }
message AudioRawFrame { uint64 id=1; string name=2; bytes audio=3; uint32 sample_rate=4; uint32 num_channels=5; optional uint64 pts=6; }
message TranscriptionFrame { uint64 id=1; string name=2; string text=3; string user_id=4; string timestamp=5; }
message MessageFrame { string data=1; }
message InterruptionFrame { uint64 id=1; string name=2; }
message Frame { oneof frame { TextFrame text=1; AudioRawFrame audio=2; TranscriptionFrame transcription=3; MessageFrame message=4; InterruptionFrame interruption=5; } }`;
const FrameMsg = protobuf.parse(PROTO).root.lookupType("pipecat.Frame");

const $ = id => document.getElementById(id);
const SESSION = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
let ws, micCtx, micNode, micStream, playCtx, playT = 0, playing = [];
let running = false;

async function loadTurns(){
  const r = await fetch(`/turns?session=${SESSION}`);
  const turns = await r.json();
  const ms = v => v != null ? v + ' ms' : '—';
  $('turns').innerHTML = turns.map(m => {
    const ttfb = m.ttfb_ms || {};
    const url = `/turns/${encodeURIComponent(m.file)}`;
    return `<tr><td>${m.turn}${m.interrupted ? ' ✂' : ''}</td><td>${m.speaker}</td>` +
      `<td>${m.audio_seconds.toFixed(1)}s</td><td>${ms(m.response_ms)}</td>` +
      `<td>${ms(ttfb.OpenAIRealtimeSTTService ?? ttfb.OpenAISTTService)}</td>` +
      `<td>${ms(ttfb.OpenAILLMService)}</td>` +
      `<td>${ms(ttfb.MegakernelTTSService)}</td>` +
      `<td><audio controls preload="none" src="${url}"></audio></td>` +
      `<td><a class="dl" href="${url}" download="${m.file}">save</a></td></tr>`;
  }).reverse().join('');
  $('status').textContent = `${turns.length} recorded turn(s) in this session`;
}
$('load').onclick = loadTurns;

function log(cls, who, text){
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = who + ': ' + text;
  $('log').prepend(d);
}

function flushPlayback(){
  playing.forEach(s => { try { s.stop(); } catch(e){} });
  playing = [];
  if (playCtx) playT = playCtx.currentTime;
}

function playAudio(bytes, sr){
  const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  if (!i16.length) return;
  const f32 = Float32Array.from(i16, v => v / 32768);
  const buf = playCtx.createBuffer(1, f32.length, sr);
  buf.copyToChannel(f32, 0);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);
  playT = Math.max(playT, playCtx.currentTime + 0.05);
  src.start(playT);
  playT += buf.duration;
  playing.push(src);
  src.onended = () => { playing = playing.filter(s => s !== src); };
}

async function start(){
  playCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
  await playCtx.resume();

  micStream = await navigator.mediaDevices.getUserMedia({audio: {
    echoCancellation: true, noiseSuppression: true, autoGainControl: true,
  }});
  micCtx = new AudioContext({sampleRate: 16000});
  const src = micCtx.createMediaStreamSource(micStream);
  micNode = micCtx.createScriptProcessor(512, 1, 1);
  micNode.onaudioprocess = e => {
    if (!ws || ws.readyState !== 1) return;
    const f = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++){
      const s = Math.max(-1, Math.min(1, f[i]));
      i16[i] = s < 0 ? s * 32768 : s * 32767;
    }
    const msg = FrameMsg.create({audio: {
      audio: new Uint8Array(i16.buffer), sampleRate: 16000, numChannels: 1,
    }});
    ws.send(FrameMsg.encode(msg).finish());
  };
  src.connect(micNode);
  micNode.connect(micCtx.destination);

  ws = new WebSocket(`ws://${location.host}/ws?speaker=${$('speaker').value}&session=${SESSION}`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('status').textContent = 'connected — talk!'; };
  ws.onclose = () => { $('status').textContent = 'disconnected'; stop(); };
  ws.onerror = () => { $('status').textContent = 'websocket error'; };
  ws.onmessage = ev => {
    const frame = FrameMsg.decode(new Uint8Array(ev.data));
    if (frame.audio) {
      playAudio(frame.audio.audio, frame.audio.sampleRate || 24000);
    } else if (frame.interruption) {
      flushPlayback();
    } else if (frame.transcription) {
      log('you', 'You', frame.transcription.text);
    } else if (frame.text) {
      log('bot', 'Bot', frame.text.text);
    }
  };
}

function stop(){
  running = false;
  $('btn').textContent = 'Start';
  $('btn').classList.remove('stop');
  if (ws && ws.readyState === 1) ws.close();
  if (micNode) micNode.disconnect();
  if (micCtx) micCtx.close();
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  flushPlayback();
  ws = micCtx = micNode = micStream = null;
}

$('speaker').onchange = () => {
  if (ws && ws.readyState === 1) {
    const msg = FrameMsg.create({message: {data: JSON.stringify(
      {type: 'set_speaker', speaker: $('speaker').value})}});
    ws.send(FrameMsg.encode(msg).finish());
    $('status').textContent = `speaker -> ${$('speaker').value} (next reply)`;
  }
};

$('btn').onclick = async () => {
  if (running) { stop(); $('status').textContent = 'idle'; return; }
  running = true;
  $('btn').textContent = 'Stop';
  $('btn').classList.add('stop');
  $('status').textContent = 'connecting…';
  try { await start(); } catch(e){ $('status').textContent = 'error: ' + e.message; stop(); }
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — put it in .env (see .env.example)")
    get_engine()  # load model + capture CUDA graphs before accepting clients
    logger.info(f"Voice agent ready: http://localhost:{PORT} (tunnel: ssh -L {PORT}:localhost:{PORT} ...)")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
