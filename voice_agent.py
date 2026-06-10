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

import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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


async def run_pipeline(websocket: WebSocket):
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
        engine=get_engine(), engine_lock=engine_lock, speaker=TTS_SPEAKER
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
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
    await websocket.accept()
    await run_pipeline(websocket)


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
</style></head><body>
<h1>🎙️ Megakernel Voice Agent</h1>
<p>STT: OpenAI · LLM: OpenAI · TTS: Qwen3-TTS megakernel (local 5090)</p>
<button id="btn">Start</button>
<div id="status">idle</div>
<div id="log"></div>
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
let ws, micCtx, micNode, micStream, playCtx, playT = 0, playing = [];
let running = false;

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

  ws = new WebSocket(`ws://${location.host}/ws`);
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
