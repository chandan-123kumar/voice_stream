#!/usr/bin/env python3
"""End-to-end voice agent latency benchmark (no microphone needed).

Connects to the voice agent's WebSocket, plays a synthesized question at it
(paced in real time like a mic), and measures the gap between the end of the
spoken question and the first audio frame of the bot's reply — the latency a
human would perceive.

Requires both servers running: app.py on :8000 (used once to synthesize the
question audio) and voice_agent.py on :8001.

    python3 tests/bench_e2e.py [n_turns]
"""

import asyncio
import sys
import time
import urllib.request

import numpy as np
import resampy
import websockets

import pipecat.frames.protobufs.frames_pb2 as pb

WS_URL = "ws://127.0.0.1:8001/ws"
TTS_URL = "http://127.0.0.1:8000/generate"
QUESTION = "What is the capital of France?"
CHUNK = 512  # samples per packet @16kHz = 32 ms, like the browser client
SR = 16000


def make_question_pcm() -> np.ndarray:
    """Synthesize the question via app.py and resample 24k -> 16k mono int16."""
    req = urllib.request.Request(
        TTS_URL,
        data=('{"text": "%s", "speaker": "serena"}' % QUESTION).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        pcm24 = np.frombuffer(r.read(), dtype=np.int16)
    f32 = pcm24.astype(np.float32) / 32768.0
    f16k = resampy.resample(f32, 24000, SR)
    return (np.clip(f16k, -1, 1) * 32767).astype(np.int16)


def audio_msg(samples: np.ndarray) -> bytes:
    frame = pb.Frame(audio=pb.AudioRawFrame(
        audio=samples.tobytes(), sample_rate=SR, num_channels=1))
    return frame.SerializeToString()


async def main(n_turns: int = 3):
    question = make_question_pcm()
    silence = np.zeros(CHUNK, dtype=np.int16)
    print(f"question audio: {len(question)/SR:.2f}s")

    last_audio_rx = [0.0]
    rx_frames: list[tuple[float, int]] = []  # (arrival time, n bytes)
    transcripts: list[str] = []
    bot_text: list[str] = []

    async with websockets.connect(WS_URL, max_size=None) as ws:

        async def receiver():
            async for data in ws:
                frame = pb.Frame.FromString(data)
                which = frame.WhichOneof("frame")
                if which == "audio":
                    last_audio_rx[0] = time.perf_counter()
                    rx_frames.append((last_audio_rx[0], len(frame.audio.audio)))
                elif which == "transcription":
                    transcripts.append(frame.transcription.text)
                elif which == "text":
                    bot_text.append(frame.text.text)

        recv_task = asyncio.create_task(receiver())

        async def quiet(for_secs: float, feed_silence: bool = True):
            """Wait until no bot audio has arrived for `for_secs`."""
            while True:
                if last_audio_rx[0] and time.perf_counter() - last_audio_rx[0] > for_secs:
                    return
                if feed_silence:
                    await ws.send(audio_msg(silence))
                await asyncio.sleep(CHUNK / SR)

        # Let the greeting play out first.
        await quiet(2.0)
        print("greeting done, starting turns")

        latencies = []
        for turn in range(n_turns):
            rx_frames.clear()
            # Speak the question, paced like a live mic.
            for i in range(0, len(question), CHUNK):
                await ws.send(audio_msg(question[i:i + CHUNK]))
                await asyncio.sleep(CHUNK / SR)
            t_end = time.perf_counter()

            # Feed silence until reply audio newer than t_end shows up.
            while last_audio_rx[0] <= t_end:
                await ws.send(audio_msg(silence))
                await asyncio.sleep(CHUNK / SR)
            lat = last_audio_rx[0] - t_end
            latencies.append(lat)
            print(f"turn {turn}: speech-end -> first reply audio = {lat*1000:.0f} ms")

            await quiet(2.0)  # let the reply finish

            # Streaming profile: if audio were buffered-then-sent, all frames
            # would arrive in one burst (span ~0). Real-time pacing means the
            # arrival span tracks the audio duration.
            if len(rx_frames) > 1:
                span = rx_frames[-1][0] - rx_frames[0][0]
                audio_s = sum(n for _, n in rx_frames) / 2 / 24000
                gaps = [b[0] - a[0] for a, b in zip(rx_frames, rx_frames[1:])]
                print(f"  streaming profile: {len(rx_frames)} frames over "
                      f"{span:.2f}s for {audio_s:.2f}s of audio "
                      f"(max inter-frame gap {max(gaps)*1000:.0f} ms)")

        recv_task.cancel()

    print(f"\nheard: {transcripts}")
    print(f"bot said: {[t.strip() for t in bot_text if t.strip()]}")
    lats = sorted(latencies)
    print(f"\nlatency ms: min={lats[0]*1000:.0f} "
          f"median={lats[len(lats)//2]*1000:.0f} max={lats[-1]*1000:.0f}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
