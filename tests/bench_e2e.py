#!/usr/bin/env python3
"""End-to-end voice agent latency benchmark (no microphone needed).

Connects to the voice agent's WebSocket, plays a spoken question at it
(paced in real time like a mic), and measures the gap between the end of the
spoken question and the first audio frame of the bot's reply — the latency a
human would perceive.

Requires voice_agent.py running on :8001 and a question wav, e.g.:

    python3 synthesize.py "What is the capital of France?" \\
        --speaker serena --out /tmp/question.wav
    python3 tests/bench_e2e.py 3 --question-wav /tmp/question.wav \\
        [--agent-log /tmp/agent.log]

--agent-log additionally prints the per-stage timeline (turn-stop, per-service
TTFB) parsed from the server log after the run.
"""

import argparse
import asyncio
import re
import time
import wave

import numpy as np
import resampy
import websockets

import pipecat.frames.protobufs.frames_pb2 as pb

WS_URL = "ws://127.0.0.1:8001/ws"
CHUNK = 512  # samples per packet @16kHz = 32 ms, like the browser client
SR = 16000


def load_question_pcm(path: str) -> np.ndarray:
    """Load a mono int16 wav and resample to 16 kHz."""
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need mono int16 wav"
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr == SR:
        return pcm
    f16k = resampy.resample(pcm.astype(np.float32) / 32768.0, sr, SR)
    return (np.clip(f16k, -1, 1) * 32767).astype(np.int16)


def print_stage_timeline(log_path: str, n_turns: int):
    """Per-stage breakdown of the last n_turns, parsed from the agent log."""
    pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*?"
        r"(User stopped speaking|(\w+Service)#\d+ TTFB: ([\d.]+)s)")
    turns, current = [], None
    for line in open(log_path, errors="replace"):
        m = pat.match(line)
        if not m:
            continue
        if m.group(2) == "User stopped speaking":
            current = {"stop": m.group(1)}
            turns.append(current)
        elif current is not None:
            current[m.group(3)] = float(m.group(4))
    print(f"\nper-stage TTFB from {log_path} (last {n_turns} turns):")
    for t in turns[-n_turns:]:
        stages = " | ".join(f"{k} {v*1000:.0f} ms" for k, v in t.items() if k != "stop")
        print(f"  turn-stop {t['stop'][11:]} | {stages}")


def audio_msg(samples: np.ndarray) -> bytes:
    frame = pb.Frame(audio=pb.AudioRawFrame(
        audio=samples.tobytes(), sample_rate=SR, num_channels=1))
    return frame.SerializeToString()


async def main(n_turns: int, question_wav: str, agent_log: str | None):
    question = load_question_pcm(question_wav)
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

    if agent_log:
        print_stage_timeline(agent_log, n_turns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n_turns", nargs="?", type=int, default=3)
    ap.add_argument("--question-wav", required=True,
                    help="spoken question (mono int16 wav, any sample rate)")
    ap.add_argument("--agent-log", default=None,
                    help="voice_agent.py log file for the per-stage breakdown")
    args = ap.parse_args()
    asyncio.run(main(args.n_turns, args.question_wav, args.agent_log))
