"""In-process Pipecat TTSService backed by the megakernel engine.

No network hop: the Pipecat pipeline calls MegakernelTTS.stream() directly and
audio chunks are pushed downstream as TTSAudioRawFrames the moment the codec
decodes them. The blocking decode loop runs in a worker thread; one utterance
generates at a time (single KV cache), guarded by an engine lock shared with
any other in-process user of the engine.

Usage:
    from qwen_tts_megakernel.engine import MegakernelTTS
    from qwen_tts_megakernel.pipecat_tts import MegakernelTTSService

    tts = MegakernelTTSService(engine=MegakernelTTS(), speaker="ryan")
    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
"""

import asyncio
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

from .engine import MegakernelTTS

# Qwen3-TTS wants full language names; map from BCP-47 primary subtags so any
# regional variant (en-US, pt-BR, ...) resolves to its family.
_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
}


@dataclass
class MegakernelTTSSettings(TTSSettings):
    """Settings for MegakernelTTSService (voice = Qwen speaker name)."""

    pass


class MegakernelTTSService(TTSService):
    """Local streaming TTS using the Qwen3-TTS megakernel engine in-process.

    Each utterance is one blocking generation on the GPU; chunks are handed to
    the asyncio side through a queue as soon as the speech tokenizer decodes
    them, so the first TTSAudioRawFrame leaves after `first_chunk_frames`
    codec frames rather than after the full utterance.
    """

    Settings = MegakernelTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        engine: Optional[MegakernelTTS] = None,
        speaker: str = "ryan",
        language: str = "English",
        engine_lock: Optional[threading.Lock] = None,
        gen_params: Optional[dict[str, Any]] = None,
        warmup: bool = True,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the megakernel TTS service.

        Args:
            engine: A loaded MegakernelTTS instance. Built (and CUDA-graph
                captured) here if not provided — pass a pre-warmed engine to
                keep pipeline startup fast.
            speaker: Default Qwen speaker (overridable via settings.voice).
            language: Default language name (overridable via settings.language).
            engine_lock: Lock serializing access to the engine's single KV
                cache; share it if anything else in the process generates.
            gen_params: Extra kwargs forwarded to MegakernelTTS.stream()
                (temperature, top_k, first_chunk_frames, chunk_frames, ...).
            warmup: Run a short generation on the worker thread at init.
                PyTorch CUDA state (cuBLAS handles, ...) is per-thread, so an
                unwarmed thread costs ~35 ms extra on its first utterance.
            settings: Runtime-updatable settings; values here win over the
                speaker/language args.
            **kwargs: Forwarded to the parent TTSService.
        """
        default_settings = self.Settings(model=None, voice=speaker, language=language)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._engine = engine if engine is not None else MegakernelTTS()
        self._engine_lock = engine_lock or threading.Lock()
        # first_chunk_frames=2 keeps TTFB under 90 ms (engine default of 4
        # measures ~100 ms through the pipeline); callers can override.
        self._gen_params = {"first_chunk_frames": 2, **(gen_params or {})}

        # One persistent generation thread: serializes utterances and keeps
        # per-thread CUDA state warm across them (a fresh thread per utterance
        # costs ~35 ms of cuBLAS/thread-local init on the first kernels).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="megakernel-tts")
        if warmup:
            self._executor.submit(self._warm_thread).result()

    def _warm_thread(self):
        fcf = {k: v for k, v in self._gen_params.items() if k == "first_chunk_frames"}
        with self._engine_lock:
            for _ in self._engine.stream("Warm up.", speaker=self._settings.voice or "ryan",
                                         max_new_tokens=30, **fcf):
                pass

    async def cleanup(self):
        """Shut down the generation thread."""
        await super().cleanup()
        self._executor.shutdown(wait=False)

    def can_generate_metrics(self) -> bool:
        """TTFB and usage metrics are supported."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Map a Pipecat Language to a Qwen3-TTS language name."""
        return _LANGUAGES.get(str(language.value).split("-")[0].lower())

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize text, yielding TTSAudioRawFrames as chunks decode.

        Args:
            text: The text to synthesize.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        _DONE = object()

        def worker():
            try:
                with self._engine_lock:
                    for audio, _sr, _timing in self._engine.stream(
                        text,
                        speaker=self._settings.voice or "ryan",
                        language=self._settings.language or "English",
                        **self._gen_params,
                    ):
                        if cancelled.is_set():
                            break
                        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                        loop.call_soon_threadsafe(queue.put_nowait, pcm)
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)
            except Exception as e:  # surfaced as ErrorFrame on the asyncio side
                logger.exception(f"{self} generation error")
                loop.call_soon_threadsafe(queue.put_nowait, e)

        async def pcm_chunks() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item

        self._executor.submit(worker)
        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(
                pcm_chunks(),
                in_sample_rate=self._engine.sample_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            yield ErrorFrame(error=f"{self} error: {e}")
        finally:
            # On interruption the worker stops at the next chunk boundary,
            # releasing the engine lock for the next utterance.
            cancelled.set()
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
