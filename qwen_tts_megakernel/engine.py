"""Streaming Qwen3-TTS engine with the megakernel as the talker decode backend.

Pipeline per utterance:
  1. Prompt build + prefill: qwen_tts builds the talker input embeddings
     (system/role tokens, speaker embed, language/think codec prefix, text);
     one HF forward produces the prefill KV cache, first-token logits, and the
     code-predictor conditioning hidden.
  2. Decode loop: each codec frame = one megakernel step (28-layer talker) +
     top-k sampling over the 3072-entry codec head + a CUDA-graphed 5-layer
     code predictor producing the remaining 15 codebook groups.
  3. Codec decode: every `chunk_frames` frames, the 12Hz speech tokenizer
     decodes accumulated codes (sliding 25-frame context once calibrated) and
     the new audio samples are yielded immediately — no full-utterance buffering.
"""

import logging
import time
from typing import Generator, Optional, Tuple

import numpy as np
import torch

from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, sample_logits
from .talker_megakernel import TalkerMegakernel

logger = logging.getLogger(__name__)


class MegakernelTTS:
    """Qwen3-TTS-12Hz-0.6B CustomVoice synthesis with megakernel talker decode."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        max_seq_len: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        temperature: float = 0.9,
    ):
        from qwen_tts import Qwen3TTSModel

        logger.info("Loading %s ...", model_name)
        self.base = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        m = self.base.model
        self.m = m
        self.talker = m.talker
        self.config = m.config.talker_config
        self.speech_tokenizer = m.speech_tokenizer
        self.sample_rate = int(getattr(self.speech_tokenizer, "sample_rate", 24000))

        logger.info("Building megakernel talker decoder ...")
        self.mk = TalkerMegakernel(self.talker, max_seq_len=max_seq_len)

        logger.info("Capturing code-predictor CUDA graph ...")
        self.predictor_graph = PredictorGraph(
            self.talker.code_predictor,
            self.talker.code_predictor.model.config,
            self.config.hidden_size,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
        )
        self.predictor_graph.capture(num_warmup=3)

        # Suppress special codec ids (top 1024 of vocab) except EOS, like upstream.
        vocab = self.config.vocab_size
        eos = self.config.codec_eos_token_id
        self.eos_id = eos
        self.suppress_mask = torch.zeros(vocab, dtype=torch.bool, device="cuda")
        self.suppress_mask[vocab - 1024:] = True
        self.suppress_mask[eos] = False

    # ------------------------------------------------------------------ #
    # Prompt building (vendored from qwen_tts / faster-qwen3-tts, batch=1,
    # custom-voice path)
    # ------------------------------------------------------------------ #
    def _build_inputs(self, text: str, speaker: str, language: str,
                      instruct: Optional[str] = None):
        m = self.m
        cfg = m.config
        tcfg = self.config

        input_id = self.base._tokenize_texts([self.base._build_assistant_text(text)])[0]

        instruct_embed = None
        if instruct:
            iid = self.base._tokenize_texts([self.base._build_instruct_text(instruct)])[0]
            instruct_embed = m.talker.text_projection(m.talker.get_text_embeddings()(iid))

        spk_id = tcfg.spk_id[speaker.lower()]
        speaker_embed = m.talker.get_input_embeddings()(
            torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
        )

        if language.lower() == "auto":
            language_id = None
        else:
            language_id = tcfg.codec_language_id[language.lower()]
        if tcfg.spk_is_dialect.get(speaker.lower()) and language.lower() in ("chinese", "auto"):
            language_id = tcfg.codec_language_id[tcfg.spk_is_dialect[speaker.lower()]]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
            m.talker.get_text_embeddings()(
                torch.tensor(
                    [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                    device=m.talker.device, dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill = [[tcfg.codec_nothink_id, tcfg.codec_think_bos_id,
                              tcfg.codec_think_eos_id]]
        else:
            codec_prefill = [[tcfg.codec_think_id, tcfg.codec_think_bos_id,
                              language_id, tcfg.codec_think_eos_id]]

        emb0 = m.talker.get_input_embeddings()(
            torch.tensor(codec_prefill, device=m.talker.device, dtype=input_id.dtype)
        )
        emb1 = m.talker.get_input_embeddings()(
            torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]],
                         device=m.talker.device, dtype=input_id.dtype)
        )
        codec_embedding = torch.cat([emb0, speaker_embed.view(1, 1, -1), emb1], dim=1)

        role_embed = m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, :3]))
        talker_input_embed = torch.cat(
            (tts_pad_embed.expand(-1, codec_embedding.shape[1] - 2, -1), tts_bos_embed),
            dim=1,
        ) + codec_embedding[:, :-1]
        talker_input_embed = torch.cat((role_embed, talker_input_embed), dim=1)

        # Streaming text-feeding layout (non_streaming_mode=False): first text
        # token rides on codec_bos; the rest arrive one per decode step.
        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, 3:4]))
                + codec_embedding[:, -1:],
            ],
            dim=1,
        )
        trailing_text_hidden = torch.cat(
            (
                m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, 4:-5])),
                tts_eos_embed,
            ),
            dim=1,
        )

        if instruct_embed is not None:
            talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    # ------------------------------------------------------------------ #
    # Streaming generation
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def stream(
        self,
        text: str,
        speaker: str = "ryan",
        language: str = "English",
        instruct: Optional[str] = None,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        first_chunk_frames: int = 4,
        chunk_frames: int = 12,
        context_frames: int = 25,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        """Yield (audio_chunk float32, sample_rate, timing) as frames are decoded."""
        t0 = time.perf_counter()

        tie, tth, tpe = self._build_inputs(text, speaker, language, instruct)

        # --- Prefill through HF forward (variable length, runs once) ---
        self.talker.rope_deltas = None
        out = self.talker.forward(
            inputs_embeds=tie,
            attention_mask=torch.ones(tie.shape[:2], dtype=torch.long, device=tie.device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        rope_deltas = getattr(self.talker, "rope_deltas", None)
        if rope_deltas is not None and int(rope_deltas.flatten()[0]) != 0:
            raise RuntimeError(f"Unexpected nonzero rope_deltas: {rope_deltas}")

        prefill_len = self.mk.load_prefill_cache(out.past_key_values)
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        logits = out.logits[:, -1, :].float()
        token = sample_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, suppress_mask=self.suppress_mask,
            suppress_tokens=[self.eos_id] if min_new_tokens > 0 else None,
        )
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        codec_embed = self.talker.get_input_embeddings()
        predictor_codec_embeds = self.talker.code_predictor.model.codec_embedding
        num_groups = self.config.num_code_groups

        all_codes: list[torch.Tensor] = []
        first_tokens: list[torch.Tensor] = []
        chunk_buffer: list[torch.Tensor] = []
        emitted_frames = 0
        prev_audio_len = 0
        samples_per_frame = None
        next_chunk = first_chunk_frames
        chunk_idx = 0
        t_decode_start = time.perf_counter()
        steps_done = 0

        def decode_chunk(is_final: bool):
            nonlocal prev_audio_len, samples_per_frame, emitted_frames, chunk_idx
            all_flat = torch.stack(all_codes, dim=0)  # [n, 16]
            n_total = all_flat.shape[0]
            n_new = n_total - emitted_frames
            if n_new <= 0:
                return None
            if samples_per_frame is None:
                audio_list, sr = self.speech_tokenizer.decode(
                    {"audio_codes": all_flat.unsqueeze(0)}
                )
                audio = audio_list[0]
                audio = audio.flatten().float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio).flatten()
                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)
                if n_total >= max(context_frames, chunk_frames):
                    samples_per_frame = len(audio) / n_total
            else:
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new
                audio_list, sr = self.speech_tokenizer.decode(
                    {"audio_codes": window.unsqueeze(0)}
                )
                audio = audio_list[0]
                audio = audio.flatten().float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio).flatten()
                new_audio = audio[int(round(n_ctx * samples_per_frame)):] if n_ctx > 0 else audio
            emitted_frames = n_total
            chunk_idx += 1
            return new_audio, sr

        for step_idx in range(max_new_tokens):
            if token.item() == self.eos_id:
                break

            # Code predictor: 15 remaining codebook groups (CUDA graph)
            last_id_hidden = codec_embed(token.unsqueeze(1))
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            cb_tokens = self.predictor_graph.run(pred_input)

            all_codes.append(torch.cat([token.view(1), cb_tokens]))
            first_tokens.append(token.view(1))
            chunk_buffer.append(all_codes[-1])

            # Next talker input embedding: sum of all 16 group embeddings (+text)
            embeds = last_id_hidden
            for i in range(num_groups - 1):
                embeds = embeds + predictor_codec_embeds[i](cb_tokens[i].view(1, 1))
            if gen_step < tth.shape[1]:
                embeds = embeds + tth[:, gen_step].unsqueeze(1)
            else:
                embeds = embeds + tpe

            # Talker megakernel step
            hidden = self.mk.step(embeds.view(-1))
            logits = self.mk.logits_from_hidden(hidden)

            if repetition_penalty != 1.0 and first_tokens:
                history = torch.cat(first_tokens)
                logits = apply_repetition_penalty(logits, history, repetition_penalty)
            token = sample_logits(
                logits, temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, suppress_mask=self.suppress_mask,
                suppress_tokens=[self.eos_id] if len(first_tokens) < min_new_tokens else None,
            )
            past_hidden = hidden.to(torch.bfloat16).view(1, 1, -1).clone()
            gen_step += 1
            steps_done += 1

            if len(all_codes) >= next_chunk:
                res = decode_chunk(is_final=False)
                if res is not None:
                    audio, sr = res
                    now = time.perf_counter()
                    yield audio, sr, {
                        "chunk_index": chunk_idx - 1,
                        "prefill_ms": t_prefill * 1000,
                        "elapsed_ms": (now - t0) * 1000,
                        "decode_ms": (now - t_decode_start) * 1000,
                        "frames": emitted_frames,
                        "is_final": False,
                    }
                next_chunk = emitted_frames + chunk_frames

        if len(all_codes) > emitted_frames:
            res = decode_chunk(is_final=True)
            if res is not None:
                audio, sr = res
                now = time.perf_counter()
                yield audio, sr, {
                    "chunk_index": chunk_idx - 1,
                    "prefill_ms": t_prefill * 1000,
                    "elapsed_ms": (now - t0) * 1000,
                    "decode_ms": (now - t_decode_start) * 1000,
                    "frames": emitted_frames,
                    "is_final": True,
                }

    @torch.inference_mode()
    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int, dict]:
        """Non-streaming convenience: concatenate all streamed chunks."""
        chunks = []
        sr = self.sample_rate
        last_timing = {}
        for audio, sr, timing in self.stream(text, **kwargs):
            chunks.append(audio)
            last_timing = timing
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return audio, sr, last_timing
