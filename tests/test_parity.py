"""Parity check: megakernel talker decode vs HF talker forward.

Runs the same prefill, then feeds the same sequence of input embeddings to
both the HF inner talker model (DynamicCache, sdpa) and the megakernel,
comparing post-final-norm hidden states and codec-head logits per step.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts_megakernel.engine import MegakernelTTS


def main():
    torch.manual_seed(0)
    eng = MegakernelTTS()
    talker = eng.talker
    mk = eng.mk

    tie, tth, tpe = eng._build_inputs("Hello there, this is a parity test.", "ryan", "English")
    print(f"prefill embeds: {tie.shape}")

    with torch.inference_mode():
        talker.rope_deltas = None
        out = talker.forward(
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
        prefill_len = mk.load_prefill_cache(out.past_key_values)
        print(f"prefill_len = {prefill_len}, rope_deltas = {talker.rope_deltas}")

        hf_cache = out.past_key_values
        codec_embed = talker.get_input_embeddings()

        n_steps = 24
        gen_step = out.generation_step
        max_h, max_l = 0.0, 0.0
        tok_mismatch = 0
        for s in range(n_steps):
            tok = torch.tensor([[100 + 7 * s]], device="cuda")
            emb = codec_embed(tok)
            if gen_step < tth.shape[1]:
                emb = emb + tth[:, gen_step].unsqueeze(1)
            else:
                emb = emb + tpe
            gen_step += 1

            pos = prefill_len + s
            cache_position = torch.tensor([pos], device="cuda")
            hf_out = talker.model(
                inputs_embeds=emb,
                past_key_values=hf_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            hf_hidden = hf_out.last_hidden_state[0, -1].float()
            hf_logits = talker.codec_head(hf_out.last_hidden_state[0, -1]).float()

            mk_hidden = mk.step(emb.view(-1)).clone()
            mk_logits = mk.logits_from_hidden(mk_hidden)[0]

            dh = (hf_hidden - mk_hidden).abs().max().item()
            rel = dh / hf_hidden.abs().max().item()
            dl = (hf_logits - mk_logits).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(hf_hidden, mk_hidden, dim=0).item()
            argmax_match = hf_logits.argmax().item() == mk_logits.argmax().item()
            tok_mismatch += (not argmax_match)
            max_h, max_l = max(max_h, dh), max(max_l, dl)
            if s < 5 or not argmax_match:
                print(f"step {s:3d}: hidden maxdiff {dh:.4f} (rel {rel:.4f}) cos {cos:.6f} "
                      f"logit maxdiff {dl:.4f} argmax {'OK' if argmax_match else 'MISMATCH'} "
                      f"(hf={hf_logits.argmax().item()} mk={mk_logits.argmax().item()})")

        print(f"\nover {n_steps} steps: max hidden diff {max_h:.4f}, max logit diff {max_l:.4f}, "
              f"argmax mismatches {tok_mismatch}/{n_steps}")
        assert tok_mismatch == 0, "greedy token mismatch"
        print("PARITY OK")


if __name__ == "__main__":
    main()
