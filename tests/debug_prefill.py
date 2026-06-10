import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

torch.manual_seed(0)
eng = MegakernelTTS()
talker = eng.talker
mk = eng.mk

with torch.inference_mode():
    tie, tth, tpe = eng._build_inputs("Hello there, this is a parity test.", "ryan", "English")
    talker.rope_deltas = None
    out = talker.forward(
        inputs_embeds=tie,
        attention_mask=torch.ones(tie.shape[:2], dtype=torch.long, device=tie.device),
        use_cache=True, output_hidden_states=True, return_dict=True,
        trailing_text_hidden=tth, tts_pad_embed=tpe,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    L = mk.load_prefill_cache(out.past_key_values)
    hf_cache = out.past_key_values

    k0, v0 = hf_cache[0]
    print("hf k0 shape:", k0.shape, "dtype", k0.dtype)
    print("cache copy check k:", (mk._k_cache[0, :, :L] - k0[0]).abs().max().item())
    print("cache copy check v:", (mk._v_cache[0, :, :L] - v0[0]).abs().max().item())

    emb = talker.get_input_embeddings()(torch.tensor([[123]], device="cuda"))

    # --- megakernel: 1 layer only, position L ---
    mk._embed_staging.view(-1).copy_(emb.view(-1))
    mk._decode(
        mk.greedy_token, 0, mk._embed_staging, mk._layer_weights_packed,
        mk._final_norm_weight, mk._codec_head_weight, mk._cos_table, mk._sin_table,
        mk._k_cache, mk._v_cache, mk._hidden, mk._act, mk._res, mk._q, mk._k, mk._v,
        mk._attn_out, mk._mlp_inter, mk.norm_out, mk._bmax_vals, mk._bmax_idxs,
        1, L, mk.max_seq_len, mk._attn_scale,
    )
    torch.cuda.synchronize()
    mk_l0 = mk._hidden.float().clone()
    mk_attn = mk._attn_out.clone()      # f32 [2048] — layer0 attn (pre-O-proj)
    mk_q = mk._q.clone()                # f32 [2048] — POST norm+rope (kernel overwrites q in-place)
    mk_k = mk._k.clone()                # f32 [1024] — pre-norm k? kernel writes k_cache normed
    kc_new = mk._k_cache[0, :, L].clone()  # the K the kernel just wrote at pos L

    # --- HF: layer 0 with the same cache ---
    pos_ids = torch.full((3, 1, 1), L, dtype=torch.long, device="cuda")
    cos, sin = talker.model.rotary_emb(emb, pos_ids)
    sa = talker.model.layers[0].self_attn

    h_in = talker.model.layers[0].input_layernorm(emb)
    qs = sa.q_norm(sa.q_proj(h_in).view(1, 1, 16, 128)).transpose(1, 2)
    ks = sa.k_norm(sa.k_proj(h_in).view(1, 1, 8, 128)).transpose(1, 2)
    from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
    qs_r, ks_r = apply_multimodal_rotary_pos_emb(
        qs, ks, cos, sin, sa.rope_scaling["mrope_section"], sa.rope_scaling["interleaved"]
    )
    print("\nq rope: maxdiff", (qs_r.reshape(-1).float() - mk_q).abs().max().item())
    print("k rope vs kernel-written cache:", (ks_r.reshape(8, 128).float() - kc_new.float()).abs().max().item())

    # full HF layer-0 forward with cache
    from transformers.cache_utils import DynamicCache
    cache_l0 = DynamicCache()
    cache_l0.update(k0, v0, 0, {})
    lay = talker.model.layers[0]
    hf_l0 = lay(
        emb, attention_mask=None, position_ids=pos_ids[0],
        past_key_values=cache_l0, use_cache=True,
        cache_position=torch.tensor([L], device="cuda"),
        position_embeddings=(cos, sin),
    )[0][0, -1].float()
    print("layer0+prefill: maxdiff", (hf_l0 - mk_l0).abs().max().item(),
          "cos", torch.nn.functional.cosine_similarity(hf_l0, mk_l0, dim=0).item())

    # --- Full 28 layers with prefill: inner model vs megakernel ---
    mk._position = L  # reset position back (we already wrote pos L above, will overwrite)
    cache_full = DynamicCache()
    for li in range(28):
        kl, vl = hf_cache[li]
        cache_full.update(kl[:, :, :L].contiguous(), vl[:, :, :L].contiguous(), li, {})
    hf_full = talker.model(
        inputs_embeds=emb,
        past_key_values=cache_full,
        cache_position=torch.tensor([L], device="cuda"),
        use_cache=True,
    ).last_hidden_state[0, -1].float()
    mk_full = mk.step(emb.view(-1)).clone()
    print("full28+prefill: maxdiff", (hf_full - mk_full).abs().max().item(),
          "cos", torch.nn.functional.cosine_similarity(hf_full, mk_full, dim=0).item())

    # --- Same but with the realistic input embedding (codec + text hidden) ---
    emb2 = talker.get_input_embeddings()(torch.tensor([[107]], device="cuda")) + tth[:, 0].unsqueeze(1)
    print("emb2 stats: max", emb2.abs().max().item(), "norm", emb2.float().norm().item())
    print("emb  stats: max", emb.abs().max().item(), "norm", emb.float().norm().item())
    mk._position = L
    cache_full2 = DynamicCache()
    for li in range(28):
        kl, vl = hf_cache[li]
        cache_full2.update(kl[:, :, :L].contiguous(), vl[:, :, :L].contiguous(), li, {})
    hf_full2 = talker.model(
        inputs_embeds=emb2,
        past_key_values=cache_full2,
        cache_position=torch.tensor([L], device="cuda"),
        use_cache=True,
    ).last_hidden_state[0, -1].float()
    mk_full2 = mk.step(emb2.view(-1)).clone()
    print("full28+prefill+texthidden: maxdiff", (hf_full2 - mk_full2).abs().max().item(),
          "cos", torch.nn.functional.cosine_similarity(hf_full2, mk_full2, dim=0).item())
    # where do they diverge? compare per-layer by limiting num_layers
    for nl in (1, 7, 14, 21, 28):
        mk._embed_staging.view(-1).copy_(emb2.view(-1))
        mk._decode(
            mk.greedy_token, 0, mk._embed_staging, mk._layer_weights_packed,
            mk._final_norm_weight, mk._codec_head_weight, mk._cos_table, mk._sin_table,
            mk._k_cache, mk._v_cache, mk._hidden, mk._act, mk._res, mk._q, mk._k, mk._v,
            mk._attn_out, mk._mlp_inter, mk.norm_out, mk._bmax_vals, mk._bmax_idxs,
            nl, L, mk.max_seq_len, mk._attn_scale,
        )
        torch.cuda.synchronize()
        mk_h_nl = mk._hidden.float().clone()
        # HF partial
        cache_p = DynamicCache()
        for li in range(28):
            kl, vl = hf_cache[li]
            cache_p.update(kl[:, :, :L].contiguous(), vl[:, :, :L].contiguous(), li, {})
        pos_ids3 = torch.full((3, 1, 1), L, dtype=torch.long, device="cuda")
        cs, sn = talker.model.rotary_emb(emb2, pos_ids3)
        h = emb2
        for li in range(nl):
            h = talker.model.layers[li](
                h, attention_mask=None, position_ids=pos_ids3[0],
                past_key_values=cache_p, use_cache=True,
                cache_position=torch.tensor([L], device="cuda"),
                position_embeddings=(cs, sn),
            )[0]
        hf_h_nl = h[0, -1].float()
        print(f"  nl={nl:2d}: maxdiff {(hf_h_nl - mk_h_nl).abs().max().item():.4f} "
              f"cos {torch.nn.functional.cosine_similarity(hf_h_nl, mk_h_nl, dim=0).item():.6f} "
              f"|h| {hf_h_nl.abs().max().item():.2f}")

    # --- Noise floor: HF sdpa vs eager for the same step ---
    import copy
    cache_a = DynamicCache(); cache_b = DynamicCache()
    for li in range(28):
        kl, vl = hf_cache[li]
        cache_a.update(kl[:, :, :L].contiguous(), vl[:, :, :L].contiguous(), li, {})
        cache_b.update(kl[:, :, :L].contiguous(), vl[:, :, :L].contiguous(), li, {})
    h_sdpa = talker.model(
        inputs_embeds=emb2, past_key_values=cache_a,
        cache_position=torch.tensor([L], device="cuda"), use_cache=True,
    ).last_hidden_state[0, -1].float()
    talker.model.config._attn_implementation = "eager"
    h_eager = talker.model(
        inputs_embeds=emb2, past_key_values=cache_b,
        cache_position=torch.tensor([L], device="cuda"), use_cache=True,
    ).last_hidden_state[0, -1].float()
    talker.model.config._attn_implementation = "sdpa"
    print("NOISE FLOOR sdpa-vs-eager: maxdiff", (h_sdpa - h_eager).abs().max().item(),
          "cos", torch.nn.functional.cosine_similarity(h_sdpa, h_eager, dim=0).item())
    lg_s = talker.codec_head(h_sdpa.to(torch.bfloat16)).float()
    lg_m = mk.logits_from_hidden(mk_full2)[0]
    print("top5 sdpa:", lg_s.topk(5).indices.tolist())
    print("top5 mk:  ", lg_m.topk(5).indices.tolist())
