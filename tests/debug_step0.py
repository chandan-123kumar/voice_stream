import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

torch.manual_seed(0)
eng = MegakernelTTS()
talker = eng.talker
mk = eng.mk

print("q_proj shape:", talker.model.layers[0].self_attn.q_proj.weight.shape)
print("k_proj shape:", talker.model.layers[0].self_attn.k_proj.weight.shape)
print("config head_dim:", getattr(talker.config, "head_dim", None))
print("attn head_dim:", talker.model.layers[0].self_attn.head_dim)
print("rotary inv_freq[:4]:", talker.model.rotary_emb.inv_freq[:4])
print("attention_scaling:", talker.model.rotary_emb.attention_scaling)
print("mrope:", talker.config.rope_scaling)

with torch.inference_mode():
    emb = talker.get_input_embeddings()(torch.tensor([[123]], device="cuda"))

    # HF single step, fresh cache, position 0
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    hf_out = talker.model(
        inputs_embeds=emb,
        past_key_values=cache,
        cache_position=torch.tensor([0], device="cuda"),
        use_cache=True,
    )
    hf_h = hf_out.last_hidden_state[0, -1].float()

    # Megakernel from scratch, position 0
    mk.reset()
    mk_h = mk.step(emb.view(-1)).clone()

    d = (hf_h - mk_h).abs()
    cos = torch.nn.functional.cosine_similarity(hf_h, mk_h, dim=0).item()
    print(f"\nstep0 no-prefill: maxdiff {d.max().item():.4f}  cos {cos:.6f}")
    print("hf_h[:8]:", hf_h[:8].tolist())
    print("mk_h[:8]:", mk_h[:8].tolist())

    # Also single-layer comparison (num_layers=1), pre-final-norm hidden
    import qwen_tts_megakernel.talker_megakernel as tm
    mk.reset()
    mk._decode(
        mk.greedy_token, 0, mk._embed_staging, mk._layer_weights_packed,
        mk._final_norm_weight, mk._codec_head_weight, mk._cos_table, mk._sin_table,
        mk._k_cache, mk._v_cache, mk._hidden, mk._act, mk._res, mk._q, mk._k, mk._v,
        mk._attn_out, mk._mlp_inter, mk.norm_out, mk._bmax_vals, mk._bmax_idxs,
        1, 0, mk.max_seq_len, mk._attn_scale,
    )
    mk._embed_staging.view(-1).copy_(emb.view(-1))
    mk._decode(
        mk.greedy_token, 0, mk._embed_staging, mk._layer_weights_packed,
        mk._final_norm_weight, mk._codec_head_weight, mk._cos_table, mk._sin_table,
        mk._k_cache, mk._v_cache, mk._hidden, mk._act, mk._res, mk._q, mk._k, mk._v,
        mk._attn_out, mk._mlp_inter, mk.norm_out, mk._bmax_vals, mk._bmax_idxs,
        1, 0, mk.max_seq_len, mk._attn_scale,
    )
    layer0_mk = mk._hidden.float()

    # HF layer 0 only
    cache2 = DynamicCache()
    pos_ids = torch.zeros(3, 1, 1, dtype=torch.long, device="cuda")
    cos, sin = talker.model.rotary_emb(emb, pos_ids)
    l0 = talker.model.layers[0]
    h = l0(
        emb,
        attention_mask=None,
        position_ids=pos_ids[0],
        past_key_values=cache2,
        use_cache=True,
        cache_position=torch.tensor([0], device="cuda"),
        position_embeddings=(cos, sin),
    )[0][0, -1].float()
    d0 = (h - layer0_mk).abs().max().item()
    c0 = torch.nn.functional.cosine_similarity(h, layer0_mk, dim=0).item()
    print(f"layer0-only: maxdiff {d0:.4f} cos {c0:.6f}")
    print("hf  l0[:8]:", h[:8].tolist())
    print("mk  l0[:8]:", layer0_mk[:8].tolist())
