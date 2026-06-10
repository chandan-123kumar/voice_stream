"""Megakernel-backed decode for the Qwen3-TTS talker.

The talker backbone is shape-identical to Qwen3-0.6B, so AlpinDale's persistent
decode kernel runs it unmodified, with three integration twists handled here:

1. The talker consumes *embeddings* (codec-embedding sums + projected text
   hiddens), not token ids. The kernel's first layer reads
   ``embed_weight + token_id * HIDDEN_SIZE``, so we point ``embed_weight`` at a
   single-row staging buffer, copy each step's input embedding into it, and
   always pass ``token_id=0``. Zero kernel changes.

2. RoPE uses theta=1e6 (vs 1e4 for the text model). The talker's interleaved
   M-RoPE selects per-frequency between three position streams that are all
   identical during TTS decode, which reduces exactly to standard rotate-half
   RoPE — the kernel's RoPE math applies as-is; only the cos/sin tables differ.

3. Prefill (variable-length prompt) runs through the HF talker forward once;
   its DynamicCache K/V (already RoPE'd) is copied into the kernel's
   [layer, kv_head, pos, head_dim] cache layout. Every subsequent decode step
   is a single megakernel launch.

The kernel's fused LM head runs over the 3072-entry codec head and produces a
free greedy argmax token; for top-k sampling we instead read the final-norm
hidden state (g_normalized) and do a trivial 1024x3072 matvec in torch.
"""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
NUM_Q_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
CODEC_VOCAB_SIZE = 3072


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack the 11-tensor-per-layer flat list into a device blob of LDGLayerWeights."""
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS * n_ptrs):
        struct.pack_into("Q", buf, i * ptr_size, layer_weights[i].data_ptr())
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TalkerMegakernel:
    """Single-token decode for the Qwen3-TTS talker via the persistent megakernel."""

    def __init__(self, talker, max_seq_len: int = 4096):
        """talker: qwen_tts Qwen3TTSTalkerForConditionalGeneration (bf16, cuda)."""
        from .build import get_extension

        get_extension()
        self._decode = torch.ops.qwen_tts_megakernel_C.decode

        config = talker.config
        assert config.hidden_size == HIDDEN_SIZE
        assert config.intermediate_size == INTERMEDIATE_SIZE
        assert config.num_hidden_layers == NUM_LAYERS
        assert config.num_attention_heads == NUM_Q_HEADS
        assert config.num_key_value_heads == NUM_KV_HEADS
        assert config.vocab_size == CODEC_VOCAB_SIZE

        self.max_seq_len = max_seq_len
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # --- Weights: reference the HF module's tensors directly (no copy) ---
        sd = talker.model.state_dict()
        names = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        self._layer_tensors = []
        for i in range(NUM_LAYERS):
            for n in names:
                t = sd[f"layers.{i}.{n}"]
                assert t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
                self._layer_tensors.append(t)
        self._layer_weights_packed = _pack_layer_weights(self._layer_tensors)
        self._final_norm_weight = sd["norm.weight"].contiguous()
        self._codec_head_weight = talker.codec_head.weight.contiguous()

        # --- RoPE tables at the talker's theta (1e6) ---
        theta = float(config.rope_theta)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
        self._sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

        # --- Embedding staging buffer: 1-row "embedding table", token id 0 ---
        self._embed_staging = torch.zeros(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        # --- KV cache in kernel layout ---
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # --- Scratch buffers ---
        f32 = dict(dtype=torch.float32, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self.norm_out = torch.empty(HIDDEN_SIZE, **f32)  # post-final-norm hidden
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self.greedy_token = torch.empty(1, dtype=torch.int32, device="cuda")

        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def load_prefill_cache(self, past_key_values) -> int:
        """Copy a HF DynamicCache (post-RoPE K) into the kernel cache layout.

        Returns the prefill length; subsequent step() calls decode from there.
        """
        seq_len = 0
        for li in range(NUM_LAYERS):
            k, v = past_key_values[li]  # [1, kv_heads, seq, head_dim]
            seq_len = k.shape[2]
            if seq_len + 1 >= self.max_seq_len:
                raise RuntimeError(
                    f"Prefill of {seq_len} tokens exceeds max_seq_len={self.max_seq_len}"
                )
            self._k_cache[li, :, :seq_len].copy_(k[0])
            self._v_cache[li, :, :seq_len].copy_(v[0])
        self._position = seq_len
        return seq_len

    def reset(self):
        self._position = 0

    def step(self, input_embed: torch.Tensor) -> torch.Tensor:
        """One talker decode step from an input embedding.

        input_embed: [HIDDEN_SIZE] or [1, 1, HIDDEN_SIZE] bf16 cuda tensor.
        Returns the post-final-norm hidden state (f32, [HIDDEN_SIZE], a view of
        an internal buffer — consume before the next step). The greedy codec
        token from the fused LM head is left in self.greedy_token.
        """
        if self._position >= self.max_seq_len - 1:
            raise RuntimeError("KV cache full")
        self._embed_staging.view(-1).copy_(input_embed.view(-1))
        self._decode(
            self.greedy_token,
            0,  # token id 0 → reads row 0 of the staging "table"
            self._embed_staging,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._codec_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self.norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            self.max_seq_len,
            self._attn_scale,
        )
        self._position += 1
        return self.norm_out

    def logits_from_hidden(self, hidden_f32: torch.Tensor) -> torch.Tensor:
        """Codec-head logits [1, 3072] from the post-norm hidden state."""
        h = hidden_f32.to(torch.bfloat16)
        return torch.mv(self._codec_head_weight, h).float().unsqueeze(0)
