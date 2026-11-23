"""CPU unit test for the Qwen3-VL CP+THD packed mRoPE reconstruction (issue #1296).

Under context parallelism each rank's THD row holds only its zigzag chunks of every packed
segment. `_reassemble_full_row` de-interleaves the all-gathered per-rank rows back to the
full natural-order row so per-segment MRoPE positions can be rebuilt and re-sliced. This
test checks that reconstruction is the exact inverse of `slice_with_cp` (the function miles
uses to shard the tokens), and that re-slicing with `_natural_to_zigzag_slice` round-trips.
It also checks that the patched forward passes each rank its local MRoPE `position_ids`.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

import miles_plugins.models.qwen3_vl as qwen3_vl
from miles_plugins.models.qwen3_vl import _natural_to_zigzag_slice, _reassemble_full_row


def _slice_with_cp(tokens, cp_size, cp_rank, pad_value=0):
    """Reference copy of cp_utils.slice_with_cp's THD zigzag slicing (per sample)."""
    token_len = len(tokens)
    chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk - token_len
    if pad:
        tokens = F.pad(tokens, (0, pad), value=pad_value)
    s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
    s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
    return torch.cat([tokens[s1:e1], tokens[s2:e2]])


def _build_like_get_batch(sample_lens, cp_size, pad_size=8):
    """Mimic miles get_batch THD+CP packing: per-sample zigzag slice, concat, pad, cu*cp."""
    samples = []
    base = 1
    for L in sample_lens:
        samples.append(torch.arange(base, base + L))  # unique nonzero ids
        base += L
    per_rank = []
    for r in range(cp_size):
        row = torch.cat([_slice_with_cp(t, cp_size, r) for t in samples])
        per_rank.append(row)
    cu = [0]
    for t in samples:
        cu.append(cu[-1] + _slice_with_cp(t, cp_size, 0).size(0))
    final_pad = (pad_size - per_rank[0].size(0) % pad_size) % pad_size
    if final_pad:
        per_rank = [F.pad(row, (0, final_pad), value=0) for row in per_rank]
        cu.append(cu[-1] + final_pad)
    cu = [x * cp_size for x in cu]
    return samples, per_rank, cu


class _FakeQwen3VLModel:
    """Stands in for megatron-bridge 0.7's Qwen3VLModel, which rejects a CP-pre-sharded THD row
    without explicit rank-local MRoPE positions before it ever calls get_rope_index."""

    _cp_local_vision_embed_indices = None
    config = types.SimpleNamespace(spatial_merge_size=2)
    image_token_id, video_token_id, vision_start_token_id = -1, -2, -3
    cp_size = 1

    def forward(self, input_ids, position_ids=None, packed_seq_params=None, **kwargs):
        if int(packed_seq_params.cu_seqlens_q[-1]) == self.cp_size * input_ids.numel() and position_ids is None:
            raise ValueError("Pre-sharded packed CP inputs require explicit rank-local 3D MRoPE position_ids.")
        return position_ids


@pytest.mark.parametrize(
    "cp_size,sample_lens",
    [(2, [10, 7, 13]), (2, [16, 16]), (4, [20, 9, 30, 5]), (2, [3]), (4, [40, 17])],
)
def test_reassemble_is_inverse_of_slice_with_cp(cp_size, sample_lens):
    samples, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    local_len = per_rank[0].size(0)
    assert cu[-1] == cp_size * local_len

    full = _reassemble_full_row(per_rank, cu, cp_size)
    assert full is not None and full.numel() == cu[-1]

    # Each real sample's tokens reappear (in order) at the start of its segment.
    for i, t in enumerate(samples):
        seg = full[cu[i] : cu[i + 1]]
        assert torch.equal(seg[: t.numel()], t)

    # Re-slicing the full row per segment recovers exactly each rank's local chunks.
    for r in range(cp_size):
        recon = []
        for i in range(len(cu) - 1):
            recon.append(_natural_to_zigzag_slice(full[cu[i] : cu[i + 1]], cp_size, r, dim=0))
        assert torch.equal(torch.cat(recon), per_rank[r])


def test_reassemble_bails_on_indivisible_segment():
    # A segment length not divisible by 2*cp -> None (caller falls back to dense path).
    cu = [0, 6]  # 6 not divisible by 2*cp=4
    gathered = [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    assert _reassemble_full_row(gathered, cu, cp_size=2) is None


@pytest.mark.parametrize("cp_size,sample_lens", [(2, [10, 7, 13]), (4, [20, 9, 30, 5])])
def test_cp_forward_passes_rank_local_position_ids(monkeypatch, cp_size, sample_lens):
    module_name = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    fake_mod = types.ModuleType(module_name)
    fake_mod.Qwen3VLModel = type("Qwen3VLModel", (_FakeQwen3VLModel,), {"cp_size": cp_size})
    fake_mod.get_rope_index = lambda *args, **kwargs: (None, None)
    fake_mod.preprocess_packed_seqs = lambda input_ids, attention_mask, *args, **kwargs: (input_ids, None)
    monkeypatch.setitem(sys.modules, module_name, fake_mod)
    qwen3_vl._patch_model_forward_and_rope_index()

    _, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    monkeypatch.setattr(
        qwen3_vl, "_cp_allgather_unzigzag", lambda flat, cu, cp: _reassemble_full_row(per_rank, cu, cp)
    )
    psp = types.SimpleNamespace(qkv_format="thd", cu_seqlens_q=torch.tensor(cu))
    for rank in range(cp_size):
        monkeypatch.setattr(qwen3_vl, "_cp_size_rank", lambda rank=rank: (cp_size, rank))
        position_ids = fake_mod.Qwen3VLModel().forward(
            input_ids=per_rank[rank].view(1, -1), position_ids=None, packed_seq_params=psp
        )

        # Text-only segments restart at 0; each rank holds its zigzag chunks of every segment.
        expected = torch.cat(
            [_slice_with_cp(torch.arange(cu[i + 1] - cu[i]), cp_size, rank) for i in range(len(cu) - 1)]
        )
        assert torch.equal(position_ids, expected.view(1, 1, -1).expand(3, 1, -1))
