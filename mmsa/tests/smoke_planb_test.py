import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mmsa")

from mmsa.data.precompute_interpretation_video import _parse_structured_rationale

r1 = _parse_structured_rationale('{"face": "frowning", "content": "arguing", "bg": "noisy room", "bg_important": true}')
assert r1["face"] == "frowning" and r1["content"] == "arguing" and r1["bg_important"] is True, r1
r2 = _parse_structured_rationale('```json\n{"face": "smiling", "content": "talking", "bg": "park", "bg_important": false}\n```')
assert r2["face"] == "smiling" and r2["bg_important"] is False, r2
r3 = _parse_structured_rationale("The speaker looks happy and the background is a sunny beach.")
assert r3["content"].startswith("The speaker"), r3
r4 = _parse_structured_rationale("")
assert r4["content"], r4
print("parser OK")

import numpy as np
import shutil
from pathlib import Path
from mmsa.data.mmsa_pkl_dataset import _load_interpretation

root = Path("outputs/planb_smoke_cache")
shutil.rmtree(root, ignore_errors=True)
root.mkdir(parents=True, exist_ok=True)
np.save(root / "train_face.npy", np.random.rand(4, 8).astype(np.float32))
np.save(root / "train_content.npy", np.random.rand(4, 8).astype(np.float32))
np.save(root / "train_bg.npy", np.random.rand(4, 8).astype(np.float32))
np.save(root / "train_bg_flag.npy", np.array([[1], [0], [1], [0]], dtype=np.float32))
np.save(root / "train.npy", np.random.rand(4, 24).astype(np.float32))
legacy, channels = _load_interpretation(root, "train")
assert legacy is not None and legacy.shape == (4, 24), legacy.shape
assert set(channels) == {"face", "content", "bg", "bg_flag"}, channels.keys()
assert channels["bg_flag"].shape == (4, 1), channels["bg_flag"].shape
print("dataset loader OK:", {k: v.shape for k, v in channels.items()})

import torch
from mmsa.models.model import TeMoE

torch.manual_seed(0)
model = TeMoE(audio_dim=5, vision_dim=4, text_dim=6, interp_dim=24, channel_dim=8, d_model=16, num_experts=4, top_k=2)
batch = 3
audio = torch.randn(batch, 5, 5)
vision = torch.randn(batch, 5, 4)
text = torch.randn(batch, 5, 6)
face = torch.randn(batch, 8)
content = torch.randn(batch, 8)
bg = torch.randn(batch, 8)
flag = torch.rand(batch, 1)
out = model(audio, vision, text, interpretation=None, interp_face=face, interp_content=content, interp_bg=bg, interp_bg_flag=flag)
assert out["reg"].shape == (batch,), out["reg"].shape
assert out["cons_loss"].shape == (), out["cons_loss"].shape
assert model.last_bg_gate is not None and model.last_bg_gate.shape == (batch, 1)
assert model.last_cons_sims is not None and model.last_cons_sims.shape == (batch, 3)
print("model channels OK: reg", out["reg"].shape, "cons_loss", float(out["cons_loss"]), "bg_gate mean", float(model.last_bg_gate.mean()))

v_no_face = TeMoE(audio_dim=5, vision_dim=4, text_dim=6, interp_dim=24, channel_dim=8, d_model=16, use_face=False)
o = v_no_face(audio, vision, text, interp_face=face, interp_content=content, interp_bg=bg)
assert o["reg"].shape == (batch,)
print("no_face OK")

v_no_cons = TeMoE(audio_dim=5, vision_dim=4, text_dim=6, interp_dim=24, channel_dim=8, d_model=16, use_consistency=False)
o = v_no_cons(audio, vision, text, interp_face=face, interp_content=content, interp_bg=bg)
assert o["reg"].shape == (batch,) and float(o["cons_loss"]) == 0.0
print("no_cons OK")

v_no_bg = TeMoE(audio_dim=5, vision_dim=4, text_dim=6, interp_dim=24, channel_dim=8, d_model=16, use_bg_gate=False)
o = v_no_bg(audio, vision, text, interp_face=face, interp_content=content, interp_bg=bg)
assert v_no_bg.last_bg_gate is not None and float(v_no_bg.last_bg_gate.sum()) == 0.0
print("no_bg_gate OK")

# legacy path: single_channel variant consumes the concatenated interpretation vector
v_single = TeMoE(audio_dim=5, vision_dim=4, text_dim=6, interp_dim=24, channel_dim=8, d_model=16, use_channels=False)
o = v_single(audio, vision, text, interpretation=torch.randn(batch, 24))
assert o["reg"].shape == (batch,)
print("single_channel OK")

print("ALL SMOKE TESTS PASSED")
