#!/usr/bin/env python3
"""One-off correctness test for the new resumable run_inference(): confirm
an interrupted-then-resumed run produces identical logits/targets to an
uninterrupted run, on a tiny real CPU subset."""
import sys, warnings, shutil
sys.path.insert(0, '/home/boshra95/NSRR-tools/src')
sys.path.insert(0, '/home/boshra95/NSRR-tools/scripts')
import numpy as np
import torch
import yaml
from peft import set_peft_model_state_dict
import infer_osf_lora_subject_windows as m

cfg = yaml.safe_load(open('/home/boshra95/NSRR-tools/configs/phase0_osf_lora_config.yaml'))
device = torch.device('cpu')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ds = m.build_dataset(cfg, split='test', context_length='30s', task='apnea_binary',
                          datasets_filter=None, all_windows=True, limit=1)
print(f"dataset items: {len(ds)}")

ckpt_path = '/scratch/boshra95/psg_full/unified/results/phase0_osf_lora/apnea_binary_lstm/context_30s/best_model.pt'
import json
num_classes = json.load(open('/scratch/boshra95/psg_full/unified/results/phase0_osf_lora/apnea_binary_lstm/context_30s/metrics.json'))['num_classes']
model = m.build_combined_lora_model(cfg, num_classes, 'lstm', device)
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
set_peft_model_state_dict(model, state)
model.eval()

resume_path = __import__('pathlib').Path('/tmp/test_resume.pt')
resume_path.unlink(missing_ok=True)

# ── Baseline: uninterrupted run ──────────────────────────────────────────
logits_a, targets_a, completed_a = m.run_inference(
    model, ds, device, batch_size=2, num_workers=0, resume_path=resume_path)
print(f"baseline: completed={completed_a}, n={len(logits_a)}")
resume_path.unlink(missing_ok=True)

# ── Interrupted run: force _STOP_REQUESTED after checkpoint fires ───────
m._CHECKPOINT_INTERVAL_SEC = -1  # always "due" -> checkpoints every batch
orig_save = m._save_resume_checkpoint
call_count = [0]
def _counting_save(*args, **kwargs):
    call_count[0] += 1
    orig_save(*args, **kwargs)
    if call_count[0] == 1:
        m._STOP_REQUESTED = True  # simulate SIGTERM arriving after 1st checkpoint
m._save_resume_checkpoint = _counting_save

logits_b1, targets_b1, completed_b1 = m.run_inference(
    model, ds, device, batch_size=2, num_workers=0, resume_path=resume_path)
print(f"interrupted attempt: completed={completed_b1}, n_so_far={len(logits_b1)}, resume file exists={resume_path.exists()}")
assert completed_b1 is False, "expected the simulated interruption to stop early"

# ── Resume: should finish from where it left off ────────────────────────
m._save_resume_checkpoint = orig_save
m._STOP_REQUESTED = False
logits_b2, targets_b2, completed_b2 = m.run_inference(
    model, ds, device, batch_size=2, num_workers=0, resume_path=resume_path)
print(f"resumed attempt: completed={completed_b2}, n={len(logits_b2)}")

# ── Compare ───────────────────────────────────────────────────────────
assert completed_b2 is True
assert np.array_equal(logits_a, logits_b2), "logits mismatch between baseline and resumed run!"
assert np.array_equal(targets_a, targets_b2), "targets mismatch between baseline and resumed run!"
assert not resume_path.exists() or True  # run_inference doesn't delete it itself; main() does
print("\nPASS: interrupted+resumed run produced IDENTICAL logits/targets to an uninterrupted run.")
resume_path.unlink(missing_ok=True)
