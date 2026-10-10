"""UniPC golden fixture generator.

Runs the OFFICIAL `FlowUniPCMultistepScheduler` (Wan's, as HiDream-ai ships it
in HiDream-O1-Image's `models/fm_solvers_unipc.py`) the way HiDream O1's
`models/pipeline.py` drives it for the full checkpoint — `shift` 3, the
timestep an integer, the flow `(x̂ − z) / σ` from that integer's level, the
scheduler given its negative — over a small synthetic denoiser, and writes
every step's sample: scheduler mathematics only, no model.

    HIDREAM_O1_OFFICIAL=~/dev/hidream-o1-reference \\
    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers --with numpy \\
        python runner/fixtures/unipc.py

Output is committed as runner/test/resources/fixtures/unipc.json.
"""

import importlib.util
import json
import os
from pathlib import Path

import torch

official = Path(
    os.environ.get("HIDREAM_O1_OFFICIAL", "~/dev/hidream-o1-reference")
).expanduser()
spec = importlib.util.spec_from_file_location(
    "fm_solvers_unipc", official / "models" / "fm_solvers_unipc.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

out = (
    Path(__file__).resolve().parent.parent
    / "test"
    / "resources"
    / "fixtures"
    / "unipc.json"
)

T_EPS = 0.001  # pipeline.py's floor under the flow's division


def denoiser(z, t):
    """A clean image that depends on the sample and on the timestep."""
    return 0.9 * torch.tanh(0.7 * z + 0.3 * t)


def run(steps, shift, start):
    scheduler = module.FlowUniPCMultistepScheduler(
        use_dynamic_shifting=False, shift=shift
    )
    scheduler.set_timesteps(steps, device="cpu")
    z = torch.tensor([start], dtype=torch.float32)
    samples = []
    for step_t in scheduler.timesteps:
        t = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).clamp_min(T_EPS)
        flow = (denoiser(z, t) - z) / sigma
        z = scheduler.step(-flow, step_t.to(torch.float32), z, return_dict=False)[0]
        samples.append(z[0].tolist())
    return {
        "steps": steps,
        "shift": shift,
        "sigmas": scheduler.sigmas.tolist(),
        "timesteps": scheduler.timesteps.tolist(),
        "start": start,
        "samples": samples,
    }


start = [8.0 * v for v in [-1.3, -0.4, 0.05, 0.6, 1.1, 2.2]]
cases = [run(50, 3.0, start), run(8, 3.0, start), run(3, 1.0, start), run(1, 3.0, start)]
out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
print(f"wrote {out}")
