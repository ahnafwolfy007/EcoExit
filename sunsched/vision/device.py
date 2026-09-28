"""Where the frozen trunk and the exit heads run.

Nothing reported depends on this choice. The one energy input measured from the
real network is its MAC count (`sunsched/vision/cost.py`), counted analytically
from layer shapes, and the node model multiplies that by an ASSUMED energy per
MAC. Shapes do not change with the device, so no published number does either.
This is the one place v2 differs from v1, which timed the network on a CPU and
fed those wall-clock timings straight into its energy model -- there the device
was a scientific choice, and here it is not.

So a GPU buys speed only, and only in stages 1 and 2. Stages 3-7 are numpy
simulation across worker processes and ignore this setting entirely.
"""
import torch

CHOICES = ("cpu", "cuda", "auto")


def resolve(name: str = "cpu") -> torch.device:
    """`auto` takes CUDA when it is available; `cuda` insists on it."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested but torch.cuda.is_available() is False.\n"
            "  Check which torch build this venv holds:\n"
            '    python -c "import torch; print(torch.__version__)"\n'
            "  A '+cpu' suffix means no CUDA support. Either use --device cpu, or\n"
            "  reinstall a CUDA build: pip install torch torchvision --index-url \\n"
            "    https://download.pytorch.org/whl/cu129")
    return torch.device(name)


def prepare(device: torch.device) -> str:
    """Configure the device and return one line describing it for the log.

    TF32 is turned off so that a CUDA run's cached features stay as close to a
    CPU run's as the hardware allows. MobileNetV3 is dominated by depthwise
    convolutions, which are memory-bound rather than matmul-bound, so little
    speed is given up for that fidelity.
    """
    if device.type != "cuda":
        return f"cpu, {torch.get_num_threads()} threads"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True          # fixed input shapes per resolution
    p = torch.cuda.get_device_properties(device)
    return f"cuda, {p.name}, {p.total_memory / 1e9:.1f} GB VRAM, TF32 off"
