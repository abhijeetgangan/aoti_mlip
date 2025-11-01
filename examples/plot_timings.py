import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator

MODEL = "mattersim-v1.0.0-5M.pth"


def load_timings(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_timings(
    *, out_path: str, aoti_only: dict[str, Any] | None, torch_only: dict[str, Any] | None
) -> None:
    # Determine inputs from separate files only
    n_calls = 1
    series: dict[str, tuple[list[int], list[float]]] = {}

    if aoti_only is not None:
        n_calls = int(aoti_only.get("n_calls", n_calls))
        entries = aoti_only["entries"]
        xs = [int(e["num_atoms"]) for e in entries]
        ts = [float(e["time"]) for e in entries]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        series["aoti"] = ([xs[i] for i in order], [ts[i] for i in order])

    if torch_only is not None:
        n_calls = int(torch_only.get("n_calls", n_calls))
        entries = torch_only["entries"]
        xs = [int(e["num_atoms"]) for e in entries]
        ts = [float(e["time"]) for e in entries]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        series["torch script"] = ([xs[i] for i in order], [ts[i] for i in order])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"CUDA timings for {MODEL}", fontsize=18)

    for label, (xs_ser, ts_ser) in series.items():
        tp = [n_calls / t for t in ts_ser]
        ax.loglog(xs_ser, tp, label=label, marker="o", markersize=8)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.xaxis.set_major_formatter(LogFormatter(base=2))

    ax.set_xlabel("Number of Atoms", fontsize=14)
    ax.set_ylabel("Throughput [timesteps/sec]", fontsize=14)
    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle="--", alpha=0.5)
    ax.set_ylim(1e4, 1e6)
    ax.legend(fontsize=12)

    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot timing results")
    default_dir = os.path.dirname(__file__)
    default_aoti = os.path.join(
        default_dir, f"timings_cuda_aoti_{MODEL.replace('.pth', '').replace('/', '_')}.json"
    )
    default_torch = os.path.join(
        default_dir, f"timings_cuda_torchscript_{MODEL.replace('.pth', '').replace('/', '_')}.json"
    )
    default_png = os.path.join(
        os.getcwd(), f"mattersim_timings_cuda_{MODEL.replace('.pth', '').replace('/', '_')}.png"
    )
    parser.add_argument("--aoti", default=None, help="Path to aoti-only timings JSON")
    parser.add_argument(
        "--torch", dest="torch_path", default=None, help="Path to torch-only timings JSON"
    )
    parser.add_argument("--out", default=default_png, help="Path to save plot PNG")

    args = parser.parse_args()

    aoti_only = (
        load_timings(args.aoti or default_aoti)
        if (args.aoti or os.path.exists(default_aoti))
        else None
    )
    torch_only = (
        load_timings(args.torch_path or default_torch)
        if (args.torch_path or os.path.exists(default_torch))
        else None
    )

    if not any([aoti_only, torch_only]):
        raise SystemExit("No input provided. Pass --aoti and/or --torch.")

    plot_timings(out_path=args.out, aoti_only=aoti_only, torch_only=torch_only)
