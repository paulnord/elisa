"""Bounded pageable FP64 host/device calibration; no hardware benefit claim.

Run unprofiled for timing and separately under Nsight Systems for a timeline.
Every H2D uses fresh host storage and every D2H a fresh computed device result.
The enclosing timer includes host dispatch and synchronization; these are not
pure wire bandwidth or GPU DRAM traffic measurements.
"""
# ruff: noqa: INP001, T201, PLR2004 - standalone diagnostic
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=[4096, 65536, 1048576, 8388608])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.verified_bounded_gpu_window or args.repeats < 10 or any(n < 8 or n > 67108864 for n in args.sizes):
        msg = "Use a registered bounded GPU window,10+repeats and8..67108864bytes"
        raise ValueError(msg)
    import jax  # noqa: PLC0415 - optional explicit GPU dependency
    import jax.numpy as jnp  # noqa: PLC0415
    if not jax.config.x64_enabled:
        msg = "FP64 must remain enabled"
        raise RuntimeError(msg)
    device = jax.devices("gpu")[0]
    compute = jax.jit(lambda a, s: a+jnp.asarray(s), device=device)
    rows = []
    for size in args.sizes:
        count = size//8
        samples = []
        for i in range(args.repeats+1):
            host = np.full(count, 1.+i*.001, dtype=np.float64)
            start = perf_counter()
            placed = jax.device_put(host, device, may_alias=False).block_until_ready()
            h2d = perf_counter()
            result = compute(placed, float(i)).block_until_ready()
            computed = perf_counter()
            output = np.asarray(result)
            done = perf_counter()
            np.testing.assert_array_equal(output, host+i)
            samples.append({"h2d_dispatch_sync_seconds": h2d-start,
                            "kernel_dispatch_sync_seconds": computed-h2d,
                            "d2h_first_host_read_seconds": done-computed,
                            "enclosing_seconds": done-start})
        rows.append({"bytes_per_direction": count*8, "first_call": samples[0], "warm_samples": samples[1:]})
    args.output.write_text(json.dumps({"device": str(device), "rows": rows,
        "allocator_stats": device.memory_stats(), "contract": "Pageable numpy source; JAX may internally stage. "
        "Fresh FP64 buffers; blocking per stage; no intentional overlap or host-read cache. "
        "Not pinned-memory calibration, pure interconnect bandwidth or measured DRAM traffic."}, indent=2)+"\n")
    print(json.dumps({"sizes": args.sizes, "repeats": args.repeats}))


if __name__ == "__main__":
    main()
