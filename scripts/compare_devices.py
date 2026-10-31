"""Pilot native CPU versus MPS with identical embedding weights and float32 outputs."""

import time

import numpy as np

from vector_masterclass.evidence import ResourceMeter, json_write, new_run, provenance
from vector_masterclass.hardware import hardware_info
from vector_masterclass.models import embed, release_models


def main():
    texts = [
        f"Atlas unit {i}: reset the router after exporting the configuration." for i in range(96)
    ]
    rows, outputs = [], {}
    with ResourceMeter(docker=False) as meter:
        for device in ["cpu", "mps"]:
            start = time.perf_counter()
            embed(texts[:1], device=device)
            cold = time.perf_counter() - start
            for batch_size in [1, 16, 64]:
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    vectors = embed(texts, device=device, batch_size=batch_size)
                    samples.append(time.perf_counter() - start)
                rows.append(
                    {
                        "device": device,
                        "batch_size": batch_size,
                        "cold_load_and_first_query_seconds": cold,
                        "median_warm_seconds": float(np.median(samples)),
                        "samples_seconds": samples,
                    }
                )
                outputs[(device, batch_size)] = vectors
            release_models()
    difference = float(np.max(np.abs(outputs[("cpu", 16)] - outputs[("mps", 16)])))
    result = {
        "provenance": provenance(),
        "hardware": hardware_info(),
        "rows": rows,
        "max_absolute_cpu_mps_difference": difference,
        "resources": meter.report(),
        "interpretation": "Fixed tiny input set, three warm repetitions. Not a representative throughput benchmark; includes host transfers and explicit MPS synchronization.",
    }
    out = new_run("device_comparison")
    json_write(out / "manifest.json", result)
    print(out)
    print(rows)
    print("max absolute difference:", difference)
    assert difference < 1e-4, "Device drift needs investigation before choosing MPS"


if __name__ == "__main__":
    main()
