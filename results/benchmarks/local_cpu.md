# Local warm inference benchmark

resize + CV + optional YOLO + fusion + annotation; excludes decode, disk, HTTP and model loading

Images: 12; repetitions: 5; warmup: 3; device: cpu; threads: 4.

| Mode | Mean ms | Median ms | p95 ms | Images/s |
|---|---:|---:|---:|---:|
| classical_only | 89.85 | 94.18 | 119.81 | 11.13 |
| hybrid | 218.42 | 233.77 | 267.36 | 4.58 |

Throughput is serial warm inference throughput, not concurrent API capacity. See JSON for environment, image hashes, original resolutions, configuration and individual timings.