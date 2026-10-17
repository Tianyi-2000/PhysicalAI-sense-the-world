# PhysicalAI-sense-the-world

See → think → act, all on the edge. Qualcomm Physical AI Hackathon, IoT track.

## What it does

Snap a photo with a phone. The board recognizes the object and reacts, with no cloud involved:

```text
Phone photo → Linux: preprocess → DINOv2 (ImageNet-1k) → STM32: LED matrix scrolls the label,
                                                          RGB LED green (sure) / yellow (unsure)
```

A web UI shows each step's latency on the phone (`/`) and live on a big screen (`/screen`).
Model: [`facebook/dinov2-small-imagenet1k-1-layer`](https://huggingface.co/facebook/dinov2-small-imagenet1k-1-layer), the same one the upstream ExecuTorch Qualcomm example exports.

## Architecture

```text
Sensors / actuators ↔ STM32H5F5 ↔ Arduino Bridge / Router ↔ Linux application
                                                            │
                                                      ExecuTorch runtime
                                                            │
                                                         QNN HTP
                                                            │ FastRPC
                                                        Hexagon NPU
```

## Run the demo

On the board:

1. Start the MCU app: see [mcu/README.md](mcu/README.md).
2. Start the server: `cd linux && .venv/bin/python server.py`. Add `--cpu` until the NPU bundle is installed. Setup: [linux/README.md](linux/README.md).
3. Open `http://<board IP>:8000` on a phone and `http://<board IP>:8000/screen` on the big screen.

The phone and the board must be on a network that allows device-to-device traffic. The venue Wi-Fi blocks it, so use a phone hotspot. Over USB instead: `adb forward tcp:8100 tcp:8000`, then open `http://localhost:8100`.

## Results

Measured on the board on 2026-10-17: 3 photos ([COCO val2017](https://cocodataset.org) `000000039769`, `000000000285`, `000000000139`) × 5 runs each.

| Stage | Where | Median |
| --- | --- | --- |
| Preprocess | Linux CPU (Pillow + NumPy) | 11 ms |
| DINOv2 | Linux CPU, PyTorch (`--cpu`) | 904 ms |
| DINOv2 | Hexagon NPU (QNN) | TBD: waiting for the track's model bundle |
| MCU action | STM32 over the Router, round trip | 3.4 ms |
| **Total on board (CPU)** | | **915 ms** |

| Photo | Top-1 | Confidence |
| --- | --- | --- |
| Two cats on a blanket | tabby cat | 76.6% |
| Brown bear | brown bear | 99.8% |
| Living room | dining table | 16.3% (shown as "Not sure") |

## Limitations

- Only the 1,000 ImageNet classes. Busy scenes get low confidence, so below 30% the UI and the RGB LED say "not sure".
- The model can be confidently wrong: random noise comes back as "balance beam" at 38.5%.
- The NPU path (`classify.py` without `--cpu`) follows the upstream `qnn_executor_runner` input/output format but hasn't run on the board yet.

## Repo layout

```
PhysicalAI-sense-the-world/
├── README.md
├── .gitignore
├── mcu/                 # STM32 side: App Lab app (LED matrix + RGB LED)
│   ├── README.md
│   ├── app.yaml
│   ├── python/main.py
│   └── sketch/          # sketch.ino + sketch.yaml
├── linux/               # Linux side: Python app + inference
│   ├── README.md
│   ├── classify.py      # photo → DINOv2 on the NPU
│   ├── server.py        # web server: UI + /classify, then tells the MCU
│   ├── mcu.py           # calls the sketch through the Router
│   ├── imagenet_labels.txt
│   ├── main.py
│   ├── msgpack_test.py
│   └── requirements.txt
└── web/                 # Web UI (phone + big screen): React + TypeScript
    ├── README.md
    ├── src/
    └── dist/            # built output, committed so the board needs no Node
```

## Team

| Role | Owns | Name |
| --- | --- | --- |
| MCU | `mcu/` | TBD |
| Inference | `linux/` | TBD |
| Integration + demo | `web/`, `README.md`, wiring it together | TBD |

## How we work

Everyone commits straight to `main`. Three rules:

1. `git pull` before you push.
2. Only touch your own folder.
3. Small, frequent commits.

## Plan

- **Day 1:** board setup, LED works, model runs.
- **Day 2:** close the loop, rehearse the demo.

## Links

- [Reference guide (IoT track)](https://github.com/aaishikasb/qc-executorch-guide/tree/main/IoT)
- [Arduino VENTUNO Q docs](https://docs.arduino.cc/hardware/ventuno-q)
- [Arduino Bridge docs](https://docs.arduino.cc/tutorials/ventuno-q/bridge/)
- [ExecuTorch Qualcomm backend](https://docs.pytorch.org/executorch/1.5/backends-qualcomm.html)
