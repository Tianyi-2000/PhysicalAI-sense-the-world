# PhysicalAI-sense-the-world

See → think → act, all on the edge. Qualcomm Physical AI Hackathon, IoT track.

## What it does

Input → on-device inference on the Hexagon NPU → bounded MCU action. Project idea TBD (DINOv2 vision or Qwen3 text).

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

## Repo layout

```
PhysicalAI-sense-the-world/
├── README.md
├── .gitignore
├── mcu/                 # STM32 side: Arduino sketch
│   ├── README.md
│   └── sketch.ino
├── linux/               # Linux side: Python app + inference
│   ├── README.md
│   ├── classify.py      # photo → DINOv2 on the NPU
│   ├── server.py        # web server: UI + /classify
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
