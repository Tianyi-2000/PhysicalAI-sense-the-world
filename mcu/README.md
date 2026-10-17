# mcu/

An Arduino App Lab app. The sketch runs on the STM32 and drives the LED matrix and an RGB LED.

```
mcu/
├── app.yaml
├── python/main.py      # empty on purpose: linux/server.py talks to the sketch through the Router
└── sketch/
    ├── sketch.ino
    └── sketch.yaml     # declares ArduinoGraphics (text on the LED matrix)
```

Handlers the sketch provides over Arduino Bridge:

| Handler | What the board does |
| --- | --- |
| `show_thinking()` | Scanning bar on the LED matrix, RGB LED blue |
| `show_result(label, confidence)` | Scrolls the label; RGB LED green (≥ 30%) or yellow |
| `show_error()` | Clears the matrix, RGB LED red |
| `set_led_state(state)` | Built-in LED (the original LED demo) |

The sketch owns the limits: it cleans the label (A–Z, 0–9, max 16 chars), clamps the confidence, and picks the colors.

## Run it

On the board:

```bash
mkdir -p ~/ArduinoApps && cp -r ~/PhysicalAI-sense-the-world/mcu ~/ArduinoApps/sense-the-world
arduino-app-cli app start ~/ArduinoApps/sense-the-world
```

Or open the folder in Arduino App Lab and click Run.

## Rules

- Register any handler that touches GPIO with `Bridge.provide_safe`, not `Bridge.provide`.
- Keep handlers short. Do the drawing in `loop()` so the Bridge never blocks.
- Never open the Bridge UART (`Serial1` on the MCU, `/dev/ttyHS1` on Linux) from app code. The Bridge owns it.

Docs: https://docs.arduino.cc/tutorials/ventuno-q/bridge/
