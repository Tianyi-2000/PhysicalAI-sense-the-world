# mcu/

An Arduino App Lab app. The sketch runs on the STM32, drives the LED matrix and an RGB LED, and reads the Modulino Distance sensor (Qwiic). It replaces `depth-sensor/` when you want the live demo on the LEDs: same `get_distance()`, plus the displays.

```
mcu/
├── app.yaml
├── python/main.py      # empty on purpose: linux/server.py talks to the sketch through the Router
├── sketch/
│   ├── sketch.ino
│   └── sketch.yaml     # declares ArduinoGraphics (LED matrix text) and Arduino_Modulino (sensor)
└── depth-sensor/       # sensor only: get_distance(), no displays
```

Handlers the sketch provides over Arduino Bridge:

| Handler | What the board does |
| --- | --- |
| `show_thinking()` | Scanning bar on the LED matrix, RGB LED blue |
| `show_result(label, confidence)` | Scrolls the label; RGB LED green (≥ 30%) or yellow |
| `show_error()` | Scrolls "ERROR", RGB LED red |
| `show_frame(bits)` | Draws a 13×8 picture: 104 `0`/`1` characters, row by row. Stays until the next call |
| `show_distance(mm)` | Shows the distance in cm (up to 999) on the LED matrix |
| `set_led(r, g, b)` | RGB LED, 0–255 per channel. On/off only: a channel lights at ≥ 128 |
| `get_distance()` | Latest sensor reading in mm; -1 before the first reading or with no sensor |
| `set_led_state(state)` | Built-in LED (the original LED demo) |

With nothing to show (at boot, 20 s after a result, or 2 s after the last distance), the matrix shows a blinking eye that looks around.

The sketch owns the limits: it cleans the label (A–Z, 0–9, max 16 chars), clamps the confidence and the distance, and picks the colors.

`linux/live_server.py --led` calls `set_led` (red near, amber mid, green far) and `show_distance` with the camera's estimate for every frame.

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
