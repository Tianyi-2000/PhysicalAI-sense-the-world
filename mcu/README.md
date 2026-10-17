# mcu/

The Arduino sketch that runs on the STM32 (sensors, GPIO). Open `sketch.ino` in Arduino App Lab.

- Register any handler that touches GPIO with `Bridge.provide_safe`, not `Bridge.provide`.
- Never open the Bridge UART (`Serial1` on the MCU, `/dev/ttyHS1` on Linux) from app code. The Bridge owns it.

Docs: https://docs.arduino.cc/tutorials/ventuno-q/bridge/
