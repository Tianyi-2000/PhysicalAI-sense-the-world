# linux/

The Python app that runs on the Linux side (QCS8275).

- `classify.py`: photo → DINOv2 on the NPU → top-5 labels + latency.
  ```bash
  python3 classify.py photo.jpg         # on the board (NPU)
  python3 classify.py photo.jpg --cpu   # on a laptop, same model on CPU
  ```
- `server.py`: serves the web UI (`../web/dist`) and classifies uploaded photos. Run `python3 server.py`, then open `http://<board IP>:8000` on a phone and `http://<board IP>:8000/screen` on the big screen. Add `--cpu` on a laptop.
- `mcu.py`: calls the sketch's handlers (`show_result`, ...) through the Router socket. `server.py` uses it after each photo; on a laptop it just returns `None`.
- `imagenet_labels.txt`: the model's 1000 class names.
- `main.py`: the App Lab loop. Right now it toggles the LED every second.
- `msgpack_test.py`: toggles the LED directly, without App Lab. Keep the sketch running, stop `main.py` first, then:
  ```bash
  python3 msgpack_test.py 1   # LED on
  python3 msgpack_test.py 0   # LED off
  ```

## Setup

- Board: `pip install -r requirements.txt`. `arduino.app_utils` comes with App Lab, so it is not listed.
- Laptop (`--cpu` only): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt torch transformers`.

Model files live in `$HOME/model_output` on the board, not in git. Override paths with `MODEL_OUTPUT`, `QNN_RUNNER`, or `DINO_PTE` if the bundle puts them elsewhere.
