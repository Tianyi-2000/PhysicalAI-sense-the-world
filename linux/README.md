# linux/

The Python app that runs on the Linux side (QCS8275).

- `main.py`: the app loop, run from Arduino App Lab. Right now it toggles the LED every second; it will become sensor → inference → MCU call.
- `msgpack_test.py`: toggles the LED directly, without App Lab. Keep the sketch running, stop `main.py` first, then:
  ```bash
  python3 msgpack_test.py 1   # LED on
  python3 msgpack_test.py 0   # LED off
  ```
- `requirements.txt`: `pip install -r requirements.txt`. `arduino.app_utils` comes with App Lab, so it is not listed.

Model files live in `$HOME/model_output` on the board, not in git.
