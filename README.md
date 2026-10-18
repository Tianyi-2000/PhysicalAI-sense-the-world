# Sensor Teaches Camera

**A distance sensor labels camera frames automatically, the board learns to estimate distance from images alone, and then the sensor can be taken away.**

Built at the ExecuTorch Hackathon (Meta, Qualcomm, GitHub, PyTorch Foundation), IoT track, October 17 to 18, 2026, on an Arduino VENTUNO Q.

---

## Contents

1. [The demo](#the-demo)
2. [The problem and the idea](#the-problem-and-the-idea)
3. [System overview](#system-overview)
4. [Hardware](#hardware)
5. [Software stack](#software-stack)
6. [Data collection](#data-collection)
7. [Model](#model)
8. [Training and evaluation methodology](#training-and-evaluation-methodology)
9. [ExecuTorch export and optimization](#executorch-export-and-optimization)
10. [Results](#results)
11. [Live demo system](#live-demo-system)
12. [Key design decisions](#key-design-decisions)
13. [Engineering log: problems we hit and how we solved them](#engineering-log-problems-we-hit-and-how-we-solved-them)
14. [Limitations](#limitations)
15. [Repository layout](#repository-layout)
16. [How to run](#how-to-run)
17. [Protocols and file formats](#protocols-and-file-formats)
18. [Troubleshooting](#troubleshooting)
19. [Questions we expect](#questions-we-expect)
20. [What we would do next](#what-we-would-do-next)

---

## The demo

1. Hold an object in front of the camera. The screen shows two numbers: the **camera's estimate** of its distance, and the **sensor's real reading**.
2. Move the object between ~10 and ~60 cm. The two numbers track each other, with the live error shown underneath.
3. Cover the sensor with a finger. The sensor reads "covered", and **the camera keeps estimating distance on its own**.

Everything after the camera frame (the vision model, the regression, the sensor reading) runs on the VENTUNO Q. The vision model runs through **ExecuTorch on the board's CPU**, 3.6x faster than plain PyTorch on the same hardware.

---

## The problem and the idea

**Depth is useful everywhere** (robots avoiding obstacles, appliances that react when you approach, inventory systems that check fill levels), but measuring it usually needs dedicated hardware: time-of-flight sensors, lidar, stereo cameras. Cameras, meanwhile, are already everywhere and see far more than a single distance.

Learning depth from a camera normally needs **lots of labeled data**, and labeling distances by hand is slow and imprecise.

**Our idea:** let a cheap, precise sensor do the labeling. Mount a time-of-flight distance sensor next to the camera and move objects around. Every frame is automatically paired with an exact distance label, about 500 labeled frames per minute with zero manual labeling. Train a model on those pairs, and the camera learns to estimate distance by itself. After that, the sensor is no longer needed.

This is **cross-modal self-supervision**: one sensor teaches another. It also happens entirely **on the edge**: collection, training, export, and inference all run on the board, and no data leaves the device.

---

## System overview

```
 LAPTOP                      BOARD: LINUX SIDE (Qualcomm QCS8275)        BOARD: MCU SIDE (STM32H5F5)
 ┌──────────────┐   frames   ┌────────────────────────────────┐ distance ┌────────────────────────┐
 │ Laptop webcam├───────────►│ Frame server                   │◄─────────┤ Distance sensor        │
 └──────────────┘  TCP 5000  │  pairs each frame with the     │  Arduino │ VL53L4CD time-of-flight│
                   over USB  │  sensor reading at that moment │  Bridge  │ on the Qwiic connector │
                             └───────────────┬────────────────┘   RPC    └────────────────────────┘
                                             ▼
                             ┌────────────────────────────────┐
                             │ DINOv2-small feature extractor │
                             │ ExecuTorch, XNNPACK, CPU       │
                             │ image → 768-number vector      │
                             └───────────────┬────────────────┘
 ┌──────────────┐ prediction ┌───────────────▼────────────────┐
 │ Live view    │◄───────────┤ Regression head (trained here) │
 │ cam vs sensor│            │ vector → distance in cm        │
 └──────────────┘            └────────────────────────────────┘
```

The board has two processors, and each does what it is good at:

| Processor | Role in this project |
|---|---|
| **STM32H5F5 microcontroller** (Arduino side) | Talks to the sensor over I2C with deterministic timing; exposes the latest reading as an RPC method |
| **QCS8275 application processor** (Ubuntu Linux) | Receives frames, runs the vision model and regression, trains, serves the live demo |

The two sides communicate through **Arduino Bridge**, a MessagePack-RPC channel routed by the `arduino-router` service on Linux.

---

## Hardware

| Component | Details |
|---|---|
| Board | Arduino VENTUNO Q |
| Application processor | Qualcomm Dragonwing IQ8 (QCS8275), octa-core Arm (4 performance + 4 efficiency cores), Ubuntu Linux, Hexagon NPU |
| Microcontroller | STM32H5F5, Arm Cortex-M33 at 250 MHz, Arduino Zephyr core |
| Distance sensor | Arduino Modulino Distance (ST VL53L4CD time-of-flight), ~1 mm to ~1.3 m range, narrow ~18° cone, connected to the MCU's Qwiic port |
| Camera | Laptop webcam, frames resized to 320 px wide before sending |
| Link laptop ↔ board | USB-C with `adb forward` (or Wi-Fi via phone hotspot) |
| Power | USB-C PD wall charger (laptop USB power is not enough; the board flashes red) |

The sensor is mounted directly beside the camera, pointing the same way, so the distance it measures belongs to whatever is at the centre of the camera frame (marked with a green crosshair in the UI).

---

## Software stack

| Layer | Version / detail |
|---|---|
| Board OS | Ubuntu Linux on QCS8275 |
| Python (board) | 3.12, virtual environment at `~/venv` |
| ExecuTorch | 1.5.1 (pip wheel for Linux aarch64), XNNPACK backend |
| PyTorch | CPU build for aarch64 (pip) |
| Vision model | DINOv2-small (`dinov2_vits14`) via `torch.hub` |
| Arduino core | `arduino:zephyr` 1.0.0 |
| Arduino libraries | Arduino_Modulino 0.10.0, Arduino_RouterBridge 0.4.3, STM32duino VL53L4CD 1.0.5 |
| Laptop | Python with OpenCV for webcam capture and display |

---

## Data collection

### Protocol

- The sensor is fixed next to the laptop camera, aimed at the frame centre.
- One object per **session**. Each session is a separate recording, started with its own name.
- The object is held at the crosshair and swept slowly between close range and arm's length, about 5 seconds per pass, with short pauses at several distances, plus small tilts and rotations between passes.
- Background and lighting are kept constant across sessions.
- The laptop sends ~8 frames per second; the board reads the sensor the moment each frame arrives (3.3 ms median per read) and saves the pair.

### Sessions recorded

| Session | Object | Role | Frames recorded | Valid frames | Distance range |
|---|---|---|---|---|---|
| `train_notebook` | Notebook | Training | 278 | 250 | 11 to 630 mm |
| `train_phone` | Phone | Training | 306 | 272 | 22 to 619 mm |
| `train_hand` | Open hand | Training | 166 | 135 | 35 to 474 mm |
| `val_notebook2` | Same notebook, new recording | Test: seen object | 241 | 220 | 16 to 523 mm |
| `test_mug` | Coffee mug, never used in training | Test: unseen object | 159 | 159 | 53 to 506 mm |

**Valid** means the sensor read between 50 and 1200 mm. Below 5 cm the object covers the camera and the image carries no usable information; above ~1.2 m the sensor becomes unreliable.

---

## Model

### Feature extractor: DINOv2-small

**DINOv2** is a Vision Transformer released by Meta, trained with self-supervised learning on ~142 million images without human labels. Given an image, it outputs embeddings that capture shape, texture, and structure. Images that look alike produce nearby embeddings.

We use it **frozen** (no fine-tuning):

| Setting | Value |
|---|---|
| Variant | `dinov2_vits14` (ViT-S/14, ~21M parameters) |
| Input size | 308 × 168 px (multiples of the 14 px patch size, keeping the frame's aspect ratio) |
| Patch grid | 22 × 12 patches |
| Normalization | ImageNet mean and std |
| Output feature | CLS token (384) + mean of the centre 7 × 4 patch tokens (384) = **768 numbers** |

The centre patches cover roughly the same part of the image as the sensor's cone, so the feature focuses on the object being measured.

### Regression head

- **Target:** log of the distance in mm. Apparent size scales with 1/distance, so log makes the relationship closer to linear.
- **Model:** ridge regression on standardized features (mean and std from the training set only).
- **Regularization:** chosen by leave-one-session-out validation over the training sessions.
- **Calibration:** a linear correction in log space, `log(true) = a × log(pred) + b`, fit on **out-of-fold** training predictions. Ridge regression shrinks predictions toward the average (overestimating close objects, underestimating far ones); calibration undoes that compression without ever using test data.

The head trains in seconds on the board's CPU.

---

## Training and evaluation methodology

### Splits

- **Training:** notebook, phone, hand (657 valid frames, 3 sessions).
- **Seen-object test:** a second, separate recording of the notebook.
- **Unseen-object test:** a coffee mug, never shown during training.

**We split by recording session, never by random frame.** Consecutive frames are nearly identical, so a random frame split would put near-duplicates in both training and test, and the accuracy would look far better than it really is.

### Choosing the regularization strength

Leave-one-session-out on the three training sessions (train on two objects, validate on the third):

| alpha | 0.1 | 1 | **10** | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| Validation error (mm) | 97.5 | 87.6 | **84.2** | 97.3 | 104.6 | 111.4 | 126.1 |

Chosen alpha = 10. Note that this validation **is itself an unseen-object test** (each fold validates on an object left out of training), and its ~8.4 cm error is consistent with the mug result below. The cross-object difficulty was visible before we ever touched the test set.

### Metrics

- Mean and median absolute error in cm
- Share of frames within 5 cm
- Mean signed error (bias)
- **Baseline:** always predicting the median training distance (36 cm). Any useful model must clearly beat it.

### Latency measurement

- Single image (batch size 1), 5 warm-up runs, then 30 to 50 timed runs; we report p50 and p95.
- Measured on the board, with all other workloads stopped.
- The board mixes performance and efficiency cores, which explains part of the gap between p50 and p95.

---

## ExecuTorch export and optimization

### Export path

1. **Wrap** DINOv2 in a module that outputs exactly our 768-number feature.
2. **Precompute the position encoding** for the fixed 308 × 168 input. DINOv2 normally interpolates its position embeddings at run time with a bicubic resize; baking them in removes that operation from the exported graph.
3. `torch.export.export(...)` → `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])` → `.to_executorch()` → `dinov2_fp32.pte`.
4. **Verify:** the wrapper reproduces the training features exactly (max difference 0.0), and the exported program matches PyTorch to 3.4e-5.

### int8 post-training quantization

- `XNNPACKQuantizer` with the symmetric quantization config, through the PT2E flow (`prepare_pt2e` → calibrate → `convert_pt2e`).
- Calibrated on 32 frames from a training session.
- Lowered the same way to `dinov2_int8.pte`.

### Results of export

| Runtime | Size | p50 | p95 | Speed vs PyTorch |
|---|---|---|---|---|
| PyTorch eager, fp32 | 86.7 MB | 870 ms | 975 ms | 1.0x |
| **ExecuTorch XNNPACK, fp32** | 86.7 MB | **242 ms** | 532 ms | **3.6x** |
| ExecuTorch XNNPACK, int8 | **22.0 MB** | 397 ms | 425 ms | 2.2x |

---

## Results

### Accuracy (cm, on data never used for training)

| Test set | Frames | Mean error | Median error | Within 5 cm | Baseline error |
|---|---|---|---|---|---|
| Same notebook, new recording | 220 | **4.2** | 3.9 | 67% | 12.3 |
| Coffee mug, never seen | 159 | **10.2** | 9.9 | 10% | 13.1 |

*Numbers above are before calibration; calibrated results will be added after the final training run.*

**What this shows**
- On a **seen object in a new recording**, the camera estimates distance to within ~4 cm on average, **3x better than the baseline**.
- On an **unseen object**, it only slightly beats the baseline. With three training objects, the model partly learns *how large each known object looks* rather than distance in general. The direct fix is more, and more varied, training objects.

### Full comparison across runtimes

| Runtime | Size | p50 latency | Seen-object error | Unseen-object error |
|---|---|---|---|---|
| PyTorch eager, fp32 | 86.7 MB | 870 ms | 4.2 cm | 10.2 cm |
| **ExecuTorch XNNPACK, fp32 (deployed)** | 86.7 MB | **242 ms** | **4.2 cm** | 10.2 cm |
| ExecuTorch XNNPACK, int8 | 22.0 MB | 397 ms | 10.4 cm | 6.4 cm |

### The int8 decision

**We deploy ExecuTorch fp32**, and the reasons matter as much as the numbers:

- **fp32 is 3.6x faster than PyTorch** with identical accuracy.
- **int8 is 4x smaller, but 1.6x slower than fp32** and loses 6 cm on the seen object. In a vision transformer, only the large matrix multiplies are quantized; layer norms, softmax, and GELU stay in float. The program keeps converting between int8 and float, and at this model size that overhead outweighs the faster integer math.
- **int8's better mug number is not better generalization.** The regression head was trained on fp32 features. Quantization shifts every feature, which happened to move the mug's predictions closer while making the notebook's much worse. A fair int8 comparison would retrain the head on int8 features.
- int8 would be the right choice if **storage** were the constraint (22 MB vs 87 MB). Here, speed and accuracy matter more.

---

## Live demo system

- **Board (`live_server.py`):** for every incoming frame, reads the sensor, runs the ExecuTorch feature extractor and the regression head, and replies with the sensor reading, the prediction, and the inference time. Every frame is also logged to `~/depth_data/live_log.csv`, so a live session can be scored afterward.
- **Laptop (`live_client.py`):** shows the webcam at full frame rate. A background thread always sends the **newest** frame to the board and overlays the latest prediction, so old frames never queue up. Predictions update ~3 to 4 times per second; the displayed value is lightly smoothed.
- **"Sensor covered" mode:** when the sensor reads outside 50 to 1200 mm, the UI shows "Sensor: covered" and the camera's estimate keeps running.
- **Board LEDs (`--led`, needs the `mcu/` app):** for every frame, the LED matrix shows the camera's estimate in cm, and the RGB LED shows near (red), mid (yellow), or far (green). The board shows its own result, with no laptop needed to read it.

---

## Key design decisions

| Decision | Why |
|---|---|
| Split evaluation by session, not by frame | Neighbouring frames are near-duplicates; a frame split leaks |
| Hold out one object entirely | Tests whether the model learned distance or memorized object sizes |
| Frozen DINOv2 + linear head instead of fine-tuning | ~650 labeled frames is far too few to fine-tune a ViT; a linear head trains in seconds on the board |
| Regress log distance | Apparent size ∝ 1/distance; log is closer to linear |
| CLS + centre patches as the feature | Matches the region the sensor actually measures |
| Choose alpha by leave-one-session-out | Validation reflects the real question: performance on a new recording |
| Calibrate on out-of-fold predictions | Fixes shrinkage toward the mean without touching test data |
| Precompute the position encoding before export | Fixed input size; keeps an interpolation op out of the exported graph |
| Deploy fp32 over int8 | Faster and more accurate here; int8 only wins on size |
| Background prediction thread in the live view | Video stays smooth regardless of inference time |
| Train and export on the board itself | No laptop-to-board version mismatch; the whole loop stays on the edge |

---

## Engineering log: problems we hit and how we solved them

| Problem | Cause | Fix |
|---|---|---|
| Board flashed red when plugged into the laptop | Laptop USB can't supply enough power | Wall USB-C PD charger; connect to the laptop over network or `adb` |
| Teammates couldn't reach the board over Wi-Fi | Venue Wi-Fi isolates devices from each other | Phone hotspot, SSH keys installed through `adb` |
| Sensor read `-1` after a cable disconnect | Sensor stuck in a bad state | Power-cycle the whole board, then restart the App Lab app |
| Several recordings landed in one session | The board server wasn't restarted with a new session name | Split sessions by pauses of more than 3 s in the timestamps, then renamed each part |
| ExecuTorch program failed with `xnn_status_invalid_parameter` | Input tensor was a non-contiguous (transposed) view | `np.ascontiguousarray` in preprocessing |
| Live video was laggy | Display waited for each ~250 ms round trip | Background prediction thread; video at full frame rate |
| Camera overestimated close objects | Ridge shrinkage toward the mean | Out-of-fold calibration |

---

## Limitations

- **Range:** training data spans ~1 to 63 cm, so predictions are only meaningful in that range.
- **Single-point labels:** the sensor gives one distance per frame, so the model estimates the distance of the object at the centre of the frame, not a full depth map.
- **Unseen objects:** generalization to new objects is weak with three training objects.
- **Small dataset:** ~650 training frames from one room, one background, one lighting setup.
- **Camera on the laptop:** frames come from the laptop webcam and are processed on the board. A USB camera on the board would make the system fully self-contained.
- **NPU not used:** the Hexagon NPU needs a QNN-lowered model, and QNN export currently requires an x86 host with the board's matching SDK. All inference here runs on the CPU through XNNPACK.
- **Label timing:** the frame and sensor reading are taken tens of milliseconds apart; slow movement during collection keeps the resulting error to about a centimeter.

---

## Repository layout

```
linux/collector_server.py   board: receives frames, reads the sensor, saves (frame, distance) pairs
linux/live_server.py        board: live demo server (sensor reading + model prediction per frame)
laptop/capture_client.py    laptop: streams webcam frames for data collection
laptop/live_client.py       laptop: live demo view, full-rate video + prediction overlay
ml/depth_ml.py              board: DINOv2 embedding, training, calibration, evaluation, live predictor
ml/export_et.py             board: ExecuTorch export (fp32 / int8), benchmark, accuracy check
mcu/depth-sensor/sketch.ino MCU: reads the distance sensor, exposes get_distance() over Bridge
mcu/sketch/sketch.ino       MCU: same get_distance(), plus the LED matrix and RGB LED (live_server.py --led)
```

Data (`~/depth_data/`) and trained models (`ml/model/`, `*.pte`) stay on the board and are not in git.

---

## How to run

### Quick test (board already set up)

The board already has the venv, the trained model (`~/ml/model/`), and the `mcu/` app, so you only need steps 1 to 6 below to run the live demo. Power the board from a wall charger and plug it into your laptop over USB. Run everything from the repo root.

1. `git pull`, then `adb devices`. The board should be listed.
2. `adb shell arduino-app-cli app list` should show "Sense the World MCU" as `running`. If it doesn't, run `adb shell arduino-app-cli app start ~/ArduinoApps/sense-the-world`. If you changed `mcu/sketch/`, copy it first: `adb push mcu/sketch/sketch.ino mcu/sketch/sketch.yaml /home/arduino/ArduinoApps/sense-the-world/sketch/`, then start the app.
3. Optional 5-second MCU check, without the model:
   ```bash
   adb shell 'cd ~ && ~/venv/bin/python -c "from collector_server import RouterClient; r = RouterClient(); print(r.call(\"get_distance\")); r.call(\"show_distance\", 450); r.call(\"set_led\", 255, 160, 0)"'
   ```
   It prints the sensor reading in mm (-1 if nothing has been in range yet). The LED matrix shows `45` and the RGB LED turns yellow, then after 2 s the matrix goes back to the blinking eye.
4. Copy the latest server to the board and start it. Keep this terminal open, and wait about a minute for `Live server ready on port 5000`:
   ```bash
   adb push linux/live_server.py /home/arduino/live_server.py
   adb shell 'cd ~ && ~/venv/bin/python -u ~/live_server.py --led'
   ```
5. In a second terminal on the laptop:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install opencv-python
   adb forward tcp:5000 tcp:5000
   .venv/bin/python laptop/live_client.py
   ```
   Allow camera access when macOS asks. If you run Claude Code in the same terminal app, start the client from a different terminal app: granting camera access means restarting the app, and restarting it ends the Claude Code session.
6. Hold an object at the green crosshair and move it between ~10 and ~60 cm:

   | Where | What you should see |
   |---|---|
   | Window, big text | `Camera: xx cm`: the model's estimate from the camera frame, computed on the board |
   | Window, small text | `Sensor: xx cm`: the sensor's real reading, and `Error`: the gap between the two |
   | LED matrix | The camera's estimate in cm |
   | RGB LED | Red under 30 cm, yellow from 30 to 60 cm, green above 60 cm |

   Cover the sensor with a finger: the window shows `Sensor: covered`, while the camera's estimate and the LEDs keep updating. Press q in the window to quit. Ctrl+C in the board terminal only stops `adb`, and the server keeps running on the board. To stop it: `adb shell "pkill -f '[l]ive_server.py'"`.

Tested on 2026-10-18 with this setup: the model takes ~250 to 500 ms per frame on the board CPU, and a sensor read through the router takes ~7 ms.

<details>
<summary>Or paste this into Claude Code</summary>

```text
I'm running the live demo of our hackathon project on an Arduino VENTUNO Q connected to this laptop over USB (adb). Read README.md ("How to run", "Quick test") and mcu/README.md first. Don't change the model, the data, or any code. Just run it. Never use sudo.

Do the "Quick test" steps 1 to 4. Run the board server (step 4) in the background and wait for "Live server ready on port 5000". Then give me the step 5 commands to run in a separate terminal app (the camera window needs camera access there), and tell me what to check in step 6. If something fails, check "Troubleshooting" in README.md before changing anything, and tell me what you found.
```

</details>

### 1. Board setup (once)

```bash
python3 -m venv --without-pip ~/venv
source ~/venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install numpy msgpack pandas pillow torch executorch
```

Copy `linux/*.py` to `~/` and `ml/*.py` to `~/ml/` on the board (`adb push` or `scp`).

### 2. MCU sketch

Open `mcu/depth-sensor/sketch.ino` in Arduino App Lab (add the Modulino and Arduino_RouterBridge libraries if prompted), plug the Modulino Distance into the Qwiic port, and run the app. Keep it running for every step below.

For the live demo on the board's LEDs, run the `mcu/` app instead (see [mcu/README.md](mcu/README.md)). It provides the same `get_distance()`, and also shows the camera's estimate in cm on the LED matrix, with the RGB LED red (near), yellow (mid), or green (far).

### 3. Collect data (one session per object)

Board:
```bash
source ~/venv/bin/activate
python ~/collector_server.py --session train_notebook
```
Laptop:
```bash
pip install opencv-python
adb forward tcp:5000 tcp:5000
python laptop/capture_client.py
```
SPACE starts and stops recording, q quits. Restart the board server with a **new `--session` name for every object**.

### 4. Train and evaluate

```bash
cd ~/ml
python depth_ml.py embed train_notebook train_phone train_hand val_notebook2 test_mug
python depth_ml.py train --train train_notebook train_phone train_hand --test val_notebook2 test_mug
python depth_ml.py bench
```

For the final demo model, train on every session except the unseen-object test:
```bash
python depth_ml.py train --train train_notebook train_phone train_hand val_notebook2 --test test_mug
```

### 5. Export to ExecuTorch

```bash
python export_et.py --int8
```

Retraining the head later does **not** require re-exporting; the feature extractor doesn't change.

### 6. Live demo

Board:
```bash
python ~/live_server.py
```
Add `--led` when the `mcu/` app is running, to show the estimate on the LED matrix and RGB LED.
Laptop:
```bash
adb forward tcp:5000 tcp:5000
python laptop/live_client.py
```

---

## Protocols and file formats

### Laptop → board (TCP port 5000)

Each frame: a 12-byte header `!dI` (capture timestamp as a double, JPEG length as uint32), followed by the JPEG bytes.

Replies:
- Collector: `!i` (sensor distance in mm)
- Live server: `!iff` (sensor distance in mm, predicted distance in mm, inference time in ms)

### Linux ↔ MCU (Arduino Bridge)

MessagePack-RPC over the router's Unix socket `/var/run/arduino-router.sock`. Request `[0, id, "get_distance", []]`, response `[1, id, null, <mm>]`.

### Data on the board

```
~/depth_data/<session>/frames/000123.jpg
~/depth_data/<session>/labels.csv      idx, file, t_capture_laptop, t_recv_board, distance_mm, rpc_ms
~/depth_data/<session>/emb.npz         cached DINOv2 features (X), labels (y), file names
~/depth_data/live_log.csv              t, sensor_mm, pred_mm, infer_ms
~/ml/model/head.npz                    regression head + calibration
~/ml/model/config.json                 input size, training sessions, results, which .pte to use
~/ml/model/dinov2_fp32.pte             ExecuTorch program (deployed)
~/ml/model/dinov2_int8.pte             ExecuTorch program (int8, for comparison)
~/ml/model/et_results.json             export benchmark and accuracy results
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Board LED flashes red | Not enough power; use a wall USB-C PD charger |
| Sensor reads `-1` | Something must be within range; otherwise power-cycle the board and rerun the App Lab app |
| `python: command not found` on the board | `source ~/venv/bin/activate` first |
| `Could not open camera` on macOS | System Settings → Privacy & Security → Camera → enable your terminal app, then quit and reopen it. If Claude Code runs in that app, run `live_client.py` from another terminal app instead |
| LED matrix shows the blinking eye during the live demo | Start `live_server.py` with `--led`, and check that "Sense the World MCU" is the running app (`adb shell arduino-app-cli app list`) |
| Sensor keeps showing the same number after the object is gone | It keeps its last valid reading while nothing is within range (~1.3 m); put something in front of it |
| `Connection refused` from a laptop script | Board server not running, or `adb forward tcp:5000 tcp:5000` not run |
| Port 5000 already in use | Only one of `collector_server.py` / `live_server.py` can run at a time. An old one may still be running after Ctrl+C: `adb shell "pkill -f '[l]ive_server.py'"` (or `[c]ollector_server.py`) |
| Recordings ended up in the same folder | Restart the board server with a new `--session` name for each object |
| App Lab app stopped unexpectedly | Starting any app stops the running one; coordinate who owns App Lab |
| Pasted commands behave strangely in zsh | Don't paste lines containing `#` comments; zsh doesn't treat them as comments interactively |

---

## Questions we expect

**Isn't the model just measuring how big the object looks?**
Partly, yes, and our results show it: accuracy on an unseen object is much worse than on a seen one. That's why we hold out an unseen object instead of only reporting the easy number. More varied training objects is the fix.

**Why not just use the sensor?**
The sensor measures one point; the camera sees the whole scene and is already present in most devices. This project shows a way to transfer the sensor's precision into the camera with no manual labeling, and the same idea extends to denser sensors and depth maps.

**Why not run it on the NPU?**
The Hexagon NPU needs a model lowered through Qualcomm's QNN backend, and that export currently requires an x86 host with the board's matching SDK. We focused on the CPU path and measured it carefully; NPU lowering is the first next step.

**Why is int8 slower?**
In a transformer, quantization covers only the matrix multiplies. The remaining float operations force conversions between int8 and float throughout the network, and at this size that overhead outweighs the gain.

**How do you know the results aren't inflated?**
We split by recording session (never by frame), choose hyperparameters using training sessions only, fit calibration on out-of-fold training predictions only, and report against a baseline.

---

## What we would do next

1. **More training objects** (10+ varied shapes and sizes) to close the unseen-object gap.
2. **USB camera on the board** so the whole system is self-contained.
3. **Hexagon NPU** via QNN, compared against the CPU numbers above.
4. **Multi-zone time-of-flight sensor** to supervise a coarse depth map instead of a single distance.
5. **Retrain the head on int8 features** for a fair int8 comparison.
6. **Wider range and more environments** (lighting, backgrounds) for robustness.
