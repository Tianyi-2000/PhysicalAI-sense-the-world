# Sensor Teaches Camera

**A distance sensor labels camera frames automatically, the board learns to estimate distance from images alone, and then the sensor can be taken away.**

Built in under 24 hours at the ExecuTorch Hackathon (Meta, Qualcomm, GitHub, PyTorch Foundation), IoT track, on an Arduino VENTUNO Q.

---

## The demo

1. Hold an object in front of the camera. The screen shows two numbers: the **camera's estimate** of the distance and the **sensor's real reading**.
2. Move the object between ~10 and ~60 cm. The two numbers track each other.
3. Cover the sensor with a finger. The sensor reads "covered", and **the camera keeps estimating distance on its own**.

Everything after the camera frame (feature extraction, regression, sensor reading) runs on the VENTUNO Q. The model runs through **ExecuTorch on the board's CPU**.

---

## Why this is interesting

- **Labels for free.** Supervised depth estimation normally needs expensive labeled data. Here a $15 time-of-flight sensor produces a perfect label for every frame while you simply move objects around. About 500 labeled frames per minute, zero manual labeling.
- **Cross-modal self-supervision.** One sensor teaches another. Once trained, the cheaper and more general sensor (a camera) replaces the specialized one.
- **On-device, end to end.** Collection, training, export, and inference all happen on the board. No cloud, no data leaves the device.

---

## How it works

```
 LAPTOP                     BOARD: LINUX SIDE (QCS8275)              BOARD: MCU SIDE (STM32H5)
 ┌──────────────┐  frames   ┌──────────────────────────────┐ distance ┌──────────────────────┐
 │ Laptop webcam├──────────►│ Frame server                 │◄─────────┤ Distance sensor      │
 └──────────────┘ TCP 5000  │  saves (frame, distance)     │  Bridge  │ VL53L4CD, Qwiic      │
                            └──────────────┬───────────────┘   RPC    └──────────────────────┘
                                           ▼
                            ┌──────────────────────────────┐
                            │ DINOv2-small (ExecuTorch,    │
                            │ XNNPACK, CPU): image → vector│
                            └──────────────┬───────────────┘
 ┌──────────────┐ prediction ┌─────────────▼───────────────┐
 │ Live view    │◄───────────┤ Regression head: vector → cm│
 └──────────────┘            └──────────────────────────────┘
```

**1. Collection.** The laptop streams webcam frames (320 px wide, ~8 per second) to the board. For each frame the board immediately reads the distance sensor over Arduino Bridge (3.3 ms per read) and saves the pair. The sensor is mounted next to the camera, pointing the same way, so its reading is the distance to whatever is at the centre of the frame (green crosshair in the UI).

**2. Features.** Each frame goes through **DINOv2-small** (Meta's self-supervised vision transformer, 21M parameters). The feature vector is the CLS token concatenated with the mean of the centre patch tokens, the region roughly matching the sensor's ~18° cone.

**3. Regression head.** Ridge regression on **log distance** (apparent size scales with 1/distance, so log makes the relation closer to linear). The regularization strength is chosen by leave-one-session-out validation on the training sessions only.

**4. Calibration.** Ridge shrinks predictions toward the average distance (too far when objects are close, too near when far). A linear correction in log space is fit on out-of-fold training predictions only, never on test data.

**5. Export.** The DINOv2 feature extractor is exported with `torch.export` and lowered to an ExecuTorch `.pte` program with the XNNPACK backend, in fp32 and int8. The position encoding is precomputed for the fixed input size so the export contains no bicubic interpolation.

**6. Live inference.** The live server reads the sensor and runs the ExecuTorch model on each incoming frame; the laptop shows the video at full frame rate and overlays the latest prediction.

---

## Results

### Accuracy (cm, on data the model never trained on)

Trained on 3 objects (notebook, phone, hand; 657 frames, 3 recording sessions).

| Test set | Frames | Mean error | Median error | Within 5 cm | Baseline error |
|---|---|---|---|---|---|
| Same notebook, new recording | 220 | **4.2** | 3.9 | 67% | 12.3 |
| Coffee mug, never seen in training | 159 | **10.2** | 9.9 | 10% | 13.1 |

Baseline = always predicting the median training distance (36 cm).

**Reading:** the model is 3x better than the baseline on a seen object in a new recording, but only slightly better on an unseen object. With three training objects it partly learns *how large each known object looks* rather than distance in general. More, and more varied, training objects is the direct fix.

### Speed and size (DINOv2-small feature extractor on the board's CPU)

| Runtime | Size | Latency p50 | Latency p95 | Seen-object error | Unseen-object error |
|---|---|---|---|---|---|
| PyTorch eager, fp32 | 86.7 MB | 870 ms | 975 ms | 4.2 cm | 10.2 cm |
| **ExecuTorch XNNPACK, fp32** | 86.7 MB | **242 ms** | 532 ms | **4.2 cm** | 10.2 cm |
| ExecuTorch XNNPACK, int8 (PTQ) | **22.0 MB** | 397 ms | 425 ms | 10.4 cm | 6.4 cm |

**Decision: we deploy ExecuTorch fp32.**
- It is **3.6x faster than PyTorch** with identical accuracy (outputs match to 3e-5), which also verifies the export.
- int8 is 4x smaller but **1.6x slower** and loses 6 cm on the seen object. In a vision transformer only the matrix multiplies are quantized; layer norms, softmax, and GELU stay in float, so the program keeps converting between int8 and float, and that overhead outweighs the gain at this model size.
- int8's better mug number is **not** better generalization: the regression head was trained on fp32 features, and quantization shifts every feature, which happened to move the mug's predictions closer while degrading the notebook's.

---

## Key design decisions

| Decision | Why |
|---|---|
| Split evaluation by recording session, never by random frame | Neighbouring frames are near-duplicates; a random split leaks and inflates accuracy |
| Hold out one object entirely (the mug) | Tests whether the model learned distance or just memorized object sizes |
| Regress log distance | Apparent size is proportional to 1/distance; log makes it closer to linear for a linear head |
| Frozen DINOv2 + linear head instead of fine-tuning | ~650 labeled frames is too little to fine-tune a ViT; a linear head trains in seconds on the board |
| CLS token + centre patches as the feature | The centre patches cover the same region the sensor measures |
| Calibration fit on out-of-fold training predictions | Corrects shrinkage toward the mean without touching test data |
| Precomputed position encoding before export | Fixed input size; avoids exporting an interpolation op |
| Sensor readings outside 50 to 1200 mm treated as invalid | Below 5 cm the camera sees no usable image; beyond ~1.2 m the sensor is unreliable |

---

## Limitations

- **Range:** all training data is between ~1 and 63 cm, so predictions are only meaningful in that range.
- **Single-point labels:** the sensor gives one distance per frame, so the model estimates the distance of the object at the centre of the frame, not a full depth map.
- **Unseen objects:** generalization to new objects is weak with three training objects (see results).
- **Camera on the laptop:** frames come from the laptop webcam and are processed on the board. A USB camera plugged into the board would make the system fully self-contained.
- **NPU not used:** the Hexagon NPU needs a QNN-lowered model, and QNN export currently requires an x86 host with the board's matching SDK. All inference here runs on the CPU through XNNPACK.

---

## Repository layout

```
linux/collector_server.py   board: receives frames, reads the sensor, saves (frame, distance) pairs
linux/live_server.py        board: live demo server (sensor reading + model prediction per frame)
laptop/capture_client.py    laptop: streams webcam frames for data collection
laptop/live_client.py       laptop: live demo view, video at full frame rate + prediction overlay
ml/depth_ml.py              board: DINOv2 embedding, training, calibration, evaluation, live predictor
ml/export_et.py             board: ExecuTorch export (fp32 / int8), benchmark, accuracy check
mcu/depth-sensor/sketch.ino MCU: reads the distance sensor, exposes get_distance() over Bridge
```

Data (`~/depth_data/`) and trained models (`ml/model/`, `*.pte`) stay on the board and are not in git.

---

## How to run

### Board setup (once)

```bash
python3 -m venv --without-pip ~/venv && source ~/venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install numpy msgpack pandas pillow torch executorch
```

Flash `mcu/depth-sensor/sketch.ino` from Arduino App Lab (needs the Modulino and Arduino_RouterBridge libraries) and keep the app running. Copy `linux/*.py` to `~/` and `ml/*.py` to `~/ml/` on the board.

### 1. Collect data (one session per object)

Board:
```bash
python ~/collector_server.py --session train_notebook
```
Laptop:
```bash
pip install opencv-python
adb forward tcp:5000 tcp:5000
python laptop/capture_client.py
```
SPACE starts and stops recording, q quits. Sweep the object slowly across the range at the crosshair. Restart the board server with a new `--session` name for every object.

### 2. Train and evaluate

```bash
cd ~/ml
python depth_ml.py embed train_notebook train_phone train_hand val_notebook2 test_mug
python depth_ml.py train --train train_notebook train_phone train_hand --test val_notebook2 test_mug
python depth_ml.py bench
```

### 3. Export to ExecuTorch

```bash
python export_et.py --int8
```

### 4. Live demo

Board:
```bash
python ~/live_server.py
```
Laptop:
```bash
adb forward tcp:5000 tcp:5000
python laptop/live_client.py
```

### Data format

```
~/depth_data/<session>/frames/000123.jpg
~/depth_data/<session>/labels.csv    idx, file, t_capture_laptop, t_recv_board, distance_mm, rpc_ms
```

---

## What we would do next

1. Train on 10+ varied objects to close the unseen-object gap.
2. Mount a USB camera on the board so the whole system is self-contained.
3. Lower the model to the Hexagon NPU through QNN and compare against the CPU numbers.
4. Replace the single-point sensor with a multi-zone time-of-flight sensor to supervise a coarse depth map instead of one distance.
