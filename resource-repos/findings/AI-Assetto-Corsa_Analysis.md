# AI-Assetto-Corsa: Code and Implementation Assessment

## Summary
- **Goal (per README)**: Use imitation learning (IL) and reinforcement learning (RL) to drive in Assetto Corsa and achieve better lap times than an expert.
- **What’s actually implemented**: Two separate capabilities, not integrated into a driving agent:
  - Real-time lane/road segmentation via two different pipelines (a small Keras U-Net and a full LaneNet TF1 implementation) driven by screen capture.
  - Virtual gamepad/joystick control via `vJoy` DLL calls (Windows-only) with demo routines to move axes and press buttons.
- **Missing**: A closed-loop policy that consumes perception (lanes/road) and outputs control commands; any imitation learning or RL training code; the `main.py` referenced in the README; integration glue between perception and control.
- **Bottom line**: This is a partially implemented prototype. Perception and synthetic control are present as demos; the IL/RL driving agent and integration with Assetto Corsa are not implemented here.

---

## Repository structure (relevant parts)
`src/`
- `segmentation.py`: Keras U-Net-style segmentation model; runs on screen-captured frames; loads `final-road-seg-model-v2.h5`; displays thresholded predictions.
- `ac_state_variables.py`: Nearly identical to `segmentation.py` but at a different resolution; captures screen and runs the model; no downstream use of predictions.
- `input.py`: Pygame joystick input reader; prints axes/buttons/hats in a loop.
- `control_test.py`: `vJoy` Windows DLL wrapper via `ctypes`; demo functions to move axes, throttle/brake, and look left/right; includes a `__main__` demo.
- `final-road-seg-model-v2.h5`: Keras model weights used by `segmentation.py` and `ac_state_variables.py`.
- `vJoyInterface.dll`, `x360ce_x64.exe`, `xinput1_3.dll`, `x360ce.ini`: Windows-only virtual joystick components for control injection.

`src/lanenet/`
- Full LaneNet implementation (ported from `MaybeShewill-CV/lanenet-lane-detection`).
- Key files:
  - `lanenet_model/lanenet.py`, `lanenet_front_end.py`, `lanenet_back_end.py`: TF1 computation graph; front-end feature extractor (VGG/BiSeNetV2), back-end for binary+instance segmentation.
  - `lanenet_model/lanenet_postprocess.py`: Morphological filtering; DBSCAN clustering on embeddings; inverse perspective mapping (IPM) remap; polynomial lane fitting; overlay rendering.
  - `test_lanenet_video.py`: Real-time screen capture; TF1 placeholder/session inference; postprocess; displays and writes outputs.
  - `model/`: Contains Tusimple LaneNet weights (`tusimple_lanenet.ckpt*`, `checkpoint`).
  - `requirements.txt`: numpy, tqdm, glog, easydict, tensorflow_gpu, matplotlib, opencv, scikit_learn, loguru.

`README.md`
- Short description of IL+RL workflow and a one-line “Getting Started: `python main.py`” (no such `main.py` exists in this repo).

---

## How the pieces work

### 1) Perception via segmentation (two implementations)

- `segmentation.py` (Keras/TensorFlow 2 style):
  - Builds a small U-Net-like model (downsample conv blocks with max-pooling → upsample via transposed convolutions with skip connections; final 1-channel sigmoid).
  - Loads weights from `final-road-seg-model-v2.h5`.
  - Captures a fixed screen region using `mss` with a hard-coded bounding box sized to the author’s monitor setup (`width=1400`, `height=1000`).
  - Runs `model.predict` each frame, thresholds outputs at 0.5, and displays both the source frame and a 3-channelized prediction mask via OpenCV windows.
  - Notes:
    - GPU memory growth is enabled if a GPU is present.
    - Outputs are visual; there is no downstream control.

- `ac_state_variables.py` (similar variant):
  - Same basic model structure and `mss` capture but at `width=600`, `height=400`.
  - Produces predictions but does not visualize the mask by default (mask show is commented out); also not used for control.

- `lanenet/` (TensorFlow 1):
  - `test_lanenet_video.py` constructs a TF1 graph:
    - `LaneNet.inference(input_tensor)` to produce binary and instance segmentation maps.
    - `LaneNetPostProcessor.postprocess(...)` to:
      - Morphologically close regions, connected-components prune small blobs.
      - Cluster pixel embeddings with DBSCAN to separate individual lanes.
      - Apply inverse perspective mapping using `data/tusimple_ipm_remap.yml`.
      - Fit polynomials per-lane and overlay sampled lane points on the source image.
  - Captures frames with `mss`, resizes to `512x256`, normalizes, and runs session inference; shows and saves overlays.
  - Notes:
    - Path to weights is hardcoded to a Windows drive: `D:\...\tusimple_lanenet.ckpt` while a valid `model/` folder exists in-repo; needs path fix.
    - Strict TF1 session/placeholder code; requires TF1 GPU build if using GPU flags.

### 2) Control via virtual joystick (Windows-only)

- `control_test.py`:
  - Wraps `vJoyInterface.dll` with `ctypes` for virtual joystick control.
  - Exposes helpers to open/close the vJoy device, pack axis/button states into the joystick struct, and update the device.
  - Provides demo functions: `test()`, `test2()`, `test3()`, `test5()`, `look_left()`, `look_right()`, `throttle()`, `reverse_brake()`, `ultimate_release()`.
  - The `__main__` block runs `test()` which sweeps axes sinusoidally.
  - Works alongside `x360ce_x64.exe` and `xinput1_3.dll` to emulate an Xbox controller for games.

### 3) Input reading

- `input.py`:
  - Initializes Pygame and enumerates joystick axes, buttons, and hats.
  - Reads and prints raw input values at 60 Hz.
  - Useful for debugging gamepad input mapping, not used by any agent.

---

## What’s missing vs. the stated goal
- **End-to-end driving agent**: There is no policy that takes lane/road perception and outputs steering/throttle/brake in a closed loop.
- **Imitation Learning**: No dataset collection, behavior cloning scripts, loss definitions, or training loops for a driving policy.
- **Reinforcement Learning**: No environment wrapper for Assetto Corsa with observations/actions/rewards; no RL algorithms (e.g., DQN, SAC, PPO) or trainers.
- **Integration glue**: No code to wire perception outputs to the control layer (`vJoy`) or to interpret lane geometry into steering commands.
- **Entry point**: `README.md` references `python main.py` but no such file exists in this repo.
- **Cross-platform control**: Virtual joystick mechanism is Windows-specific; no alternative provided for macOS/Linux.
- **Config/runtime hygiene**: Hard-coded screen capture bounding boxes and Windows paths; mixed TF1 (lanenet) and TF2/Keras (u-net) stacks; no unified environment or launcher.

---

## Extent of implementation (practical readiness)
- **Perception**:
  - Keras U-Net demo is runnable if the correct Python/TensorFlow 2 and GPU/CPU environment is present and the `mss` capture bounding box matches the user’s display. It provides a binary mask visualization only.
  - LaneNet demo is runnable with TF1 and correct weight path; produces more structured lane geometry overlays and could serve as a foundation for control.
- **Control**:
  - The vJoy demo works on Windows with proper vJoy/x360ce setup. It can command axes/buttons and therefore can be used to inject game controls once a policy exists.
- **Integration**: Not present. There is no control logic that uses the segmentation output to compute steering/throttle or any safety/governor logic.
- **IL/RL**: Not present. No training, evaluation, or data pipelines for a driving policy.

Overall, this repository is a **prototype** with working perception demos and a working (Windows-only) virtual control demo, but without the connective tissue or learning components required to realize the stated IL+RL goal.

---

## How to run the existing demos (high-level)
- Keras U-Net segmentation (`segmentation.py`):
  1. Install Python 3, TensorFlow 2.x and dependencies (`opencv-python`, `mss`, `numpy`).
  2. Ensure `final-road-seg-model-v2.h5` is present (it is in `src/`).
  3. Adjust `bounding_box` values to your display; run `python segmentation.py` from `src/`.

- LaneNet segmentation (`lanenet/test_lanenet_video.py`):
  1. Use Python 3 + TensorFlow 1.x environment matching `lanenet/requirements.txt`.
  2. Fix `weights_path` to point to `src/lanenet/model/tusimple_lanenet.ckpt` (relative path).
  3. Ensure `data/tusimple_ipm_remap.yml` exists (it does in `lanenet/data/`).
  4. Adjust `bounding_box`; run `python lanenet/test_lanenet_video.py` from `src/`.

- vJoy control demo (`control_test.py`, Windows only):
  1. Install vJoy driver and configure a virtual device; keep `vJoyInterface.dll` accessible.
  2. Optionally configure `x360ce_x64.exe` to emulate an Xbox controller in Assetto Corsa.
  3. Run `python control_test.py` to see axis movements; adapt `setJoy(...)` for control injection.

---

## Risks and limitations
- **OS-specific control**: The control path depends on Windows vJoy/x360ce and will not run on macOS/Linux without alternatives.
- **Mixed TF stacks**: TF1 (lanenet) and TF2/Keras (u-net) complicate environment setup.
- **Hard-coded paths/geometry**: Screen capture areas and weight paths are author-specific.
- **Performance**: Real-time TF1/TF2 inference performance depends on GPU availability; CPU-only may be insufficient for high FPS.

---

## What would be needed to complete the project
- Define an observation model: e.g., lane centerline offset, heading error, curvature from LaneNet postprocess.
- Implement control mapping: convert geometry to steering/brake/throttle commands; close the loop at 20–60 Hz via vJoy.
- Add an environment wrapper for Assetto Corsa: screen capture + telemetry (if available) → observation; vJoy → action; lap time and penalties → rewards.
- Implement IL (behavior cloning from expert laps) and an RL algorithm (e.g., DQfD/SAC/PPO) with logging and evaluation.
- Provide a real `main.py` (or CLI) to orchestrate modes: collect → train → evaluate → run.
- Parameterize configs (resolutions, capture regions, model paths) and document setup. 