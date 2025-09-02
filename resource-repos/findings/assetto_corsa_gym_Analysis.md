# Assetto Corsa Gym - Architecture and Implementation Review

This document summarizes how `assetto_corsa_gym` works and assesses the extent of its implementation, based on the repository README and the code under `assetto_corsa_gym/assetto_corsa_gym`.

## High-level overview
- The project exposes Assetto Corsa as an RL-ready environment using the classic Gym API.
- A Python plugin (running inside Assetto Corsa) streams telemetry to a UDP server and accepts control commands.
- The Gym environment (`AssettoCorsaEnv`) connects to the plugin, builds observations, computes rewards, handles terminations, and logs per-episode telemetry.
- Training entrypoint (`train.py`) wires the environment with a PyTorch SAC baseline (optionally DisCor) and integrates W&B logging.

## Code structure (selected)
- `assetto_corsa_gym/README.md`: Usage, benchmarks, dataset links, and setup.
- `assetto_corsa_gym/config.yml`: All configuration knobs for env and training.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py`: Core Gym env logic.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/ac_client.py`: UDP client for state/control + TCP management API.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/assettoCorsa.py`: Factory building the env from config.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/{track.py, reference_lap.py, gap.py, ...}`: Track geometry, racing line, occupancy grid, ray sensors, and gap computation.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/`: Assetto Corsa plugin sources (UDP ego server, simulation management, vJoy control, screen capture).
- `algorithm/discor`: SAC/DisCor implementation (networks, buffer, agent loop).
- `common/`: Logging/W&B utilities and misc helpers used by `train.py`.

## Data flow end-to-end
1. Assetto Corsa runs the `sensors_par` plugin, which:
   - Spawns an `EgoServer` UDP server exposing the current car state at `ego_sampling_freq` (default 25 Hz).
   - Runs a TCP Simulation Management server for reset, config, and static info (`get_static_info`, `get_config`, `get_track_info`).
   - (Optionally) executes vJoy control locally if `vjoy_executed_by_server=True`.
   - (Optionally, Windows) publishes screen-captured images via shared memory using a dual buffer.
2. The Gym environment (`AssettoCorsaEnv`):
   - Opens a UDP client (`ac_client.Client`) to receive state and send control commands. It also talks to the management TCP server for resets and metadata.
   - Maintains action state with optional relative actions and per-channel rate limits calibrated by per-car steer maps.
   - Builds vector observations: normalized telemetry subset + optional ray-casting sensors + curvature lookahead + previous actions + last applied action + out-of-track flag + optional task one-hot + optional previous obs and extra fields.
   - Computes rewards: primarily speed-scaled (km/h to m/s) with optional racing-line gap modulation and action smoothness penalty.
   - Handles terminations: out-of-track (plugin or occupancy grid), low-speed timeout, max steps, lap limits, or AC lap end.
   - Logs per-episode stats and saves raw trajectory data to Parquet alongside static info.
3. `train.py` creates the env from `config.yml`, instantiates SAC/DisCor, sets up logging, optionally pre-trains from human trajectories, then trains or evaluates.

## Key components and behavior

### Environment construction
- Factory: `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/assettoCorsa.py`.
- `ModuleConfig` clones `cfg.AssettoCorsa`, sets remote IP/ports, computes `max_episode_steps = max_episode_py_time * ego_sampling_freq`, locates `AssettoCorsaConfigs` for cars/tracks.
- `make_ac_env` returns an `AssettoCorsaEnv` bound to the working directory for logs.

### Networking and control
- UDP control/state (`ac_client.Client`):
  - `setup_connection()` sends `connect` until `identified` is received.
  - `get_servers_input()` receives state JSON; on arrival and if screen capture enabled, pulls the image from dual-buffer shared memory.
  - `DriverControls` carries `steer`, `acc`, `brake`, and gear shift flags; either applied locally via vJoy (`Controls`) or sent to the server for execution.
  - `SimulationManagement` (TCP) handles `reset`, `get_track_info`, `get_static_info`, and `get_config`.

### Observations
- Base telemetry features (`obs_enabled_channels`) normalized by per-channel scales, e.g.: speed, RPM, accel, yaw rate, local velocities, slip angles, gear.
- Optional ray sensors: `SensorsRayCasting` builds walls from track borders and answers distances to the nearest border over evenly spaced rays.
- Out-of-track flag: from AC tyres-out or computed using occupancy grid (`TrackOccupancyGrid`) over a precomputed pickle map.
- Curvature lookahead vector: from the reference lap (`ReferenceLap`) interpolated to distance; vector length and lookahead distance are configurable constants.
- Action history: previous absolute controls and the last applied action are appended.
- Optional: previous obs window, task one-hot, and extra channels (`obs_extra` separate array when enabled).

Resulting `observation_space` is an unbounded `Box`; shape is dynamically composed based on enabled components.

### Actions
- Action space: `Box([-1, -1, -1], [1, 1, 1])` for [steer, accel, brake].
- Relative action mode: deltas are scaled by per-channel rate limits (deg/s for steer scaled by the car’s `steer_map.csv`, pedals per-step), then clipped to simulator min/max.
- Absolute mode: actions are interpreted directly within simulator min/max.

### Reward
- Base: proportional to speed (km/h) normalized by 300.
- Optional: multiplied by `(1 - |gap| / 12)` using signed lateral gap to racing line.
- Optional: L2 penalty on action difference when `penalize_actions_diff` is enabled.

### Termination and truncation
- AC signals lap-end via `state["done"]`.
- Manual termination flags:
  - Out-of-track (either plugin tyres-out or occupancy grid result) if `enable_out_of_track_termination`.
  - Low-speed timeout: if speed < `TERMINAL_LIMIT_PROGRESS_SPEED` for `TERMINAL_JUDGE_TIMEOUT` seconds.
  - Max steps (`TimeLimit.truncated`) and max lap count.
  - Optional gap threshold (`max_gap`).
- On done, can auto-recover the car via `recover_car()` if configured.

### Track, racing line, and geometry
- Track borders and racing line are loaded from CSVs referenced in `AssettoCorsaConfigs/tracks/config.yaml`.
- Occupancy grid (pickle) encodes inside/outside-of-track via cell lookup; supports CPU and Torch tensors.
- Reference lap computes: cumulative distance, yaw, curvature; provides distance-indexed channels and lookahead slices.
- Gap to racing line computed on CPU/Torch: signed lateral distance to the nearest segment, handling segment overlap.

### Plugin internals (inside AC)
- `sensors_par.py` orchestrates:
  - UDP ego server (`ego_server.py`): manages a single client, locks, applies control (if server-side), maintains sampling cadence and sends JSON-serialized car state.
  - TCP simulation management server: responds to reset/static/config/track info requests.
  - Optional screen capture thread using named Windows events and a dual-buffer shared memory mechanism (Windows only).
  - vJoy-based local control application on Windows/Linux via separate backends.

### Training integration
- `train.py`:
  - Loads `config.yml` (OmegaConf), applies CLI overrides (dotlist), sets up a per-run `work_dir`.
  - Creates the env, asserts CUDA availability, instantiates SAC or DisCor with dimensions from the env.
  - Optional W&B `Logger` for structured metrics and artifact logging.
  - Offline data loading: reads laps from dataset paths, rebuilds observations/rewards with current env config, converts logged absolute controls to (relative) actions via `inverse_preprocess_actions`, and seeds replay buffer; supports ensemble buffer.
  - Training loop (`discor.agent.Agent`): interleaves action application (`set_actions`) and asynchronous `step(None)` with concurrent gradient updates; regular checkpointing and summary logging; evaluation uses `set_eval_mode()` and `evaluate()`.

## Extent of implementation and readiness
- Networking/control: Implemented and robust for single-client training; supports both client-side vJoy (local) and server-side execution.
- Env API: Complete Gym-style API with well-defined `action_space`, `observation_space`, `reset`, `step`, `close`, and info dict extras.
- Observations: Rich, configurable vector observations with sensors and history; optional image stream provided, but env doesn’t integrate pixels into the observation by default (images available via `get_current_image()`).
- Reward/termination: Implemented with configurable components; defaults promote lane-centering and speed.
- Track assets: Several tracks provided with CSV borders, racing line, and occupancy grids; easy to add new tracks via provided notebooks/instructions.
- Algorithms: Full SAC/DisCor implementations (PyTorch), compatible with env dimensions and action ranges; replay buffers support n-step returns and offline pretraining.
- Logging and outputs: Per-episode raw Parquet logs with static info, episode summaries (`episodes_stats.csv`), and optional W&B logging; periodic checkpointing.
- Screen capture: Implemented in plugin (Windows); dual-buffer shared memory with named events; README includes usage steps. Image retrieval in env is available via client.

## Notable considerations and limitations
- OS support:
  - Training loop asserts CUDA (`assert device.type == "cuda"`), so GPU is required.
  - Screen capture code path is Windows-only; the README mentions Linux support via Proton (April 2025 update), but images require Windows APIs.
- Single client: `ego_server` assumes one active client at a time; it will switch to a new client if a second connects.
- Image observations: The env currently offers images externally (`get_current_image`) rather than as part of the observation vector; integrating pixels would require extending the env/state pipeline.
- Reference speed: `use_target_speed` exists but defaults to False; when enabled, it appends a target-speed lookahead to observations.
- vJoy dependency: Local control via vJoy requires OS-specific setup (Windows drivers / Linux vjoy backend).

## How to use (quick start)
- Install plugin in AC and Python deps per `README.md`/`INSTALL.md`.
- Ensure track assets (CSV + occupancy grid + racing line) are in `AssettoCorsaConfigs/tracks`.
- Start AC with the target car/track and ensure the plugin loads.
- From repo root:
  - Train default setup: `python train.py`
  - Train another combo: `python train.py AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3`
  - Evaluate a checkpoint: `python train.py --test --load_path <path>/model/checkpoints/step_XXXXXX AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3`
  - Offline pretrain: `python train.py load_offline_data=True Agent.use_offline_buffer=True dataset_path=<path>`

## Conclusion
The `assetto_corsa_gym` project provides a complete and practical bridge between Assetto Corsa and RL training, with:
- A fully functional Gym environment, configurable observations, and a reasonable reward/termination scheme.
- Production-ready plugin servers (ego + management), optional screen capture, and both client/server control execution modes.
- End-to-end training with SAC/DisCor, replay buffers, offline data ingestion, and logging/checkpointing.

Areas for potential extension include: integrating image observations into the training pipeline, broadening OS-native image capture support, and adding more baseline algorithms. 