# Plan: HDF5 Output + Decoupled Visualization

## Goal
Separate simulation from visualization. The simulation writes a single HDF5 file as its
durable output. Visualization and analysis are independent scripts that read from it.
A visualization bug can never abort or corrupt a simulation run.

---

## HDF5 Schema

```
trial_1.h5
├── config                          # JSON-serialized config (string attribute)
├── metrics/
│   └── simulation_metrics          # Dataset: (T, N_cols) float32, columns as attribute
├── frames/
│   ├── hardiness                   # Dataset: (F, H, W) float32 — one slice per GIF frame
│   ├── lifespan
│   ├── strength
│   ├── metabolism
│   ├── reproduction_threshold
│   ├── speed
│   ├── trophism
│   ├── kin_attraction
│   ├── threat_response
│   ├── age
│   ├── genetic_distance
│   └── color                       # Dataset: (F, H, W, 3) float32
├── captures/
│   ├── hardiness                   # Dataset: (C, H, W) — one slice per PNG capture
│   └── ... (same attrs)
└── phylogeny                       # Dataset: (S, 3) int32 — [parent, child, step]
```

**Key decisions:**
- `frames/` uses chunked + gzip compression — spatial coherence compresses very well
- `captures/` same but fewer slices
- `phylogeny` is a simple 2D int array, trivially queryable
- Metrics stored as a single 2D array with column names as an HDF5 attribute
- Config stored as a JSON string attribute on the root group

---

## Implementation Steps

### 1. Add `h5py` dependency
- Add `h5py` to `requirements.txt`
- No C++ changes needed — HDF5 writing stays in Python

### 2. New `SimulationOutput` class (`py/output.py`)
Replaces both `SimulationMetrics` CSV logging and the frames/captures dicts.
- Opens HDF5 file on init, creates group structure
- `log_metrics(step, result)` — appends a row to `metrics/simulation_metrics`
- `log_frame(step, attr, matrix)` — appends a slice to `frames/<attr>`
- `log_capture(step, attr, matrix)` — appends a slice to `captures/<attr>`
- `log_phylogeny(events)` — writes phylogeny array on close
- `close()` — flushes and closes the file
- Uses chunked writes so data is durable mid-simulation (crash-safe)

### 3. Rewrite `simulation.py`
- Replace `SimulationMetrics` + CSV with `SimulationOutput`
- Replace `frames` dict + `captures` dict with `SimulationOutput.log_frame/log_capture`
- Remove GIF writing entirely from simulation loop
- Remove `save_gifs`, `save_capture_images` imports
- Remove `frame_save_interval` from the hot path (keep in config, pass to output)
- Output: one `trial_N.h5` file per trial, nothing else written during simulation

### 4. New `visualize.py` (standalone script)
Reads an HDF5 file and produces MP4s + PNGs. Can be run independently.

```
python visualize.py path/to/trial_1.h5 [--output-dir path/]
```

- Reads `frames/<attr>` → encodes MP4 using file-based ffmpeg (no pipe):
  - Write each frame as temp PNG → run ffmpeg on the image sequence → delete PNGs
  - Avoids the Windows pipe size limit entirely
  - Uses `libx264` + `yuv420p`
- Reads `captures/<attr>` → writes PNGs
- Reads `phylogeny` → draws phylogenetic tree, saves as PNG
- Reads `config` → gates outputs behind feature flags (same as current)

### 5. Update `experiment.py`
- After all trials complete, call `visualize.py` as a subprocess (or import and call directly)
- Notebook execution also moves here, after visualization
- If visualization fails, log the error but don't fail the experiment

### 6. Update notebooks
- Replace `pd.read_csv('./simulation_metrics.csv')` with `pd.read_hdf` or `h5py` reads
- `phylogeny.csv` → read from HDF5 `phylogeny` dataset directly
- Aggregate notebook reads across multiple `trial_N.h5` files

### 7. Remove dead code
- Delete `py/visualization.py` (replaced by `visualize.py`)
- Delete `save_gifs`, `encode_videos`, `save_capture_images`
- Delete `SimulationMetrics` CSV path (keep state string logic for terminal output)
- Remove `frame_save_interval` from being a CSV concern

---

## File Changes Summary

| File | Action |
|------|--------|
| `py/output.py` | **New** — HDF5 writer |
| `py/simulation.py` | Rewrite — use output.py, no visualization |
| `py/visualize.py` | **New** — standalone MP4/PNG producer |
| `py/experiment.py` | Update — call visualize.py after simulation |
| `py/visualization.py` | Delete |
| `py/metrics.py` | Simplify — keep terminal state string, remove CSV |
| `notebooks/*.ipynb` | Update — read HDF5 instead of CSV |
| `requirements.txt` | Add h5py |

---

## Notes
- Sexual reproduction (gene 10, `sexuality`) is planned but gated — implement after this refactor
- `predation_threshold` config key is now unused in C++ logic — can be removed from config.json
- Recompile C++ extension before testing (phylogeny log + check_speciation changes pending)
