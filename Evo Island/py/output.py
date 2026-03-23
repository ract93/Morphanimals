"""SimulationOutput — writes all simulation data to a single HDF5 file.

Replaces SimulationMetrics CSV logging and the frames/captures dicts.
Metric rows are flushed after each step so a crash mid-run preserves all
completed steps.  Visualization reads this file independently; a viz bug
can never abort or corrupt a simulation run.
"""

import json

import h5py
import numpy as np

from genes import GENES, column_name


METRIC_COLUMNS = (
    ["Timestep", "Population Count", "Cumulative Deaths",
     "Deaths from Aging", "Deaths from Competition",
     "Deaths from Starvation", "Deaths from Exposure", "Deaths from Predation",
     "Average Age"]
    + [column_name(attr) for attr, *_ in GENES]
    + ["Number of Species"]
)


class SimulationOutput:
    """Opens *h5_path* for writing and streams simulation data into it.

    Call order per step:
        log_metrics(metrics_obj, timestep)
        log_frame(attr, matrix)        # every frame_save_interval steps
        log_capture(attr, matrix)      # at capture checkpoints
        note_capture_step(step)        # once per capture event (all attrs)

    Call once at the end:
        log_phylogeny(events)
        close()
    """

    def __init__(self, h5_path, cfg):
        self._f = h5py.File(h5_path, "w")
        self._f.attrs["config"] = json.dumps(cfg)

        n_cols = len(METRIC_COLUMNS)
        mg = self._f.create_group("metrics")
        self._metrics_ds = mg.create_dataset(
            "simulation_metrics",
            shape=(0, n_cols),
            maxshape=(None, n_cols),
            dtype="float64",
            chunks=(256, n_cols),
        )
        self._metrics_ds.attrs["columns"] = METRIC_COLUMNS

        self._frames_g          = self._f.create_group("frames")
        self._captures_g        = self._f.create_group("captures")
        self._capture_step_list = []

    # ------------------------------------------------------------------
    # Terrain (written once before the step loop)
    # ------------------------------------------------------------------

    # lzf is bundled with h5py and is 5-10x faster than gzip at comparable
    # compression ratios for sparse float32 spatial data.  Terrain is written
    # once so gzip-4 is fine here (not on the hot path).
    def log_terrain(self, world_matrix, food_matrix):
        tg = self._f.create_group("terrain")
        tg.create_dataset(
            "world_matrix",
            data=np.array(world_matrix, dtype="float32"),
            compression="gzip", compression_opts=4,
        )
        tg.create_dataset(
            "food_matrix",
            data=np.array(food_matrix, dtype="float32"),
            compression="gzip", compression_opts=4,
        )

    # ------------------------------------------------------------------
    # Per-step metrics
    # ------------------------------------------------------------------

    def log_metrics(self, metrics_obj, timestep):
        row = [
            timestep,
            metrics_obj.population_count,
            metrics_obj.cumulative_deaths,
            metrics_obj.deaths_from_aging,
            metrics_obj.death_from_competition,
            metrics_obj.deaths_from_starvation,
            metrics_obj.deaths_from_exposure,
            metrics_obj.deaths_from_predation,
            metrics_obj.average_age,
        ]
        for attr, *_ in GENES:
            row.append(getattr(metrics_obj, f"average_{attr}"))
        row.append(metrics_obj.species_counts)

        ds = self._metrics_ds
        n  = ds.shape[0]
        ds.resize((n + 1, ds.shape[1]))
        ds[n] = row
        self._f.flush()

    # ------------------------------------------------------------------
    # Frames and captures
    # ------------------------------------------------------------------

    def _append_slice(self, group, attr, matrix):
        arr = np.array(matrix, dtype="float32")
        if attr not in group:
            # lzf: bundled with h5py, no extra deps, ~5x faster than gzip-4
            # at equivalent ratio for sparse agent maps.  One chunk = one frame
            # so sequential writes and random per-frame reads are both O(1).
            if arr.ndim == 2:
                H, W = arr.shape
                group.create_dataset(
                    attr,
                    shape=(0, H, W),
                    maxshape=(None, H, W),
                    dtype="float32",
                    chunks=(1, H, W),
                    compression="lzf",
                )
            else:  # color: (H, W, 3)
                H, W, C = arr.shape
                group.create_dataset(
                    attr,
                    shape=(0, H, W, C),
                    maxshape=(None, H, W, C),
                    dtype="float32",
                    chunks=(1, H, W, C),
                    compression="lzf",
                )
        ds = group[attr]
        n  = ds.shape[0]
        ds.resize((n + 1,) + ds.shape[1:])
        ds[n] = arr

    def log_frame(self, attr, matrix):
        self._append_slice(self._frames_g, attr, matrix)

    def log_capture(self, attr, matrix):
        self._append_slice(self._captures_g, attr, matrix)

    def note_capture_step(self, step):
        """Record the simulation step for the most recent capture event."""
        self._capture_step_list.append(step)

    # ------------------------------------------------------------------
    # Phylogeny (written once, just before close)
    # ------------------------------------------------------------------

    def log_phylogeny(self, events):
        """Write speciation events as a (N, 3) int32 array [parent, child, step]."""
        if events:
            arr = np.array(
                [[e["parent_species"], e["child_species"], e["step"]] for e in events],
                dtype="int32",
            )
        else:
            arr = np.empty((0, 3), dtype="int32")
        self._f.create_dataset("phylogeny", data=arr)

    # ------------------------------------------------------------------

    def close(self):
        if self._capture_step_list:
            self._captures_g.create_dataset(
                "step_indices",
                data=np.array(self._capture_step_list, dtype="int32"),
            )
        self._f.flush()
        self._f.close()
