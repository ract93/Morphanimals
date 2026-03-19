import os

import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


_VIDEO_SPECS = [
    ("strength",               "viridis", 0,   100, "strength_map.mp4"),
    ("hardiness",              "viridis", 0,   100, "hardiness_map.mp4"),
    ("age",                    "viridis", 0,    50, "age_map.mp4"),
    ("lifespan",               "inferno", 0,   100, "lifespan_map.mp4"),
    ("metabolism",             "inferno", 0,   100, "metabolism_map.mp4"),
    ("reproduction_threshold", "magma",   0,    50, "reproduction_threshold_map.mp4"),
    ("speed",                  "plasma",  0,    20, "speed_map.mp4"),
    ("genetic_distance",       "magma",   0,    50, "genetic_drift_map.mp4"),
    ("color",                  None,      None, None, "species_map.mp4"),
]


def _to_rgb(raw, cmap_name, vmin, vmax):
    arr = np.array(raw)
    if cmap_name is None:
        return (arr * 255).astype(np.uint8)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return (plt.get_cmap(cmap_name)(norm(arr))[:, :, :3] * 255).astype(np.uint8)


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))
    plt.close(fig)


class FrameWriter:
    """Streams frames directly to an MP4 file — no in-memory accumulation."""

    def __init__(self, path, frame_rate, cmap_name, vmin, vmax):
        self._writer = imageio.get_writer(path, fps=frame_rate, macro_block_size=1)
        self._cmap = cmap_name
        self._vmin = vmin
        self._vmax = vmax

    def write(self, raw):
        self._writer.append_data(_to_rgb(raw, self._cmap, self._vmin, self._vmax))

    def close(self):
        self._writer.close()


def open_frame_writers(videos_dir, frame_rate):
    """Open one streaming FrameWriter per attribute. Call writer.close() when done."""
    return {
        attr: FrameWriter(os.path.join(videos_dir, fname), frame_rate, cmap, vmin, vmax)
        for attr, cmap, vmin, vmax, fname in _VIDEO_SPECS
    }


def save_capture_images(captures, images_dir):
    """Write PNG snapshots from the captures dict {attr: [(step, raw), ...]}."""
    for attr, cmap_name, vmin, vmax, _ in _VIDEO_SPECS:
        for step, raw in captures.get(attr, []):
            imageio.imwrite(
                os.path.join(images_dir, f"{attr}_step_{step}.png"),
                _to_rgb(raw, cmap_name, vmin, vmax),
            )
