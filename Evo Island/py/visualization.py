import os
from concurrent.futures import ThreadPoolExecutor

import imageio
import matplotlib.pyplot as plt
import numpy as np

from genes import VIDEO_SPECS


def _build_lut(cmap_name, vmin, vmax, n=256):
    """Pre-build a 256×3 uint8 LUT for a given colormap and value range."""
    cmap = plt.get_cmap(cmap_name)
    lut = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
    return lut, vmin, vmax


def _apply_lut(arr, lut, vmin, vmax):
    """Map a float32 H×W array to uint8 H×W×3 RGB via a pre-built LUT."""
    n = len(lut)
    indices = np.clip(
        ((arr - vmin) / (vmax - vmin) * (n - 1)).astype(np.int32),
        0, n - 1,
    )
    return lut[indices]


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))
    plt.close(fig)


def encode_videos(frames_dir, videos_dir, frame_rate):
    """Post-process: load saved .npy frames and encode one MP4 per attribute.
    All attributes are encoded in parallel via ThreadPoolExecutor."""

    def encode_one(spec):
        attr, cmap, vmin, vmax, fname = spec
        attr_dir = os.path.join(frames_dir, attr)
        if not os.path.isdir(attr_dir):
            return
        frame_files = sorted(
            f for f in os.listdir(attr_dir) if f.endswith(".npy")
        )
        if not frame_files:
            return
        out_path = os.path.join(videos_dir, fname)
        writer_kwargs = dict(fps=frame_rate, macro_block_size=1, codec='libx264', pixelformat='yuv420p')
        if cmap is None:
            # Raw RGB [0,1] array — just scale to uint8 directly.
            with imageio.get_writer(out_path, **writer_kwargs) as writer:
                for fname_frame in frame_files:
                    writer.append_data((np.load(os.path.join(attr_dir, fname_frame)) * 255).astype(np.uint8))
        else:
            lut, v0, v1 = _build_lut(cmap, vmin, vmax)
            with imageio.get_writer(out_path, **writer_kwargs) as writer:
                for fname_frame in frame_files:
                    writer.append_data(_apply_lut(np.load(os.path.join(attr_dir, fname_frame)), lut, v0, v1))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(encode_one, spec) for spec in VIDEO_SPECS]
        for f in futures:
            f.result()


def save_capture_images(captures, images_dir):
    """Write PNG snapshots from the captures dict {attr: [(step, raw), ...]}."""
    for attr, cmap, vmin, vmax, _ in VIDEO_SPECS:
        for step, raw in captures.get(attr, []):
            if cmap is None:
                rgb = (np.array(raw) * 255).astype(np.uint8)
            else:
                lut, v0, v1 = _build_lut(cmap, vmin, vmax)
                rgb = _apply_lut(np.array(raw), lut, v0, v1)
            imageio.imwrite(
                os.path.join(images_dir, f"{attr}_step_{step}.png"),
                rgb,
            )
