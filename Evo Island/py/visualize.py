"""Stand-alone visualization script.

    python visualize.py path/to/simulation.h5 [--output-dir path/]

Reads a simulation.h5 file and produces:
  - Videos/       — one MP4 per gene attribute (via file-based ffmpeg)
  - Images/       — PNG capture snapshots
  - Game_World.png
  - phylogeny.png

Because this script is separate from the simulation, a visualization bug
can never abort or corrupt a simulation run.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import h5py
import imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Allow importing genes when run as a CLI script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genes import VIDEO_SPECS


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def _build_lut(cmap_name, vmin, vmax, n=256):
    cmap = plt.get_cmap(cmap_name)
    lut  = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
    return lut, vmin, vmax


def _apply_lut(arr, lut, vmin, vmax):
    n = len(lut)
    indices = np.clip(
        ((arr - vmin) / (vmax - vmin) * (n - 1)).astype(np.int32),
        0, n - 1,
    )
    return lut[indices]


def _to_rgb(raw, cmap, vmin, vmax):
    if cmap is None:
        return (np.array(raw, dtype="float32") * 255).astype(np.uint8)
    lut, v0, v1 = _build_lut(cmap, vmin, vmax)
    return _apply_lut(np.array(raw, dtype="float32"), lut, v0, v1)


# ---------------------------------------------------------------------------
# MP4 encoding via file-based ffmpeg (avoids Windows pipe-size limits)
# ---------------------------------------------------------------------------

def _frames_to_mp4(frames_ds, cmap, vmin, vmax, output_path, frame_rate):
    """Write frames dataset (F, H, W[, 3]) to an MP4 via temp PNGs + ffmpeg."""
    n_frames = frames_ds.shape[0]
    if n_frames == 0:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_frames):
            rgb = _to_rgb(frames_ds[i], cmap, vmin, vmax)
            imageio.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), rgb)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-framerate", str(frame_rate),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path}:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Terrain world map
# ---------------------------------------------------------------------------

def save_matrix_image(matrix, file_path):
    """Save a 2D float matrix as a coloured viridis PNG."""
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    plt.colorbar(im, ax=ax)
    plt.savefig(str(file_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phylogenetic tree
# ---------------------------------------------------------------------------

def _save_phylogeny_png(phylo_arr, output_path):
    if phylo_arr.shape[0] == 0:
        return
    G = nx.DiGraph()
    G.add_node(1, step=0)
    for parent, child, step in phylo_arr:
        G.add_node(int(child), step=int(step))
        G.add_edge(int(parent), int(child))

    y_pos    = {}
    leaf_idx = [0]
    for node in nx.dfs_postorder_nodes(G, 1):
        children = list(G.successors(node))
        if not children:
            y_pos[node] = leaf_idx[0]
            leaf_idx[0] += 1
        else:
            y_pos[node] = sum(y_pos[c] for c in children) / len(children)

    x_pos    = {n: G.nodes[n]["step"] for n in G.nodes}
    n_leaves = leaf_idx[0]

    fig, ax = plt.subplots(figsize=(14, max(6, n_leaves * 0.25)))
    for parent, child in G.edges():
        px, py = x_pos[parent], y_pos[parent]
        cx, cy = x_pos[child],  y_pos[child]
        ax.plot([px, cx], [py, py], color="steelblue", lw=0.6, alpha=0.6)
        ax.plot([cx, cx], [py, cy], color="steelblue", lw=0.6, alpha=0.6)
    xs = [x_pos[n] for n in G.nodes]
    ys = [y_pos[n] for n in G.nodes]
    ax.scatter(xs, ys, s=8, color="steelblue", zorder=3)
    ax.set_xlabel("Timestep")
    ax.set_yticks([])
    ax.set_title(
        f"Phylogenetic Tree — {G.number_of_nodes()} species, "
        f"depth {nx.dag_longest_path_length(G)}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main visualize() function — callable from Python or CLI
# ---------------------------------------------------------------------------

def visualize(h5_path, output_dir=None):
    """Read *h5_path* and write all visualizations into *output_dir*."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(h5_path))

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        cfg        = json.loads(f.attrs["config"])
        frame_rate = cfg.get("frame_rate", 10)

        # Terrain world map
        if "terrain" in f and "world_matrix" in f["terrain"]:
            wm = np.array(f["terrain"]["world_matrix"])
            save_matrix_image(wm, os.path.join(output_dir, "Game_World.png"))

        # Frames → MP4
        if "frames" in f and len(f["frames"]) > 0:
            videos_dir = os.path.join(output_dir, "Videos")
            os.makedirs(videos_dir, exist_ok=True)
            for attr, cmap, vmin, vmax, fname in VIDEO_SPECS:
                if attr not in f["frames"]:
                    continue
                print(f"  Encoding {fname}...")
                _frames_to_mp4(
                    f["frames"][attr], cmap, vmin, vmax,
                    os.path.join(videos_dir, fname),
                    frame_rate,
                )

        # Captures → PNG
        if "captures" in f and len(f["captures"]) > 0:
            images_dir = os.path.join(output_dir, "Images")
            os.makedirs(images_dir, exist_ok=True)
            step_indices = (
                list(f["captures"]["step_indices"])
                if "step_indices" in f["captures"] else None
            )
            for attr, cmap, vmin, vmax, _ in VIDEO_SPECS:
                if attr not in f["captures"]:
                    continue
                ds = f["captures"][attr]
                for i in range(ds.shape[0]):
                    rgb = _to_rgb(np.array(ds[i], dtype="float32"), cmap, vmin, vmax)
                    step_label = step_indices[i] if step_indices is not None else i
                    imageio.imwrite(
                        os.path.join(images_dir, f"{attr}_step_{step_label}.png"),
                        rgb,
                    )

        # Phylogeny → PNG
        if "phylogeny" in f:
            _save_phylogeny_png(
                np.array(f["phylogeny"]),
                os.path.join(output_dir, "phylogeny.png"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations from a simulation HDF5 file."
    )
    parser.add_argument("h5_path", help="Path to simulation.h5")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same directory as h5_path)",
    )
    args = parser.parse_args()
    visualize(args.h5_path, args.output_dir)
