import os
from concurrent.futures import ThreadPoolExecutor

import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Matrix Creation, Visualization, Saving
def transform_matrix(agent_matrix, attribute):
    if attribute == "color":
        return [[(getattr(agent, attribute, (0, 0, 0)) if agent.alive else (0, 0, 0)) for agent in row] for row in agent_matrix]
    else:
        return [[getattr(agent, attribute, 0) for agent in row] for row in agent_matrix]


def save_matrix_image(matrix, file_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Game World")
    cbar = plt.colorbar(im, ax=ax)
    plt.savefig(str(file_name))



def render_and_save_videos(frames, captures, videos_dir, images_dir, frame_rate):
    video_specs = [
        ("strength",               "viridis", 0,   100, "strength_map.mp4"),
        ("hardiness",              "viridis", 0,   100, "hardiness_map.mp4"),
        ("age",                    "viridis", 0,    50, "age_map.mp4"),
        ("lifespan",               "inferno", 0,   100, "lifespan_map.mp4"),
        ("metabolism",             "inferno", 0,   100, "metabolism_map.mp4"),
        ("reproduction_threshold", "magma",   0,    50, "reproduction_threshold_map.mp4"),
        ("genetic_distance",       "magma",   0,    50, "genetic_drift_map.mp4"),
        ("color",                  None,      None, None, "species_map.mp4"),
    ]

    def to_rgb(raw, cmap_name, vmin, vmax):
        arr = np.array(raw)
        if cmap_name is None:
            return (arr * 255).astype(np.uint8)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return (plt.get_cmap(cmap_name)(norm(arr))[:, :, :3] * 255).astype(np.uint8)

    def save_one(spec):
        attr, cmap_name, vmin, vmax, filename = spec
        rgb_frames = [to_rgb(f, cmap_name, vmin, vmax) for f in frames[attr]]
        imageio.mimsave(os.path.join(videos_dir, filename), rgb_frames, fps=frame_rate, macro_block_size=1)
        for step, raw in captures[attr]:
            imageio.imwrite(
                os.path.join(images_dir, f"{attr}_step_{step}.png"),
                to_rgb(raw, cmap_name, vmin, vmax)
            )

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_one, spec) for spec in video_specs]
        for f in futures:
            f.result()
