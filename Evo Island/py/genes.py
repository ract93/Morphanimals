# Single source of truth for gene trait definitions.
#
# Adding a gene here automatically updates:
#   - Video rendering  (visualize.py reads VIDEO_SPECS)
#   - Metrics tracking (metrics.py reads GENES for total_*/average_* fields)
#   - HDF5 output      (output.py reads GENES to build METRIC_COLUMNS)
#   - Simulation loop  (simulation.py reads GENES to copy StepResult totals)
#   - Notebooks        (auto-discover "Average *" columns from HDF5 — no changes needed)
#
# The only other required changes when adding a gene are in C++:
#   agent.h / agent.cpp     — add decoded trait, expand genome array
#   simulation_core.h/.cpp  — add total_* to StepResult, collect_metrics, get_attribute_matrix
#   evo_core.cpp            — expose new total_* field to Python

# (attribute_name, colormap, vmin, vmax)
# Must stay in sync with: Agent decoded traits, StepResult total_* fields,
# and get_attribute_matrix() attribute dispatch.
GENES = [
    ("lifespan",               "inferno",   0,   100),
    ("hardiness",              "viridis",   0,   100),
    ("strength",               "viridis",   0,   100),
    ("metabolism",             "inferno",   0,   100),
    ("reproduction_threshold", "magma",     0,    50),
    ("speed",                  "plasma",    0,    20),
    ("trophism",               "hot",        0,     1),
    ("kin_attraction",         "RdYlGn",   -1,     1),
    ("threat_response",        "RdYlBu",   -1,     1),
]

# Rendered in video/images but not tracked as per-step averages in metrics.
_DISPLAY_ONLY = [
    ("age",              "viridis",  0,   50),
    ("genetic_distance", "magma",    0,   50),
    ("color",            None,       None, None),
]


def _video_filename(name):
    if name == "genetic_distance":
        return "genetic_drift_map.mp4"
    return f"{name}_map.mp4"


# Full video spec consumed by visualize.py: [(name, cmap, vmin, vmax, filename), ...]
VIDEO_SPECS = [
    (name, cmap, vmin, vmax, _video_filename(name))
    for name, cmap, vmin, vmax in GENES + _DISPLAY_ONLY
]


def column_name(attr):
    """CSV / notebook column header for a gene's per-step average."""
    return f"Average {attr.replace('_', ' ').title()}"
