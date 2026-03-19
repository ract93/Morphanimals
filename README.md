## Setup

### Requirements
- Python 3.12+
- A C++ compiler (MSVC via Visual Studio 2022 on Windows)

### Install Python dependencies
```bash
pip install noise numpy matplotlib imageio pillow pandas nbformat nbconvert pybind11 scikit-build-core
```

### Build the C++ simulation core
The hot-path simulation loop is implemented in C++ and compiled as a Python extension (`evo_core`). Build it once before running:
```bash
cd "Evo Island"
pip install -e .
```
This produces `evo_core.cp312-win_amd64.pyd` (Windows) alongside `evo_island.py`. Rerun after any changes to files in `cpp/`.

### Run
```bash
python "Evo Island/evo_island.py"
```
Results are written to `Evo Island/Experimental_Results/<timestamp>/`.

---

## Project Structure
```
Evo Island/
  evo_island.py        Entry point — loads config, starts experiment
  config.json          All simulation parameters (see below)
  CMakeLists.txt       C++ build definition
  pyproject.toml       Python build config (scikit-build-core)
  cpp/                 C++ simulation core (Agent, step loop, speciation)
  py/                  Python modules (environment, metrics, visualization, notebooks, experiment)
```

---

## Genome

Each agent carries a 6-gene float genome, all clipped to [0, 100]. Traits are decoded directly from genome indices:

| Index | Trait | Effect |
|-------|-------|--------|
| 0 | `lifespan` | Controls the sigmoid death-probability curve. Higher = lives longer. |
| 1 | `hardiness` | Must exceed terrain difficulty to survive/colonise a cell. |
| 2 | `strength` | Wins contested cells when violence is enabled. |
| 3 | `metabolism` | Energy burned per step AND extraction rate from food cells. Higher = faster forager but more expensive to run. |
| 4 | `reproduction_threshold` | Minimum energy required to reproduce. Higher = less frequent but better-buffered offspring. |
| 5 | `speed` | Number of random-walk steps taken per turn. Ancestor starts at 0 (sessile); evolution can discover mobility. |

The common ancestor genome is `[20, 10, 5, 5, 3, 0]`.

---

## Evolutionary Trade-offs

Without trade-offs every gene would simply drift to its maximum, making the fitness landscape trivial. The following costs create genuine conflicts between traits:

**Trait maintenance** — Each step an agent pays:
```
burn = metabolism
     + lifespan  × longevity_cost
     + hardiness × armor_cost
     + strength  × strength_cost
     + speed     × movement_cost
```
Building and running costly tissue (muscle, immune system, cellular repair) consumes energy. An agent cannot freely maximise all traits without starving.

**Metabolism as extraction rate** — An agent can extract at most `1 + metabolism × metabolism_extraction_factor` food per cell visit. Unextracted food remains for others. High metabolism colonisers gorge on fresh cells and reproduce fast; low metabolism agents survive longer on depleted cells (r/K selection).

**Strength reproductive cost** — The effective reproduction threshold is `reproduction_threshold + strength × strength_repro_factor`. Muscle maintenance delays breeding, creating a fighter vs. breeder trade-off.

---

## Configuration Reference

### Experiment
| Key | Description |
|-----|-------------|
| `experimental_trials` | Number of independent simulation runs. |
| `simulation_steps` | Steps per trial. |

### Map
| Key | Description |
|-----|-------------|
| `map_size` | Side length of the square grid (cells). |
| `map_type` | `"perlin"` (island), `"petri_dish"` (difficulty gradient), or `"random"`. |
| `use_random_perlin_params` | Randomise Perlin noise parameters each run for unique terrain. |
| `use_rivers` | Carve low-difficulty river corridors through Perlin terrain. |

### Feature Flags
| Key | Description |
|-----|-------------|
| `enable_aging` | Agents accumulate age; death probability rises sigmoidally past half their lifespan. |
| `enable_food` | Enables food generation, consumption, starvation, and reproduction threshold. |
| `enable_violence` | Agents contest occupied cells on birth; winner determined by strength. |
| `enable_reproduction_threshold` | Agents must accumulate energy ≥ threshold before reproducing. |
| `enable_movement` | Agents actively translate themselves each turn (random walk of `speed` steps). |

### Trade-off Costs
These parameters control how expensive each trait is to maintain per step. Set to 0 to disable a trade-off.

| Key | Description |
|-----|-------------|
| `longevity_cost` | Energy cost per unit of `lifespan` per step (cellular repair). |
| `armor_cost` | Energy cost per unit of `hardiness` per step (structural tissue). |
| `strength_cost` | Energy cost per unit of `strength` per step (muscle maintenance). |
| `metabolism_extraction_factor` | Scales how much food a single visit extracts: `max = 1 + metabolism × factor`. |
| `strength_repro_factor` | Added to reproduction threshold proportional to strength. |
| `movement_cost` | Energy cost per unit of `speed` per step (locomotion overhead). |

### Food
| Key | Description |
|-----|-------------|
| `food_generation_rate` | Food units regenerated per cell per step. |
| `initial_food` | Food pre-loaded in every cell at simulation start. |
| `max_food_capacity` | Maximum food a cell can hold. |

### Genetics
| Key | Description |
|-----|-------------|
| `mutation_rate` | Per-gene probability of mutation during reproduction. |
| `speciation_threshold` | Euclidean genome distance above which a new species label is assigned. |

### Visualization
| Key | Description |
|-----|-------------|
| `frame_save_interval` | Steps between video frames. |
| `frame_rate` | Frames per second in output videos. |
