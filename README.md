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

## Configuration Settings Guide

- **map_size**: The size of the simulation grid.
- **simulation_steps**: Number of steps the simulation runs.

- **terrain**:
  - **map_type**: Determines the type of terrain to generate. Map choices (Input into config as strings):
    - **"perlin"**: Generates terrain using Perlin noise for realistic features.
    - **"petri_dish"**: Generates a gradient-style terrain resembling the petri dish experiment from LTEE.
    - **"random"**: Generates a random terrain.
  - **use_random_perlin_params**: Randomize the parameters of the Perlin noise. Only active if using Perlin Noise. 
  - **use_rivers**: Include rivers in the terrain, only active if using Perlin Noise.

- **agent_features**:
  - **use_energy_budget**: Enables balancing of genetic trait expression with an energy budget based on metabolism. Used to encourage specialization.
  - **enable_violence**: Allows agents to contest occupied cells. If off, agents that are born in an already occupied cell will just die to exposure.
  - **enable_aging**: Agents age and can die of old age.
  - **enable_food**: Controls all food-related features.
    - **food_generation_rate**: Rate at which food regenerates per step. Only active if food is enabled.
    - **initial_food**: Starting amount of food in each cell. Only active if food is enabled.
    - **enable_reproduction_threshold**: Agents need to meet a food threshold to reproduce. Only active if food is enabled.
  
  
- **genetics**:
  - **mutation_rate**: Probability of mutations during reproduction.
  - **speciation_threshold**: Threshold of genetic distance between species representatives at which a new species is formed.

- **visualization**:
  - **frame_save_interval**: Steps between saving frames for a GIF.
  - **frame_rate**: Frames per second in output GIFs.
