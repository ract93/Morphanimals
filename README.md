## Configuration Settings Guide

- **map_size**: The size of the simulation grid.
- **simulation_steps**: Number of steps the simulation runs.

- **terrain**:
  - **map_type**: Determines the type of terrain to generate. Map choices:
    - **"perlin"**: Generates terrain using Perlin noise for realistic features.
    - **"petri_dish"**: Generates a gradient-style terrain resembling a petri dish.
    - **"random"**: Generates a random terrain.
  - **use_random_perlin_params**: Randomize the parameters of the Perlin noise. Only active if using Perlin Noise. 
  - **use_rivers**: Include rivers in the terrain.

- **agent_features**:
  - **use_energy_budget**: Enables balancing of genetic trait expression with an energy budget based on metabolism. Used to encourage specialization.
  - **enable_violence**: Allows agents to contest occupied cells. If off, agents that are born in an already occupied cell will just die to exposure.
  - **enable_food**: Controls all food-related features.
    - **food_generation_rate**: Rate at which food regenerates per step. Only active if food is enabled.
    - **initial_food**: Starting amount of food in each cell. Only active if food is enabled.
    - **enable_reproduction_threshold**: Agents need to meet a food threshold to reproduce. Only active if food is enabled.
  - **enable_aging**: Agents age and can die of old age.
  
- **genetics**:
  - **mutation_rate**: Probability of mutations during reproduction.

- **visualization**:
  - **frame_save_interval**: Steps between saving frames for a GIF.
  - **frame_rate**: Frames per second in output GIFs.
