#pragma once
#include "agent.h"
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Returned to Python after each step().
// Death counts are DELTAS (this step only) — Python accumulates them.
// Trait totals are absolute sums; divide by population_count for averages.
struct StepResult {
    bool   extinct             = false;
    int    population_count    = 0;
    int    species_counts      = 0;
    int    deaths_aging        = 0;
    int    deaths_competition  = 0;
    int    deaths_starvation   = 0;
    int    deaths_exposure     = 0;  // newborns killed by terrain hardiness check
    double total_age           = 0.0;
    double total_lifespan      = 0.0;
    double total_strength      = 0.0;
    double total_hardiness     = 0.0;
    double total_metabolism    = 0.0;
    double total_reproduction_threshold = 0.0;
    double total_speed         = 0.0;
};

// One Simulation instance per trial. Owns the agent grid, food grid, and terrain.
// Driven by repeated calls to step() until extinct or the step limit is reached.
class Simulation {
public:
    Simulation(py::dict cfg,
               py::list world_matrix,   // list[list[int]]   terrain levels 1-5  (N×N)
               py::list food_matrix,    // list[list[tuple]] (amount, last_accessed) (N×N)
               int start_i, int start_j);

    StepResult          step(int current_step);
    // Returns N×N (or N×N×3 for "color") numpy array of the named attribute. Dead cells = 0.
    py::array_t<float>  get_attribute_matrix(const std::string& attr) const;
    bool                is_extinct() const { return live.empty(); }

private:
    int N;
    std::vector<Agent> agents;       // flat row-major N*N
    std::vector<int>   world;        // flat N*N terrain difficulty levels (1-5)
    std::vector<float> food_amount;  // flat N*N
    std::vector<int>   food_last;    // flat N*N, step of last access (-1 = never)

    std::vector<std::pair<int,int>> live;     // positions of all live agents
    std::unordered_set<int>         live_set; // i*N+j keys for O(1) membership tests

    std::mt19937 rng;

    // Feature flags
    float mutation_rate;
    float speciation_threshold;
    float food_gen_rate;
    float max_food_cap;
    bool  enable_aging;
    bool  enable_food;
    bool  enable_repro_thresh;
    bool  enable_violence;
    bool  enable_movement;

    // Agent hardiness must strictly exceed this value to survive a given terrain level.
    static constexpr int LEVEL_DIFF[6] = {0, 1, 10, 20, 40, 50};

    // Trade-off costs — prevent unconstrained trait maximisation.
    float longevity_cost;                // energy/step per unit of lifespan
    float armor_cost;                    // energy/step per unit of hardiness
    float strength_cost;                 // energy/step per unit of strength
    float metabolism_extraction_factor;  // max food per visit = 1 + metabolism * factor
    float strength_repro_factor;         // extra energy threshold per unit of strength
    float movement_cost;                 // energy/step per unit of speed

    inline Agent&       at(int i, int j)       { return agents[i*N+j]; }
    inline const Agent& at(int i, int j) const { return agents[i*N+j]; }
    inline bool         in_range(int x, int y) const {
        return x >= 0 && x < N && y >= 0 && y < N;
    }
    inline int          key(int i, int j) const { return i*N+j; }

    void  step_agent(int i, int j, int current_step, StepResult& res);
    void  try_birth(Agent newborn, int ni, int nj, StepResult& res);
    float get_available_food(int i, int j, int step) const;
    void  deplete_cell_food(int i, int j, int step, float consumed);
    void  kill_at(int i, int j);
    void  classify_species();
    void  collect_metrics(StepResult& res) const;
};
