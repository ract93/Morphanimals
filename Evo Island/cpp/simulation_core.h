#pragma once
#include "agent.h"
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Returned to Python each step — all per-step deltas and per-step totals.
// Death counts are DELTAS (this step only); Python accumulates them.
struct StepResult {
    bool   extinct             = false;
    int    population_count    = 0;
    int    species_counts      = 0;
    int    deaths_aging        = 0;
    int    deaths_competition  = 0;
    int    deaths_starvation   = 0;
    int    deaths_exposure     = 0;
    double total_age           = 0.0;
    double total_lifespan      = 0.0;
    double total_strength      = 0.0;
    double total_hardiness     = 0.0;
    double total_metabolism    = 0.0;
    double total_reproduction_threshold = 0.0;
};

class Simulation {
public:
    // world_matrix  : list[list[int]]   terrain levels 1-5  (N x N)
    // food_matrix   : list[list[tuple]] (amount, last_accessed) (N x N)
    // start_i/j     : initial agent placement (from find_easiest_starting_location)
    Simulation(py::dict cfg,
               py::list world_matrix,
               py::list food_matrix,
               int start_i, int start_j);

    StepResult          step(int current_step);
    py::array_t<float>  get_attribute_matrix(const std::string& attr) const;
    bool                is_extinct() const { return live.empty(); }

private:
    int N;
    std::vector<Agent> agents;       // flat row-major N*N
    std::vector<int>   world;        // flat N*N  terrain levels
    std::vector<float> food_amount;  // flat N*N
    std::vector<int>   food_last;    // flat N*N  (-1 = never accessed)

    std::vector<std::pair<int,int>> live;     // ordered live positions
    std::unordered_set<int>         live_set; // i*N+j  for O(1) ops

    std::mt19937 rng;

    // Config (extracted once at construction)
    float mutation_rate;
    float speciation_threshold;
    float food_gen_rate;
    float max_food_cap;
    bool  enable_aging;
    bool  enable_food;
    bool  enable_repro_thresh;
    bool  enable_violence;

    // level_difficulty table: index = terrain level 1-5
    static constexpr int LEVEL_DIFF[6] = {0, 1, 10, 20, 40, 50};

    // Helpers
    inline Agent&       at(int i, int j)       { return agents[i*N+j]; }
    inline const Agent& at(int i, int j) const { return agents[i*N+j]; }
    inline bool         in_range(int x, int y) const {
        return x >= 0 && x < N && y >= 0 && y < N;
    }
    inline int          key(int i, int j) const { return i*N+j; }

    void  step_agent(int i, int j, int current_step, StepResult& res);
    void  try_birth(Agent newborn, int ni, int nj, StepResult& res);
    float consume_cell_food(int i, int j, int step);
    void  kill_at(int i, int j);
    void  classify_species();
    void  collect_metrics(StepResult& res) const;
};
