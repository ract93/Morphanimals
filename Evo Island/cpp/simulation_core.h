#pragma once
#include "agent.h"
#include <atomic>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    int    deaths_exposure     = 0;
    int    deaths_predation    = 0;
    double total_age           = 0.0;
    double total_lifespan      = 0.0;
    double total_strength      = 0.0;
    double total_hardiness     = 0.0;
    double total_metabolism    = 0.0;
    double total_reproduction_threshold = 0.0;
    double total_speed         = 0.0;
    double total_trophism      = 0.0;
    double total_kin_attraction  = 0.0;
    double total_threat_response = 0.0;
};

// Placement request written into the per-cell queue during Phase 2.
struct PlacementReq {
    Agent agent;
    bool  is_birth;
};

// Output of Phase 1 (parallel) per-agent computation.
// No grid writes happen in Phase 1 — all results are collected here
// and resolved in Phase 2 (sequential).
struct AgentUpdate {
    bool  alive        = false;
    // Per-step death deltas (merged into StepResult in Phase 2)
    int   d_aging      = 0;
    int   d_starvation = 0;
    int   d_exposure   = 0;   // newborn killed by terrain
    // Survivor destination
    int   dst_i = -1, dst_j = -1;
    Agent result;             // agent state after self-processing
    // Birth
    bool  has_birth  = false;
    int   birth_i = -1, birth_j = -1;
    Agent newborn;
    // Predation this agent performed on a neighbour
    bool  has_prey    = false;
    int   prey_i  = -1, prey_j  = -1;
    float prey_damage = 0.0f;  // inf = movement-predation (always kills)
};

// One Simulation instance per trial. Owns the agent grid, food grid, and terrain.
// Driven by repeated calls to step() until extinct or the step limit is reached.
//
// Two-buffer (Jacobi) update model: all agents read the start-of-step state
// simultaneously (Phase 1, parallel), then conflicts are resolved in Phase 2
// (sequential). This eliminates Gauss-Seidel ordering artefacts and allows
// Phase 1 to run fully in parallel via OpenMP.
class Simulation {
public:
    Simulation(py::dict cfg,
               py::list world_matrix,   // list[list[int]]   terrain levels 1-5  (N×N)
               py::list food_matrix,    // list[list[tuple]] (amount, last_accessed) (N×N)
               int start_i, int start_j);

    StepResult          step(int current_step);
    py::array_t<float>  get_attribute_matrix(const std::string& attr) const;
    bool                is_extinct() const { return live.empty(); }

private:
    int N;
    std::vector<Agent> agents;       // read buffer  — start-of-step state (read-only during step)
    std::vector<Agent> agents_next;  // write buffer — next-state target; swapped with agents each step
    std::vector<int>   world;
    std::vector<float> food_amount;
    std::vector<int>   food_last;

    std::vector<std::pair<int,int>> live;     // positions of all live agents (rebuilt each step)
    std::unordered_set<int>         live_set; // i*N+j keys for O(1) membership (rebuilt each step)

    std::mt19937 rng;
    std::vector<std::mt19937> thread_rngs;   // one per OpenMP thread, seeded from rng

    // Per-cell atomic step-number for predation claiming.
    // prey_claim_step[k] == current_step means prey at k was already claimed this step.
    // Using exchange() to atomically claim; no reset needed between steps.
    std::unique_ptr<std::atomic<int>[]> prey_claim_step;

    // Per-cell placement queues (pre-allocated, cleared lazily via queued_cells).
    std::vector<std::vector<PlacementReq>> cell_queue;
    std::vector<int>                       queued_cells;

    // Pre-allocated per-step buffers — avoids heap churn each step.
    std::vector<AgentUpdate> updates;         // resized to live.size() each step

    // Flat cell→live-index map for resolve_placements, replacing the per-step
    // unordered_map. Entries are -1 when unused; reset lazily via dirty list.
    std::vector<int> cell_to_idx;
    std::vector<int> cell_to_idx_dirty;

    // Feature flags
    float mutation_rate;
    float speciation_threshold;
    float food_gen_rate;
    float max_food_cap;
    bool  enable_aging;
    bool  enable_food;
    bool  enable_repro_thresh;
    bool  enable_space_competition;
    bool  enable_movement;
    bool  enable_predation;
    bool  enable_sociality;

    // Agent hardiness must strictly exceed this value to survive a given terrain level.
    static constexpr int LEVEL_DIFF[6] = {0, 1, 10, 20, 40, 50};

    // Trade-off costs
    float longevity_cost;
    float armor_cost;
    float strength_cost;
    float metabolism_extraction_factor;
    float strength_repro_factor;
    float movement_cost;
    float predation_efficiency;
    float predation_resistance;
    float predation_threshold;
    float trophism_cost;
    float sociality_cost;

    // Reusable distributions for the sequential phase (Phase 2).
    std::uniform_real_distribution<float> uni01{0.0f, 1.0f};
    std::uniform_int_distribution<int>    dice8{0, 7};

    inline Agent&       at(int i, int j)       { return agents[i*N+j]; }
    inline const Agent& at(int i, int j) const { return agents[i*N+j]; }
    inline bool         in_range(int x, int y) const {
        return x >= 0 && x < N && y >= 0 && y < N;
    }
    inline int          key(int i, int j) const { return i*N+j; }

    // Phase 1: compute one agent's full update without touching shared state.
    void compute_agent_update(int live_idx, int current_step,
                              AgentUpdate& out, std::mt19937& rng_t);

    // Phase 2: resolve all AgentUpdates into agents_next; rebuild live/live_set.
    void resolve_placements(std::vector<AgentUpdate>& updates, StepResult& res);

    float get_available_food(int i, int j, int step) const;
    void  deplete_cell_food(int i, int j, int step, float consumed);
    void  classify_species();
    void  collect_metrics(StepResult& res) const;
};
