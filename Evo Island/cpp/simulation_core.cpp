#include "simulation_core.h"
#include <array>
#include <limits>
#include <pybind11/stl.h>

// 9 movement offsets matching Python's movement_offsets dict keys 1-9
// OFFSETS[0]=(0,0), [1]=(1,0), ..., [8]=(-1,-1)  — dice rolls 0-8 (uniform)
static constexpr int OFFSETS[9][2] = {
    { 0, 0}, { 1, 0}, {-1, 0},
    { 0, 1}, { 0,-1}, { 1, 1},
    { 1,-1}, {-1, 1}, {-1,-1}
};

// ── Construction ─────────────────────────────────────────────────────────────

Simulation::Simulation(py::dict cfg,
                       py::list world_matrix,
                       py::list food_matrix,
                       int start_i, int start_j)
{
    N = cfg["map_size"].cast<int>();

    mutation_rate        = cfg["mutation_rate"].cast<float>();
    speciation_threshold = cfg["speciation_threshold"].cast<float>();
    food_gen_rate        = cfg["food_generation_rate"].cast<float>();
    max_food_cap         = cfg["max_food_capacity"].cast<float>();
    enable_aging         = cfg["enable_aging"].cast<bool>();
    enable_food          = cfg["enable_food"].cast<bool>();
    enable_repro_thresh  = cfg["enable_reproduction_threshold"].cast<bool>();
    enable_violence      = cfg["enable_violence"].cast<bool>();

    agents.resize(N * N);
    world .resize(N * N);
    food_amount.resize(N * N, 0.0f);
    food_last  .resize(N * N, -1);

    // Copy terrain from Python list[list[int]]
    for (int i = 0; i < N; ++i) {
        py::list row = world_matrix[i].cast<py::list>();
        for (int j = 0; j < N; ++j)
            world[i*N+j] = row[j].cast<int>();
    }

    // Copy food matrix from Python list[list[tuple(float, int)]]
    for (int i = 0; i < N; ++i) {
        py::list row = food_matrix[i].cast<py::list>();
        for (int j = 0; j < N; ++j) {
            py::tuple cell  = row[j].cast<py::tuple>();
            food_amount[i*N+j] = cell[0].cast<float>();
            food_last  [i*N+j] = cell[1].cast<int>();
        }
    }

    std::random_device rd;
    rng = std::mt19937(rd());

    // Place the single ancestor
    at(start_i, start_j) = Agent::create_live_default();
    live.push_back({start_i, start_j});
    live_set.insert(key(start_i, start_j));
}

// ── Internal helpers ──────────────────────────────────────────────────────────

void Simulation::kill_at(int i, int j) {
    at(i, j).kill();
    live_set.erase(key(i, j));
}

// Mirrors Environment.calculate_food_available + update_food_matrix combined.
// The agent always takes all available food; the cell is always depleted to 0.
float Simulation::consume_cell_food(int i, int j, int step) {
    int   k         = key(i, j);
    float prev      = food_amount[k];
    int   last      = food_last[k];
    float generated = (last == -1)
                        ? step * food_gen_rate
                        : (step - last) * food_gen_rate;
    float available = std::min(prev + generated, max_food_cap);
    food_amount[k]  = 0.0f;
    food_last[k]    = step;
    return available;
}

// Mirrors the newborn-placement block inside simulate_agent_time_step.
// Faithfully replicates the Python else-branch behaviour: when violence is OFF
// the newborn silently overwrites any existing occupant (matching Python).
void Simulation::try_birth(Agent newborn, int ni, int nj, StepResult& res) {
    if (!in_range(ni, nj)) return;

    if (newborn.hardiness <= static_cast<float>(LEVEL_DIFF[world[key(ni,nj)]])) {
        ++res.deaths_exposure;
        return;
    }

    if (at(ni, nj).alive && enable_violence) {
        ++res.deaths_competition;
        if (newborn.strength > at(ni, nj).strength)
            at(ni, nj) = newborn;
        // loser (newborn or incumbent) simply goes away — no live_set change needed
    } else {
        // Cell empty OR violence disabled — newborn takes the cell
        at(ni, nj) = newborn;
        int k = key(ni, nj);
        if (live_set.find(k) == live_set.end()) {
            live.push_back({ni, nj});
            live_set.insert(k);
        }
    }
}

// ── Per-agent step ────────────────────────────────────────────────────────────

void Simulation::step_agent(int i, int j, int current_step, StepResult& res) {
    Agent& a = at(i, j);
    if (!a.alive) return;

    // Aging
    a.age_step(enable_aging, rng);
    if (!a.alive) {
        ++res.deaths_aging;
        live_set.erase(key(i, j));
        return;
    }

    // Food / starvation
    if (enable_food) {
        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }
        float food = consume_cell_food(i, j, current_step);
        a.consume_food(food);
        a.metabolize(a.metabolism);
        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }
    }

    // Reproduction threshold gate
    if (enable_repro_thresh && enable_food) {
        if (a.energy_reserves < a.reproduction_threshold) return;
        a.metabolize(a.reproduction_threshold);
    }

    // Reproduce + move
    Agent newborn = Agent::reproduce_asexually(a, mutation_rate, rng);
    std::uniform_int_distribution<int> dice(0, 8);
    int roll = dice(rng);
    int ni   = i + OFFSETS[roll][0];
    int nj   = j + OFFSETS[roll][1];
    try_birth(newborn, ni, nj, res);
}

// ── Speciation (mirrors speciation.py classify_species) ───────────────────────

void Simulation::classify_species() {
    std::vector<std::array<float,5>> rep_genomes;
    std::vector<int>                 rep_labels;
    int species_counter = 1;

    for (auto& [i, j] : live) {
        Agent& a = at(i, j);
        if (!a.alive) continue;

        bool assigned = false;
        if (!rep_genomes.empty()) {
            float best_dist = std::numeric_limits<float>::max();
            int   best_idx  = -1;
            for (int s = 0; s < static_cast<int>(rep_genomes.size()); ++s) {
                float d = 0.0f;
                for (int k = 0; k < 5; ++k) {
                    float diff = a.genome[k] - rep_genomes[s][k];
                    d += diff * diff;
                }
                d = std::sqrt(d);
                if (d < best_dist) { best_dist = d; best_idx = s; }
            }
            if (best_dist < speciation_threshold) {
                a.species = rep_labels[best_idx];
                assigned  = true;
            }
        }

        if (!assigned) {
            a.species = species_counter;
            std::array<float,5> g;
            for (int k = 0; k < 5; ++k) g[k] = a.genome[k];
            rep_genomes.push_back(g);
            rep_labels .push_back(species_counter);
            ++species_counter;
        }
    }
}

// ── Metrics collection ────────────────────────────────────────────────────────

void Simulation::collect_metrics(StepResult& res) const {
    std::unordered_set<int> species_set;
    for (auto& [i, j] : live) {
        const Agent& a = at(i, j);
        if (!a.alive) continue;
        res.total_age          += a.age;
        res.total_lifespan     += a.lifespan;
        res.total_strength     += a.strength;
        res.total_hardiness    += a.hardiness;
        res.total_metabolism   += a.metabolism;
        res.total_reproduction_threshold += a.reproduction_threshold;
        ++res.population_count;
        species_set.insert(a.species);
    }
    res.species_counts = static_cast<int>(species_set.size());
}

// ── Main step ─────────────────────────────────────────────────────────────────

StepResult Simulation::step(int current_step) {
    StepResult res;

    // Snapshot: iterate only agents that were alive at step start.
    // Newborns added to live during this step will NOT be in snapshot,
    // so they won't act until next step — matching Python list(live_agents).
    std::vector<std::pair<int,int>> snapshot = live;

    for (auto& [i, j] : snapshot)
        step_agent(i, j, current_step, res);

    classify_species();

    // Rebuild live / live_set: remove dead entries and deduplicate.
    // Duplicates can arise when a newborn fills a cell that was in live
    // but whose occupant died mid-step.
    {
        std::vector<std::pair<int,int>> new_live;
        std::unordered_set<int>         new_set;
        new_live.reserve(live.size());
        for (auto& p : live) {
            int k = key(p.first, p.second);
            if (live_set.count(k) &&
                at(p.first, p.second).alive &&
                !new_set.count(k))
            {
                new_live.push_back(p);
                new_set .insert(k);
            }
        }
        live     = std::move(new_live);
        live_set = std::move(new_set);
    }

    collect_metrics(res);
    res.extinct = live.empty();
    return res;
}

// ── Frame extraction ──────────────────────────────────────────────────────────

py::array_t<float> Simulation::get_attribute_matrix(const std::string& attr) const {
    if (attr == "color") {
        py::array_t<float> out({N, N, 3});
        auto buf = out.mutable_unchecked<3>();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const Agent& a = at(i, j);
                buf(i,j,0) = a.alive ? a.color[0] : 0.0f;
                buf(i,j,1) = a.alive ? a.color[1] : 0.0f;
                buf(i,j,2) = a.alive ? a.color[2] : 0.0f;
            }
        return out;
    }

    py::array_t<float> out({N, N});
    auto buf = out.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const Agent& a = at(i, j);
            float val = 0.0f;
            if (a.alive) {
                if      (attr == "age")                    val = a.age;
                else if (attr == "lifespan")               val = a.lifespan;
                else if (attr == "hardiness")              val = a.hardiness;
                else if (attr == "strength")               val = a.strength;
                else if (attr == "metabolism")             val = a.metabolism;
                else if (attr == "reproduction_threshold") val = a.reproduction_threshold;
                else if (attr == "genetic_distance")       val = a.genetic_distance;
            }
            buf(i, j) = val;
        }
    return out;
}
