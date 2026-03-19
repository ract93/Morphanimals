#include "simulation_core.h"
#include <array>
#include <limits>
#include <pybind11/stl.h>

// The 9 Moore-neighbourhood offsets including stay-in-place (index 0).
// Reproduction uses indices 0-8 (uniform dice roll).
// Movement uses indices 1-8 only (must actually move).
static constexpr int OFFSETS[9][2] = {
    { 0, 0}, { 1, 0}, {-1, 0},
    { 0, 1}, { 0,-1}, { 1, 1},
    { 1,-1}, {-1, 1}, {-1,-1}
};

// ── Construction ──────────────────────────────────────────────────────────────

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
    // enable_movement defaults to false so old configs remain plant-like
    enable_movement      = cfg.contains("enable_movement") ? cfg["enable_movement"].cast<bool>() : false;

    // Trade-off costs default to 0 / large-cap so old configs run unchanged
    longevity_cost               = cfg.contains("longevity_cost")               ? cfg["longevity_cost"].cast<float>()               : 0.0f;
    armor_cost                   = cfg.contains("armor_cost")                   ? cfg["armor_cost"].cast<float>()                   : 0.0f;
    strength_cost                = cfg.contains("strength_cost")                ? cfg["strength_cost"].cast<float>()                : 0.0f;
    metabolism_extraction_factor = cfg.contains("metabolism_extraction_factor") ? cfg["metabolism_extraction_factor"].cast<float>() : 1000.0f;
    strength_repro_factor        = cfg.contains("strength_repro_factor")        ? cfg["strength_repro_factor"].cast<float>()        : 0.0f;
    movement_cost                = cfg.contains("movement_cost")                ? cfg["movement_cost"].cast<float>()                : 0.0f;

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

    // Seed the simulation with a single ancestor at the easiest terrain patch
    at(start_i, start_j) = Agent::create_live_default();
    live.push_back({start_i, start_j});
    live_set.insert(key(start_i, start_j));
}

// ── Internal helpers ──────────────────────────────────────────────────────────

// Kill the agent at (i,j) and remove it from the live set.
void Simulation::kill_at(int i, int j) {
    at(i, j).kill();
    live_set.erase(key(i, j));
}

// Calculate how much food is available at (i,j) this step without consuming any.
// Food regenerates linearly at food_gen_rate per step since last access,
// capped at max_food_cap. last_accessed == -1 means the cell was never visited,
// so food has been accumulating since step 0.
float Simulation::get_available_food(int i, int j, int step) const {
    int   k         = key(i, j);
    float prev      = food_amount[k];
    int   last      = food_last[k];
    float generated = (last == -1)
                        ? step * food_gen_rate
                        : (step - last) * food_gen_rate;
    return std::min(prev + generated, max_food_cap);
}

// Consume `consumed` units from cell (i,j), leaving the remainder for future
// visitors. This allows low-metabolism agents to find leftover food after a
// high-metabolism agent has passed through.
void Simulation::deplete_cell_food(int i, int j, int step, float consumed) {
    int   k         = key(i, j);
    float available = get_available_food(i, j, step);
    food_amount[k]  = available - consumed;
    food_last[k]    = step;
}

// Attempt to place a newborn at (ni,nj).
// Hardiness check: if the newborn cannot survive the terrain difficulty it dies
// immediately (counted as an exposure death).
// Violence check: if the cell is occupied and violence is enabled, the stronger
// agent wins and the weaker is discarded. When violence is off a newborn
// silently overwrites any occupant — matching the original Python behaviour.
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
        // loser simply disappears — live_set unchanged because the cell stays occupied
    } else {
        // Cell is empty OR violence is disabled — newborn takes the cell
        at(ni, nj) = newborn;
        int k = key(ni, nj);
        if (live_set.find(k) == live_set.end()) {
            live.push_back({ni, nj});
            live_set.insert(k);
        }
    }
}

// ── Per-agent step ────────────────────────────────────────────────────────────

// Execute one full turn for the agent at (i,j):
//   1. Age — may die of old age
//   2. Eat — extract food (capped by metabolism), pay maintenance costs, may starve
//   3. Reproduce — gated by energy threshold; offspring placed in random neighbour
//   4. Move — random walk of up to `speed` steps (enable_movement must be true)
//
// last_step_acted guards against an agent acting twice in one step: when an
// agent moves into a cell that is still in the step snapshot it would otherwise
// be processed again at its new coordinates.
void Simulation::step_agent(int i, int j, int current_step, StepResult& res) {
    Agent& a = at(i, j);
    if (!a.alive) return;
    if (a.last_step_acted == current_step) return;
    a.last_step_acted = current_step;

    // ── 1. Aging ──────────────────────────────────────────────────────────────
    a.age_step(enable_aging, rng);
    if (!a.alive) {
        ++res.deaths_aging;
        live_set.erase(key(i, j));
        return;
    }

    // ── 2. Eat & maintenance ──────────────────────────────────────────────────
    if (enable_food) {
        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }

        // Metabolism gates extraction: faster metabolism → more food extracted per
        // visit, but also higher burn rate. Creates r/K selection pressure:
        //   high metabolism → coloniser (gorges on fresh cells, dies in depleted ones)
        //   low  metabolism → survivor  (eats little, persists on depleted cells)
        float available   = get_available_food(i, j, current_step);
        float max_extract = 1.0f + a.metabolism * metabolism_extraction_factor;
        float food        = std::min(available, max_extract);
        deplete_cell_food(i, j, current_step, food);
        a.consume_food(food);

        // Trait maintenance: the cost of running each body system per step.
        // Prevents every gene from drifting to its maximum — a large, armoured,
        // long-lived, fast fighter must burn far more calories to sustain itself.
        float maintenance = a.metabolism
                          + a.lifespan  * longevity_cost   // cellular repair
                          + a.hardiness * armor_cost        // structural tissue
                          + a.strength  * strength_cost     // muscle upkeep
                          + a.speed     * movement_cost;    // locomotion overhead
        a.metabolize(maintenance);

        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }
    }

    // ── 3. Reproduction ───────────────────────────────────────────────────────
    // Strong agents require more energy to breed (muscle maintenance delays
    // reproduction), creating a fighter vs. breeder trade-off.
    if (enable_repro_thresh && enable_food) {
        float effective_threshold = a.reproduction_threshold
                                  + a.strength * strength_repro_factor;
        if (a.energy_reserves < effective_threshold) return;
        a.metabolize(effective_threshold);
    }

    // ── 4. Movement ───────────────────────────────────────────────────────────
    // The agent translates itself up to `speed` cells via an unbiased random walk
    // before placing its offspring. This decouples locomotion from reproduction,
    // allowing mobile agents to seek out food-rich cells.
    // Movement is capped at 20 steps per turn to keep per-step cost bounded.
    if (enable_movement && a.speed >= 1.0f) {
        std::uniform_int_distribution<int> move_dice(1, 8);  // excludes stay-in-place
        int ci = i, cj = j;
        int steps = static_cast<int>(std::min(a.speed, 20.0f));
        for (int s = 0; s < steps; ++s) {
            int roll = move_dice(rng);
            int ni = ci + OFFSETS[roll][0];
            int nj = cj + OFFSETS[roll][1];
            if (!in_range(ni, nj))   continue;  // map boundary
            if (at(ni, nj).alive)    continue;  // occupied — skip this step of the walk
            // Terrain check: agent must be able to survive the destination cell
            if (at(ci, cj).hardiness <= static_cast<float>(LEVEL_DIFF[world[key(ni, nj)]])) continue;
            // Move: copy agent to new cell, clear old cell, update bookkeeping
            at(ni, nj) = at(ci, cj);
            at(ni, nj).last_step_acted = current_step;  // prevent double-act at new position
            at(ci, cj).kill();
            live_set.erase(key(ci, cj));
            live_set.insert(key(ni, nj));
            live.push_back({ni, nj});  // deduplicated during live rebuild at end of step()
            ci = ni;
            cj = nj;
        }
        // Update i/j so the offspring is placed relative to the final position
        i = ci;
        j = cj;
    }

    // Place offspring in a random Moore neighbour of the agent's current position.
    // Offspring start with energy equal to the threshold the parent paid — creating
    // a genuine r/K trade-off: low threshold = many cheap offspring that start
    // energy-poor; high threshold = fewer but hardier offspring.
    Agent newborn = Agent::reproduce_asexually(at(i, j), mutation_rate, rng);
    newborn.energy_reserves = std::max(1.0f, at(i, j).reproduction_threshold);
    std::uniform_int_distribution<int> dice(0, 8);
    int roll = dice(rng);
    int ni   = i + OFFSETS[roll][0];
    int nj   = j + OFFSETS[roll][1];
    try_birth(newborn, ni, nj, res);
}

// ── Speciation ────────────────────────────────────────────────────────────────

// Greedy nearest-representative clustering in 6D genome space.
// Each agent is assigned to the closest existing species representative if the
// Euclidean distance is below speciation_threshold; otherwise it founds a new
// species. Order-dependent but fast — mirrors the Python implementation.
void Simulation::classify_species() {
    std::vector<std::array<float,6>> rep_genomes;
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
                for (int k = 0; k < 6; ++k) {
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
            // New species: this agent becomes the representative
            a.species = species_counter;
            std::array<float,6> g;
            for (int k = 0; k < 6; ++k) g[k] = a.genome[k];
            rep_genomes.push_back(g);
            rep_labels .push_back(species_counter);
            ++species_counter;
        }
    }
}

// ── Metrics collection ────────────────────────────────────────────────────────

// Sum per-agent trait values into res. Python divides by population_count to
// get averages. Species count is the number of distinct labels this step.
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
        res.total_speed          += a.speed;
        ++res.population_count;
        species_set.insert(a.species);
    }
    res.species_counts = static_cast<int>(species_set.size());
}

// ── Main step ─────────────────────────────────────────────────────────────────

StepResult Simulation::step(int current_step) {
    StepResult res;

    // Snapshot live positions at step start so that newborns placed during this
    // step do not act until the next step. Matches the Python list(live_agents)
    // copy that served the same purpose in the original implementation.
    std::vector<std::pair<int,int>> snapshot = live;

    for (auto& [i, j] : snapshot)
        step_agent(i, j, current_step, res);

    classify_species();

    // Rebuild live / live_set: drop dead entries and remove duplicates.
    // Duplicates can arise when an agent moves into a cell whose original
    // occupant died earlier in the same step, causing two entries for the
    // same key to coexist in live temporarily.
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

// Return an N×N numpy array (or N×N×3 for "color") suitable for video rendering.
// Dead cells produce 0 in all channels. Called every frame_save_interval steps.
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
                else if (attr == "speed")                  val = a.speed;
                else if (attr == "genetic_distance")       val = a.genetic_distance;
            }
            buf(i, j) = val;
        }
    return out;
}
