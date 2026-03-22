#include "simulation_core.h"
#include <algorithm>
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
    enable_space_competition      = cfg["enable_space_competition"].cast<bool>();
    // New flags default to off so old configs remain unchanged
    enable_movement  = cfg.contains("enable_movement")  ? cfg["enable_movement"].cast<bool>()  : false;
    enable_predation = cfg.contains("enable_predation") ? cfg["enable_predation"].cast<bool>() : false;

    // Trade-off costs default to 0 / large-cap so old configs run unchanged
    longevity_cost               = cfg.contains("longevity_cost")               ? cfg["longevity_cost"].cast<float>()               : 0.0f;
    armor_cost                   = cfg.contains("armor_cost")                   ? cfg["armor_cost"].cast<float>()                   : 0.0f;
    strength_cost                = cfg.contains("strength_cost")                ? cfg["strength_cost"].cast<float>()                : 0.0f;
    metabolism_extraction_factor = cfg.contains("metabolism_extraction_factor") ? cfg["metabolism_extraction_factor"].cast<float>() : 1000.0f;
    strength_repro_factor        = cfg.contains("strength_repro_factor")        ? cfg["strength_repro_factor"].cast<float>()        : 0.0f;
    movement_cost                = cfg.contains("movement_cost")                ? cfg["movement_cost"].cast<float>()                : 0.0f;
    predation_efficiency         = cfg.contains("predation_efficiency")         ? cfg["predation_efficiency"].cast<float>()         : 0.6f;
    predation_resistance         = cfg.contains("predation_resistance")         ? cfg["predation_resistance"].cast<float>()         : 0.02f;
    predation_threshold          = cfg.contains("predation_threshold")          ? cfg["predation_threshold"].cast<float>()          : 0.3f;
    trophism_cost                = cfg.contains("trophism_cost")                ? cfg["trophism_cost"].cast<float>()                : 0.0f;

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
// Hardiness check: if the newborn cannot survive the terrain it dies immediately.
// Occupied cell resolution — determined by trophism relationship:
//   trophism_diff > predation_threshold  →  predation: newborn eats resident and
//                                            takes the cell (herbivore colonising a
//                                            plant patch, carnivore displacing prey).
//   otherwise                            →  competition: probabilistic strength fight.
// This lets trophic layers emerge without any hardcoded boundaries.
void Simulation::try_birth(Agent newborn, int ni, int nj, StepResult& res) {
    if (!in_range(ni, nj)) return;

    if (newborn.hardiness <= static_cast<float>(LEVEL_DIFF[world[key(ni,nj)]])) {
        ++res.deaths_exposure;
        return;
    }

    if (at(ni, nj).alive) {
        float trophism_diff = newborn.trophism - at(ni, nj).trophism;

        if (enable_predation && trophism_diff > predation_threshold) {
            // Predation: newborn eats resident's stored energy and takes the cell.
            float steal_fraction = newborn.trophism * predation_efficiency
                                 / (1.0f + (at(ni, nj).strength + at(ni, nj).hardiness) * predation_resistance);
            newborn.energy_reserves += at(ni, nj).energy_reserves * steal_fraction;
            at(ni, nj) = newborn;
            ++res.deaths_predation;
            // live_set unchanged — cell stays occupied by newborn
        } else if (enable_space_competition) {
            ++res.deaths_competition;
            // Probabilistic strength fight — prevents strength ratcheting to max.
            float total = newborn.strength + at(ni, nj).strength;
            float p_newborn_wins = (total > 0.0f) ? newborn.strength / total : 0.5f;
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            if (uni(rng) < p_newborn_wins)
                at(ni, nj) = newborn;
            // loser dies — live_set unchanged because cell stays occupied
        }
        // If neither flag is set, newborn overwrites resident unconditionally —
        // preserving the original pre-violence behaviour for configs that disable
        // both flags to allow unconstrained population growth.
        if (!enable_predation && !enable_space_competition) {
            at(ni, nj) = newborn;
            // live_set unchanged — cell stays occupied
        }
    } else {
        // Empty cell — newborn takes it unconditionally.
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
//   2. Eat — additive model: autotroph and predation channels both active,
//             scaled by (1-trophism) and trophism respectively.
//   3. Maintenance costs — pay energy for all active traits
//   4. Reproduce — gated by energy threshold; birth target prefers empty cells,
//             falls back to occupied only when saturated; collision resolved by
//             trophism_diff (predation if > threshold, else strength competition).
//   5. Move — random walk; high-trophism movers can enter occupied cells by
//             predating the resident (trophism_diff > predation_threshold).
//
// last_step_acted guards against an agent acting twice in one step.
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

    // ── 2. Eat ────────────────────────────────────────────────────────────────
    if (enable_food) {
        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }

        // Additive model: both pathways active simultaneously, scaled by trophism.
        //   autotroph_gain = (1 - trophism) * env_food
        //   predation_gain = trophism * stolen_energy
        // No fitness valley — smooth gradient from pure autotroph (t=0) to pure
        // predator (t=1). Specialists outperform generalists in their niche.

        // ── Autotrophic component ────────────────────────────────────────────
        // When predation is disabled trophism has no effect — autotroph_rate
        // stays at 1.0 so the gene can mutate freely without silently penalising
        // all agents. Mirrors how speed/strength genes are inert when their
        // feature flag is off.
        float autotroph_rate = enable_predation ? (1.0f - a.trophism) : 1.0f;
        if (autotroph_rate > 0.0f) {
            float available   = get_available_food(i, j, current_step);
            float max_extract = autotroph_rate * (1.0f + a.metabolism * metabolism_extraction_factor);
            float food        = std::min(available, max_extract);
            deplete_cell_food(i, j, current_step, food);
            a.consume_food(food);
        }

        // ── Predation component ──────────────────────────────────────────────
        // Prey strength resists the steal: steal_fraction /= (1 + strength * resistance).
        // Mobile prey (speed >= 1) flee to an adjacent empty cell after being hit,
        // making them harder to repeatedly target. Sessile prey stay put (grazed,
        // but survive as long as energy > 0 — like a plant being browsed).
        if (enable_predation && a.trophism > 0.0f) {
            int order[8] = {1, 2, 3, 4, 5, 6, 7, 8};
            std::shuffle(order, order + 8, rng);

            for (int k = 0; k < 8; ++k) {
                int ni = i + OFFSETS[order[k]][0];
                int nj = j + OFFSETS[order[k]][1];
                if (!in_range(ni, nj)) continue;
                Agent& prey = at(ni, nj);
                if (!prey.alive) continue;

                float steal_fraction = a.trophism * predation_efficiency
                                     / (1.0f + (prey.strength + prey.hardiness) * predation_resistance);
                float stolen = prey.energy_reserves * steal_fraction;
                a.consume_food(stolen);
                prey.metabolize(stolen);

                if (prey.energy_reserves <= 0.0f) {
                    kill_at(ni, nj);
                    ++res.deaths_predation;
                } else if (enable_movement && prey.speed >= 1.0f) {
                    // Mobile prey flees to a random adjacent empty cell.
                    int flee_order[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                    std::shuffle(flee_order, flee_order + 8, rng);
                    for (int f = 0; f < 8; ++f) {
                        int fi = ni + OFFSETS[flee_order[f]][0];
                        int fj = nj + OFFSETS[flee_order[f]][1];
                        if (!in_range(fi, fj)) continue;
                        if (at(fi, fj).alive)  continue;
                        at(fi, fj) = at(ni, nj);
                        at(fi, fj).last_step_acted = current_step;
                        at(ni, nj).kill();
                        live_set.erase(key(ni, nj));
                        live_set.insert(key(fi, fj));
                        live.push_back({fi, fj});
                        break;
                    }
                }
                // Sessile prey (speed == 0) stay put — repeatedly grazeable
                // but survive as long as energy > 0.
                break;  // eat at most one neighbour per step
            }
        }

        // Trait maintenance: the cost of running each body system per step.
        float maintenance = a.metabolism
                          + a.lifespan  * longevity_cost
                          + a.hardiness * armor_cost
                          + a.strength  * strength_cost
                          + (enable_movement  ? a.speed    * movement_cost  : 0.0f)
                          + (enable_predation ? a.trophism * trophism_cost  : 0.0f);
        a.metabolize(maintenance);

        if (a.energy_reserves <= 0.0f) {
            kill_at(i, j);
            ++res.deaths_starvation;
            return;
        }
    }

    // ── 3. Reproduction ───────────────────────────────────────────────────────
    if (enable_repro_thresh && enable_food) {
        float effective_threshold = a.reproduction_threshold
                                  + a.strength * strength_repro_factor;
        if (a.energy_reserves < effective_threshold) return;
        a.metabolize(effective_threshold);
    }

    // ── 4. Movement ───────────────────────────────────────────────────────────
    if (enable_movement && a.speed >= 1.0f) {
        std::uniform_int_distribution<int> move_dice(1, 8);
        int ci = i, cj = j;
        int steps = static_cast<int>(std::min(a.speed, 20.0f));
        for (int s = 0; s < steps; ++s) {
            int roll = move_dice(rng);
            int ni = ci + OFFSETS[roll][0];
            int nj = cj + OFFSETS[roll][1];
            if (!in_range(ni, nj)) continue;
            if (at(ci, cj).hardiness <= static_cast<float>(LEVEL_DIFF[world[key(ni, nj)]])) continue;

            if (at(ni, nj).alive) {
                // Occupied: only enter if mover is sufficiently more predatory
                // than resident — herbivore grazing through a plant patch, or
                // carnivore pushing through prey territory.
                float trophism_diff = at(ci, cj).trophism - at(ni, nj).trophism;
                if (!enable_predation || trophism_diff <= predation_threshold) continue;

                // Predate resident: steal energy, kill it, then move in.
                float steal_fraction = at(ci, cj).trophism * predation_efficiency
                                     / (1.0f + (at(ni, nj).strength + at(ni, nj).hardiness) * predation_resistance);
                at(ci, cj).consume_food(at(ni, nj).energy_reserves * steal_fraction);
                kill_at(ni, nj);
                ++res.deaths_predation;
            }

            at(ni, nj) = at(ci, cj);
            at(ni, nj).last_step_acted = current_step;
            at(ci, cj).kill();
            live_set.erase(key(ci, cj));
            live_set.insert(key(ni, nj));
            live.push_back({ni, nj});
            ci = ni;
            cj = nj;
        }
        i = ci;
        j = cj;
    }

    // Place offspring in a Moore neighbour, preferring empty cells.
    // Violence (competition) only triggers as a last resort when all neighbours
    // are occupied — naturally rarer for mobile agents that have spread out,
    // but still common for sessile agents packed into dense patches (canopy crowding).
    Agent newborn = Agent::reproduce_asexually(at(i, j), mutation_rate, rng);
    newborn.energy_reserves = std::max(1.0f, at(i, j).reproduction_threshold);

    int birth_order[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::shuffle(birth_order, birth_order + 9, rng);

    // First pass: find an empty in-range cell.
    int birth_ni = -1, birth_nj = -1;
    for (int b = 0; b < 9; ++b) {
        int ni = i + OFFSETS[birth_order[b]][0];
        int nj = j + OFFSETS[birth_order[b]][1];
        if (!in_range(ni, nj)) continue;
        if (!at(ni, nj).alive) { birth_ni = ni; birth_nj = nj; break; }
    }
    // Fallback: all neighbours occupied — pick any in-range cell (violence may follow).
    if (birth_ni == -1) {
        for (int b = 0; b < 9; ++b) {
            int ni = i + OFFSETS[birth_order[b]][0];
            int nj = j + OFFSETS[birth_order[b]][1];
            if (in_range(ni, nj)) { birth_ni = ni; birth_nj = nj; break; }
        }
    }
    if (birth_ni != -1) try_birth(newborn, birth_ni, birth_nj, res);
}

// ── Speciation ────────────────────────────────────────────────────────────────

// Greedy nearest-representative clustering in 7D genome space.
void Simulation::classify_species() {
    std::vector<std::array<float,7>> rep_genomes;
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
                for (int k = 0; k < 7; ++k) {
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
            std::array<float,7> g;
            for (int k = 0; k < 7; ++k) g[k] = a.genome[k];
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
        res.total_speed        += a.speed;
        res.total_trophism     += a.trophism;
        ++res.population_count;
        species_set.insert(a.species);
    }
    res.species_counts = static_cast<int>(species_set.size());
}

// ── Main step ─────────────────────────────────────────────────────────────────

StepResult Simulation::step(int current_step) {
    StepResult res;

    // Snapshot live positions at step start so that newborns placed during this
    // step do not act until the next step.
    std::vector<std::pair<int,int>> snapshot = live;

    for (auto& [i, j] : snapshot)
        step_agent(i, j, current_step, res);

    classify_species();

    // Rebuild live / live_set: drop dead entries and remove duplicates.
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
                else if (attr == "speed")                  val = a.speed;
                else if (attr == "trophism")               val = a.trophism;
                else if (attr == "genetic_distance")       val = a.genetic_distance;
            }
            buf(i, j) = val;
        }
    return out;
}
