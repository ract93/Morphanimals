#include "simulation_core.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
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
    enable_space_competition = cfg["enable_space_competition"].cast<bool>();
    enable_movement   = cfg.contains("enable_movement")   ? cfg["enable_movement"].cast<bool>()   : false;
    enable_predation  = cfg.contains("enable_predation")  ? cfg["enable_predation"].cast<bool>()  : false;
    enable_sociality  = cfg.contains("enable_sociality")  ? cfg["enable_sociality"].cast<bool>()  : false;

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
    sociality_cost               = cfg.contains("sociality_cost")               ? cfg["sociality_cost"].cast<float>()               : 0.0f;

    agents     .resize(N * N);
    agents_next.resize(N * N);
    world      .resize(N * N);
    food_amount.resize(N * N, 0.0f);
    food_last  .resize(N * N, -1);
    cell_queue .resize(N * N);

    // Atomic predation claim tracker — -1 means "unclaimed"
    prey_claim_step = std::make_unique<std::atomic<int>[]>(N * N);
    for (int i = 0; i < N * N; ++i)
        prey_claim_step[i].store(-1, std::memory_order_relaxed);

    // Copy terrain
    for (int i = 0; i < N; ++i) {
        py::list row = world_matrix[i].cast<py::list>();
        for (int j = 0; j < N; ++j)
            world[i*N+j] = row[j].cast<int>();
    }

    // Copy food matrix
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

    // One RNG per OpenMP thread, seeded from the main RNG
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    thread_rngs.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t)
        thread_rngs.emplace_back(rng());

    queued_cells.reserve(4096);

    // Seed the simulation with a single ancestor
    agents[key(start_i, start_j)] = Agent::create_live_default();
    live.push_back({start_i, start_j});
    live_set.insert(key(start_i, start_j));
}

// ── Food helpers ──────────────────────────────────────────────────────────────

float Simulation::get_available_food(int i, int j, int step) const {
    int   k         = key(i, j);
    float prev      = food_amount[k];
    int   last      = food_last[k];
    float generated = (last == -1)
                        ? step * food_gen_rate
                        : (step - last) * food_gen_rate;
    return std::min(prev + generated, max_food_cap);
}

void Simulation::deplete_cell_food(int i, int j, int step, float consumed) {
    int   k         = key(i, j);
    float available = get_available_food(i, j, step);
    food_amount[k]  = available - consumed;
    food_last[k]    = step;
}

// ── Phase 1: Per-agent parallel computation ───────────────────────────────────
//
// Reads only from agents[] (start-of-step read buffer) and the food grid.
// The food grid is written only to own cell (one agent per cell → no conflict).
// Predation claims use atomic exchange to prevent double-claiming.
// All results are written to AgentUpdate& out — no grid mutations.

void Simulation::compute_agent_update(int live_idx, int current_step,
                                      AgentUpdate& out, std::mt19937& rng_t)
{
    auto [orig_i, orig_j] = live[live_idx];
    out = AgentUpdate{};

    const Agent& slot = agents[key(orig_i, orig_j)];
    if (!slot.alive) return;

    Agent a = slot;   // working copy — all mutations stay local until out is written

    // ── 1. Aging ──────────────────────────────────────────────────────────────
    a.age_step(enable_aging, rng_t);
    if (!a.alive) { out.d_aging = 1; return; }

    // ── 2. Eat ────────────────────────────────────────────────────────────────
    if (enable_food) {
        if (a.energy_reserves <= 0.0f) { out.d_starvation = 1; return; }

        // Autotrophic channel — own cell only, no concurrent conflict.
        float autotroph_rate = enable_predation ? (1.0f - a.trophism) : 1.0f;
        if (autotroph_rate > 0.0f) {
            float available   = get_available_food(orig_i, orig_j, current_step);
            float max_extract = autotroph_rate * (1.0f + a.metabolism * metabolism_extraction_factor);
            float food        = std::min(available, max_extract);
            deplete_cell_food(orig_i, orig_j, current_step, food);
            a.consume_food(food);
        }

        // Predation channel — scan neighbours, atomically claim one prey.
        if (enable_predation && a.trophism > 0.0f) {
            std::uniform_int_distribution<int> d8(0, 7);
            int ps = d8(rng_t);
            for (int k = 0; k < 8; ++k) {
                int dir = (ps + k) % 8 + 1;
                int ni  = orig_i + OFFSETS[dir][0];
                int nj  = orig_j + OFFSETS[dir][1];
                if (!in_range(ni, nj)) continue;
                const Agent& prey = agents[key(ni, nj)];
                if (!prey.alive) continue;

                // Atomically claim: exchange returns the old value.
                // If old != current_step, we are the first claimer this step.
                int old = prey_claim_step[key(ni, nj)].exchange(
                              current_step, std::memory_order_acq_rel);
                if (old == current_step) continue;

                float steal_frac = a.trophism * predation_efficiency
                                 / (1.0f + (prey.strength + prey.hardiness) * predation_resistance);
                float stolen = prey.energy_reserves * steal_frac;
                a.consume_food(stolen);

                out.has_prey    = true;
                out.prey_i      = ni;  out.prey_j  = nj;
                out.prey_damage = stolen;  // finite → may or may not kill
                break;
            }
        }

        // Trait maintenance costs
        float maintenance = a.metabolism
                          + a.lifespan  * longevity_cost
                          + a.hardiness * armor_cost
                          + a.strength  * strength_cost
                          + (enable_movement  ? a.speed    * movement_cost  : 0.0f)
                          + (enable_predation ? a.trophism * trophism_cost  : 0.0f)
                          + (enable_sociality ? (std::abs(a.kin_attraction) + std::abs(a.threat_response)) * sociality_cost : 0.0f);
        a.metabolize(maintenance);

        if (a.energy_reserves <= 0.0f) { out.d_starvation = 1; return; }
    }

    // ── 3. Reproduction threshold ─────────────────────────────────────────────
    // Gates both movement and birth — agent stays put if below threshold.
    if (enable_repro_thresh && enable_food) {
        float effective_threshold = a.reproduction_threshold
                                  + a.strength * strength_repro_factor;
        if (a.energy_reserves < effective_threshold) {
            out.alive = true;
            out.dst_i = orig_i;  out.dst_j = orig_j;
            out.result = a;
            return;
        }
        a.metabolize(effective_threshold);
    }

    // ── 4. Movement ───────────────────────────────────────────────────────────
    int ci = orig_i, cj = orig_j;
    if (enable_movement && a.speed >= 1.0f) {
        // Social force — reads agents[] (read buffer), fully safe.
        float force_i = 0.0f, force_j = 0.0f;
        if (enable_sociality && (a.kin_attraction != 0.0f || a.threat_response != 0.0f)) {
            const float social_norm    = speciation_threshold * 3.0f;
            const float social_norm_sq = social_norm * social_norm;
            for (int d = 1; d <= 8; ++d) {
                int ni = orig_i + OFFSETS[d][0], nj = orig_j + OFFSETS[d][1];
                if (!in_range(ni, nj)) continue;
                const Agent& nb = agents[key(ni, nj)];
                if (!nb.alive) continue;
                float gdist_sq = 0.0f;
                for (int g = 0; g < 9; ++g) {
                    float diff = a.genome[g] - nb.genome[g];
                    gdist_sq += diff * diff;
                }
                if (gdist_sq >= social_norm_sq) continue;
                float similarity    = 1.0f - std::sqrt(gdist_sq) / social_norm;
                float trophism_diff = a.trophism - nb.trophism;
                float pull = a.kin_attraction  * similarity
                           + a.threat_response * trophism_diff;
                force_i += OFFSETS[d][0] * pull;
                force_j += OFFSETS[d][1] * pull;
            }
        }

        float social_strength = std::max(std::abs(a.kin_attraction), std::abs(a.threat_response));
        std::uniform_real_distribution<float> d01(0.0f, 1.0f);
        std::uniform_int_distribution<int>    d8(0, 7);
        int steps = static_cast<int>(std::min(a.speed, 20.0f));

        for (int s = 0; s < steps; ++s) {
            int roll;
            if ((force_i != 0.0f || force_j != 0.0f) && d01(rng_t) < social_strength) {
                int best = 1; float best_dot = -1e9f;
                for (int d = 1; d <= 8; ++d) {
                    float dot = OFFSETS[d][0] * force_i + OFFSETS[d][1] * force_j;
                    if (dot > best_dot) { best_dot = dot; best = d; }
                }
                roll = best;
            } else {
                roll = d8(rng_t) + 1;
            }
            int ni = ci + OFFSETS[roll][0], nj = cj + OFFSETS[roll][1];
            if (!in_range(ni, nj)) continue;
            if (a.hardiness <= static_cast<float>(LEVEL_DIFF[world[key(ni, nj)]])) continue;

            const Agent& occupant = agents[key(ni, nj)];
            if (occupant.alive) {
                if (!enable_predation) continue;
                float trophism_diff = a.trophism - occupant.trophism;
                if (trophism_diff <= predation_threshold) continue;

                // Movement predation: atomically claim occupant.
                int old = prey_claim_step[key(ni, nj)].exchange(
                              current_step, std::memory_order_acq_rel);
                if (old == current_step) continue;

                float steal_frac = a.trophism * predation_efficiency
                                 / (1.0f + (occupant.strength + occupant.hardiness) * predation_resistance);
                a.consume_food(occupant.energy_reserves * steal_frac);

                // Movement predation always kills — use infinity as sentinel.
                out.has_prey    = true;
                out.prey_i      = ni;  out.prey_j  = nj;
                out.prey_damage = std::numeric_limits<float>::infinity();
            }

            ci = ni; cj = nj;
        }
    }

    out.alive = true;
    out.dst_i  = ci;  out.dst_j  = cj;
    out.result = a;

    // ── 5. Reproduction ───────────────────────────────────────────────────────
    Agent newborn = Agent::reproduce_asexually(a, mutation_rate, rng_t);
    newborn.energy_reserves = std::max(1.0f, a.reproduction_threshold);

    std::uniform_int_distribution<int> d9(0, 8);
    int bs = d9(rng_t);

    // Prefer empty cell in read buffer; fall back to any in-range cell.
    int birth_ni = -1, birth_nj = -1;
    for (int b = 0; b < 9; ++b) {
        int ni = ci + OFFSETS[(bs + b) % 9][0], nj = cj + OFFSETS[(bs + b) % 9][1];
        if (!in_range(ni, nj)) continue;
        if (!agents[key(ni, nj)].alive) { birth_ni = ni; birth_nj = nj; break; }
    }
    if (birth_ni < 0) {
        for (int b = 0; b < 9; ++b) {
            int ni = ci + OFFSETS[(bs + b) % 9][0], nj = cj + OFFSETS[(bs + b) % 9][1];
            if (in_range(ni, nj)) { birth_ni = ni; birth_nj = nj; break; }
        }
    }
    if (birth_ni >= 0) {
        if (newborn.hardiness <= static_cast<float>(LEVEL_DIFF[world[key(birth_ni, birth_nj)]])) {
            out.d_exposure = 1;   // newborn dies on exposure — no birth queued
        } else {
            out.has_birth = true;
            out.birth_i = birth_ni;  out.birth_j = birth_nj;
            out.newborn = newborn;
        }
    }
}

// ── Phase 2: Sequential conflict resolution ───────────────────────────────────
//
// Applies predation damage, merges death counters, resolves cell conflicts via
// the same predation/competition rules as before, and writes results to agents_next.
// Rebuilds live / live_set from the final placement results.

void Simulation::resolve_placements(std::vector<AgentUpdate>& updates, StepResult& res)
{
    // Step 0: Clear previous live positions from the write buffer so stale
    // agents from the prior step don't linger in cells no new agent claims.
    for (auto& [i, j] : live)
        agents_next[key(i, j)].alive = false;

    // Step 1: Apply predation damage.
    // Build a reverse map: start-of-step cell key → index in updates[].
    // live[idx] is the source position for updates[idx].
    std::unordered_map<int, int> cell_to_idx;
    cell_to_idx.reserve(live.size() * 2);
    for (int idx = 0; idx < (int)live.size(); ++idx)
        cell_to_idx[key(live[idx].first, live[idx].second)] = idx;

    for (auto& atk : updates) {
        if (!atk.has_prey) continue;
        auto it = cell_to_idx.find(key(atk.prey_i, atk.prey_j));
        if (it == cell_to_idx.end()) continue;

        AgentUpdate& prey = updates[it->second];
        if (!prey.alive) continue;

        if (std::isinf(atk.prey_damage)) {
            // Movement predation — always kills.
            prey.alive    = false;
            prey.has_birth = false;
            res.deaths_predation++;
        } else {
            // Static predation — steal energy, may or may not kill.
            prey.result.energy_reserves -= atk.prey_damage;
            if (prey.result.energy_reserves <= 0.0f) {
                prey.alive    = false;
                prey.has_birth = false;
                res.deaths_predation++;
            } else if (enable_movement && prey.result.speed >= 1.0f) {
                // Prey survived; flee to a random empty neighbour (reads agents[] = start-of-step).
                int fs = dice8(rng);
                for (int f = 0; f < 8; ++f) {
                    int fd = (fs + f) % 8 + 1;
                    int fi = atk.prey_i + OFFSETS[fd][0];
                    int fj = atk.prey_j + OFFSETS[fd][1];
                    if (!in_range(fi, fj)) continue;
                    if (agents[key(fi, fj)].alive) continue;
                    prey.dst_i = fi;  prey.dst_j = fj;
                    break;
                }
            }
        }
    }

    // Step 2: Merge per-agent death counters into StepResult.
    for (auto& upd : updates) {
        res.deaths_aging      += upd.d_aging;
        res.deaths_starvation += upd.d_starvation;
        res.deaths_exposure   += upd.d_exposure;
    }

    // Step 3: Queue placement requests.
    for (auto& upd : updates) {
        if (upd.alive && upd.dst_i >= 0) {
            int c = key(upd.dst_i, upd.dst_j);
            if (cell_queue[c].empty()) queued_cells.push_back(c);
            cell_queue[c].push_back({upd.result, false});
        }
        if (upd.has_birth) {
            int c = key(upd.birth_i, upd.birth_j);
            if (cell_queue[c].empty()) queued_cells.push_back(c);
            cell_queue[c].push_back({upd.newborn, true});
        }
    }

    // Step 4: Resolve conflicts and write winners to agents_next.
    // Each cell with multiple claimants runs a tournament using the same
    // predation / competition rules as the original try_birth().
    live.clear();
    live_set.clear();

    for (int c : queued_cells) {
        auto& queue = cell_queue[c];
        int ci = c / N, cj = c % N;

        // Reduce queue to a single winner via pairwise tournament.
        int winner_idx = 0;
        for (int q = 1; q < (int)queue.size(); ++q) {
            Agent& chall   = queue[q].agent;
            Agent& current = queue[winner_idx].agent;
            float trophism_diff = chall.trophism - current.trophism;

            if (enable_predation && trophism_diff > predation_threshold) {
                float steal_frac = chall.trophism * predation_efficiency
                                 / (1.0f + (current.strength + current.hardiness) * predation_resistance);
                chall.consume_food(current.energy_reserves * steal_frac);
                winner_idx = q;
                res.deaths_predation++;
            } else if (enable_space_competition) {
                res.deaths_competition++;
                float total = chall.strength + current.strength;
                float p     = (total > 0.0f) ? chall.strength / total : 0.5f;
                if (uni01(rng) < p) winner_idx = q;
            } else {
                winner_idx = q;  // overwrite
            }
        }

        agents_next[c] = queue[winner_idx].agent;
        live    .push_back({ci, cj});
        live_set.insert(c);
        queue.clear();
    }
    queued_cells.clear();
}

// ── Speciation ────────────────────────────────────────────────────────────────

// Greedy nearest-representative clustering in 9D genome space.
// Uses squared distances throughout — avoids sqrt in the O(P×S) inner loop.
void Simulation::classify_species() {
    const float threshold_sq = speciation_threshold * speciation_threshold;
    std::vector<std::array<float,9>> rep_genomes;
    std::vector<int>                 rep_labels;
    int species_counter = 1;

    for (auto& [i, j] : live) {
        Agent& a = at(i, j);
        if (!a.alive) continue;

        bool assigned = false;
        if (!rep_genomes.empty()) {
            float best_dist_sq = std::numeric_limits<float>::max();
            int   best_idx     = -1;
            for (int s = 0; s < static_cast<int>(rep_genomes.size()); ++s) {
                float d = 0.0f;
                for (int k = 0; k < 9; ++k) {
                    float diff = a.genome[k] - rep_genomes[s][k];
                    d += diff * diff;
                }
                if (d < best_dist_sq) { best_dist_sq = d; best_idx = s; }
            }
            if (best_dist_sq < threshold_sq) {
                a.species = rep_labels[best_idx];
                assigned  = true;
            }
        }

        if (!assigned) {
            a.species = species_counter;
            std::array<float,9> g;
            for (int k = 0; k < 9; ++k) g[k] = a.genome[k];
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
        res.total_kin_attraction  += a.kin_attraction;
        res.total_threat_response += a.threat_response;
        ++res.population_count;
        species_set.insert(a.species);
    }
    res.species_counts = static_cast<int>(species_set.size());
}

// ── Main step ─────────────────────────────────────────────────────────────────

StepResult Simulation::step(int current_step) {
    StepResult res;
    const int  n_live = (int)live.size();
    std::vector<AgentUpdate> updates(n_live);

    // ── Phase 1: Parallel — each agent computes its update independently ───────
    // All reads from agents[] (start-of-step read buffer).
    // Food grid writes are per-cell with no concurrent conflict.
    // Predation claims use atomic exchange — no locks needed.
#pragma omp parallel for schedule(dynamic, 32)
    for (int idx = 0; idx < n_live; ++idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        compute_agent_update(idx, current_step, updates[idx], thread_rngs[tid]);
    }

    // ── Phase 2: Sequential — resolve conflicts, write agents_next ────────────
    resolve_placements(updates, res);

    // Swap buffers: agents_next becomes the new read state.
    std::swap(agents, agents_next);

    // classify_species and collect_metrics read agents[] (new state) via live[].
    classify_species();
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
                const Agent& a = agents[i*N+j];
                buf(i,j,0) = a.alive ? a.color[0] : 0.0f;
                buf(i,j,1) = a.alive ? a.color[1] : 0.0f;
                buf(i,j,2) = a.alive ? a.color[2] : 0.0f;
            }
        return out;
    }

    // Resolve attribute name to a member pointer once — avoids N*N string comparisons.
    float Agent::* field = nullptr;
    if      (attr == "age")                    field = &Agent::age;
    else if (attr == "lifespan")               field = &Agent::lifespan;
    else if (attr == "hardiness")              field = &Agent::hardiness;
    else if (attr == "strength")               field = &Agent::strength;
    else if (attr == "metabolism")             field = &Agent::metabolism;
    else if (attr == "reproduction_threshold") field = &Agent::reproduction_threshold;
    else if (attr == "speed")                  field = &Agent::speed;
    else if (attr == "trophism")               field = &Agent::trophism;
    else if (attr == "kin_attraction")         field = &Agent::kin_attraction;
    else if (attr == "threat_response")        field = &Agent::threat_response;
    else if (attr == "genetic_distance")       field = &Agent::genetic_distance;

    py::array_t<float> out({N, N});
    auto buf = out.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const Agent& a = agents[i*N+j];
            buf(i, j) = (a.alive && field) ? a.*field : 0.0f;
        }
    return out;
}
