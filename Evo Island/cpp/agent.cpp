#include "agent.h"
#include <algorithm>

void Agent::decode_genome() {
    lifespan               = genome[0];
    hardiness              = genome[1];
    strength               = genome[2];
    metabolism             = genome[3];
    reproduction_threshold = genome[4];
    speed                  = genome[5];
    trophism        = genome[6] / 100.0f;           // [0,100] → [0,1]
    kin_attraction  = (genome[7] - 50.0f) / 50.0f; // [0,100] → [-1,1]
    threat_response = (genome[8] - 50.0f) / 50.0f; // [0,100] → [-1,1]
}

void Agent::calculate_genetic_distance() {
    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) {
        float d = genome[i] - ANCESTOR[i];
        sum += d * d;
    }
    genetic_distance = std::sqrt(sum);
}

void Agent::kill() {
    alive                  = false;
    age                    = 0.0f;
    energy_reserves        = 0.0f;
    genetic_distance       = 0.0f;
    species                = 0;
    last_step_acted        = -1;
    lifespan               = 0.0f;
    hardiness              = 0.0f;
    strength               = 0.0f;
    metabolism             = 0.0f;
    reproduction_threshold = 0.0f;
    speed                  = 0.0f;
    trophism        = 0.0f;
    kin_attraction  = 0.0f;
    threat_response = 0.0f;
    for (int i = 0; i < 9; ++i) genome[i] = 0.0f;
    for (int i = 0; i < 3; ++i) color[i]  = 0.0f;
}

// Death probability follows a logistic curve centred at lifespan/2.
// steepness=10 produces a sharp S — near-zero early, near-certain past midpoint.
void Agent::age_step(bool enable_aging, std::mt19937& rng) {
    age += 1.0f;
    if (!enable_aging) return;

    float midpoint = lifespan / 2.0f;
    if (midpoint == 0.0f) { kill(); return; }

    constexpr float steepness = 10.0f;
    float death_prob = 1.0f / (1.0f + std::exp(
        -steepness * (age - midpoint) / midpoint));

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    if (uni(rng) < death_prob)
        kill();
}

void Agent::consume_food(float f) { energy_reserves += f; }
void Agent::metabolize(float e)   { energy_reserves -= e; }

// Normalise all 9 gene values relative to the genome's own min/max, then map
// index-0 → hue, index-1 → saturation, index-2 → value. Agents with similar
// genomes share similar hues, making species clusters visually identifiable.
void Agent::genome_to_color(const float g[9], float c[3]) {
    float gmin = g[0], gmax = g[0];
    for (int i = 1; i < 9; ++i) {
        if (g[i] < gmin) gmin = g[i];
        if (g[i] > gmax) gmax = g[i];
    }
    float range = (gmax != gmin) ? (gmax - gmin) : 1.0f;
    float norm[9];
    for (int i = 0; i < 9; ++i)
        norm[i] = (g[i] - gmin) / range;

    float H = norm[0];
    float S = 0.5f + norm[1] * 0.5f;  // clamp to [0.5, 1.0] — avoids washed-out colours
    float V = 0.5f + norm[2] * 0.5f;  // clamp to [0.5, 1.0] — avoids black cells

    float hh = H * 6.0f;
    int   sector = static_cast<int>(hh) % 6;
    float ff = hh - static_cast<int>(hh);
    float p  = V * (1.0f - S);
    float q  = V * (1.0f - S * ff);
    float t  = V * (1.0f - S * (1.0f - ff));

    switch (sector) {
        case 0: c[0]=V; c[1]=t; c[2]=p; break;
        case 1: c[0]=q; c[1]=V; c[2]=p; break;
        case 2: c[0]=p; c[1]=V; c[2]=t; break;
        case 3: c[0]=p; c[1]=q; c[2]=V; break;
        case 4: c[0]=t; c[1]=p; c[2]=V; break;
        default: c[0]=V; c[1]=p; c[2]=q; break;
    }
}

Agent Agent::create_live(const float g[9]) {
    Agent a;
    a.alive = true;
    for (int i = 0; i < 9; ++i) a.genome[i] = g[i];
    genome_to_color(a.genome, a.color);
    a.decode_genome();
    a.calculate_genetic_distance();
    a.energy_reserves = 5.0f;
    return a;
}

Agent Agent::create_live_default() {
    return create_live(ANCESTOR);
}

// Each gene mutates independently with probability mutation_rate;
// effect is sampled from N(0, 2) and clamped to [0, 100].
void Agent::mutate_genome(const float in[9], float out[9],
                          float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float>       norm_dist(0.0f, 2.0f);
    for (int i = 0; i < 9; ++i) {
        float effect = (uni(rng) < mutation_rate) ? norm_dist(rng) : 0.0f;
        out[i] = std::min(100.0f, std::max(0.0f, in[i] + effect));
    }
}

Agent Agent::reproduce_asexually(const Agent& parent,
                                  float mutation_rate, std::mt19937& rng) {
    float child_genome[9];
    mutate_genome(parent.genome, child_genome, mutation_rate, rng);
    return create_live(child_genome);
}
