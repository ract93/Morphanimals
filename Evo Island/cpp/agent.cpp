#include "agent.h"
#include <algorithm>

void Agent::decode_genome() {
    lifespan               = genome[0];
    hardiness              = genome[1];
    strength               = genome[2];
    metabolism             = genome[3];
    reproduction_threshold = genome[4];
}

void Agent::calculate_genetic_distance() {
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        float d = genome[i] - ANCESTOR[i];
        sum += d * d;
    }
    genetic_distance = std::sqrt(sum);
}

void Agent::kill() {
    alive                = false;
    age                  = 0.0f;
    energy_reserves      = 0.0f;
    genetic_distance     = 0.0f;
    species              = 0;
    lifespan             = 0.0f;
    hardiness            = 0.0f;
    strength             = 0.0f;
    metabolism           = 0.0f;
    reproduction_threshold = 0.0f;
    for (int i = 0; i < 5; ++i) genome[i] = 0.0f;
    for (int i = 0; i < 3; ++i) color[i]  = 0.0f;
}

void Agent::age_step(bool enable_aging, std::mt19937& rng) {
    age += 1.0f;
    if (!enable_aging) return;

    float midpoint = lifespan / 2.0f;
    if (midpoint == 0.0f) {
        kill();
        return;
    }
    constexpr float steepness = 10.0f;
    float death_prob = 1.0f / (1.0f + std::exp(
        -steepness * (age - midpoint) / midpoint));

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    if (uni(rng) < death_prob)
        kill();
}

void Agent::consume_food(float f) { energy_reserves += f; }
void Agent::metabolize(float e)   { energy_reserves -= e; }

// Replicates matplotlib.colors.hsv_to_rgb with the same H/S/V derivation
// used in the Python genome_to_color.
void Agent::genome_to_color(const float g[5], float c[3]) {
    float gmin = g[0], gmax = g[0];
    for (int i = 1; i < 5; ++i) {
        if (g[i] < gmin) gmin = g[i];
        if (g[i] > gmax) gmax = g[i];
    }
    float range = (gmax != gmin) ? (gmax - gmin) : 1.0f;
    float norm[5];
    for (int i = 0; i < 5; ++i)
        norm[i] = (g[i] - gmin) / range;

    // Match Python: hue/360 after (norm[0]*360)%360 == norm[0] when norm[0] in [0,1)
    float H = norm[0];                    // [0, 1)
    float S = 0.5f + norm[1] * 0.5f;     // [0.5, 1.0]
    float V = 0.5f + norm[2] * 0.5f;     // [0.5, 1.0]

    // Standard HSV -> RGB
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

Agent Agent::create_live(const float g[5]) {
    Agent a;
    a.alive = true;
    for (int i = 0; i < 5; ++i) a.genome[i] = g[i];
    genome_to_color(a.genome, a.color);
    a.decode_genome();
    a.calculate_genetic_distance();
    a.energy_reserves = 5.0f;
    return a;
}

Agent Agent::create_live_default() {
    return create_live(ANCESTOR);
}

void Agent::mutate_genome(const float in[5], float out[5],
                          float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float>       norm_dist(0.0f, 2.0f);
    for (int i = 0; i < 5; ++i) {
        float effect = (uni(rng) < mutation_rate) ? norm_dist(rng) : 0.0f;
        out[i] = std::min(100.0f, std::max(0.0f, in[i] + effect));
    }
}

Agent Agent::reproduce_asexually(const Agent& parent,
                                  float mutation_rate, std::mt19937& rng) {
    float child_genome[5];
    mutate_genome(parent.genome, child_genome, mutation_rate, rng);
    return create_live(child_genome);
}
