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

// Map all 9 genome values to RGB via a fixed random projection.
// Each gene is normalised to [0,1] and centred on the ANCESTOR so the
// initial population sits near the middle of the colour cube.  The fixed
// matrix means the same genome always produces the same colour — genetic
// neighbours share similar colours regardless of which genes drifted.
void Agent::genome_to_color(const float g[9], float c[3]) {
    // Fixed 9×3 projection matrix — entries sampled offline from N(0,1).
    // Never change these: doing so would recolour every existing result.
    static constexpr float W[9][3] = {
        { 0.4122f, -0.8090f,  0.4226f},
        {-0.6577f,  0.1919f,  0.6122f},
        { 0.5062f,  0.5530f, -0.2150f},
        {-0.2348f, -0.4685f, -0.7312f},
        { 0.7819f, -0.1234f,  0.3456f},
        {-0.3291f,  0.6782f, -0.5521f},
        { 0.1456f,  0.3891f,  0.8012f},
        {-0.5678f, -0.2345f,  0.1234f},
        { 0.6789f,  0.4512f, -0.3901f},
    };
    // ANCESTOR normalised to [0,1] — centres projection so initial population
    // maps to ~(0.5, 0.5, 0.5) and diverges outward as evolution proceeds.
    static constexpr float ANORM[9] = {
        0.20f, 0.10f, 0.05f, 0.05f, 0.03f, 0.00f, 0.00f, 0.50f, 0.50f
    };
    constexpr float SCALE = 3.0f;
    for (int j = 0; j < 3; ++j) {
        float y = 0.0f;
        for (int i = 0; i < 9; ++i)
            y += W[i][j] * (g[i] / 100.0f - ANORM[i]);
        c[j] = 1.0f / (1.0f + std::exp(-y * SCALE));
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
