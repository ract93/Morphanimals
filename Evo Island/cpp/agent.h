#pragma once
#include <cmath>
#include <random>

struct Agent {
    bool  alive             = false;
    float genome[5]         = {0,0,0,0,0};
    float color[3]          = {0,0,0};   // RGB [0,1]
    float age               = 0.0f;
    float energy_reserves   = 0.0f;
    float genetic_distance  = 0.0f;
    int   species           = 0;

    // Decoded traits (mirror genome indices 0-4)
    float lifespan               = 0.0f;
    float hardiness              = 0.0f;
    float strength               = 0.0f;
    float metabolism             = 0.0f;
    float reproduction_threshold = 0.0f;

    // Common ancestor genome [lifespan, hardiness, strength, metabolism, reproduction_threshold]
    static constexpr float ANCESTOR[5] = {20.0f, 10.0f, 5.0f, 5.0f, 3.0f};

    Agent() = default;

    void decode_genome();
    void calculate_genetic_distance();
    void kill();
    void age_step(bool enable_aging, std::mt19937& rng);
    void consume_food(float f);
    void metabolize(float e);

    static Agent create_live(const float genome[5]);
    static Agent create_live_default();
    static void  mutate_genome(const float in[5], float out[5],
                               float mutation_rate, std::mt19937& rng);
    static Agent reproduce_asexually(const Agent& parent,
                                     float mutation_rate, std::mt19937& rng);
    static void  genome_to_color(const float genome[5], float color[3]);
};
