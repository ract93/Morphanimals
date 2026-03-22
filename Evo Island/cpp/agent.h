#pragma once
#include <cmath>
#include <random>

// An Agent occupies one cell of the grid. Dead agents remain as zeroed
// placeholders so the grid can be indexed directly without pointer chasing.
struct Agent {
    bool  alive             = false;
    float genome[7]         = {0,0,0,0,0,0,0};
    float color[3]          = {0,0,0};  // RGB [0,1], derived from genome
    float age               = 0.0f;
    float energy_reserves   = 0.0f;
    float genetic_distance  = 0.0f;  // Euclidean distance from ANCESTOR genome
    int   species           = 0;
    int   last_step_acted   = -1;    // prevents double-acting after moving into a
                                     // cell still present in the step snapshot

    // Decoded traits — mirrors genome indices 0-6.
    // Updated by decode_genome() after any genome write.
    float lifespan               = 0.0f;  // [0] shifts the sigmoid death curve rightward
    float hardiness              = 0.0f;  // [1] must exceed terrain difficulty to survive
    float strength               = 0.0f;  // [2] wins contested cells; costs energy to maintain
    float metabolism             = 0.0f;  // [3] burn rate AND food extraction rate
    float reproduction_threshold = 0.0f;  // [4] minimum energy needed to reproduce
    float speed                  = 0.0f;  // [5] random-walk steps per turn; 0 = sessile
    float trophism               = 0.0f;  // [6] decoded from genome[6]/100; 0=autotroph, 1=predator

    // All trials begin with a single agent carrying these values.
    // trophism=0 keeps the ancestor as a pure autotroph (plant-like) baseline.
    static constexpr float ANCESTOR[7] = {20.0f, 10.0f, 5.0f, 5.0f, 3.0f, 0.0f, 0.0f};

    Agent() = default;

    void decode_genome();
    void calculate_genetic_distance();
    void kill();
    void age_step(bool enable_aging, std::mt19937& rng);
    void consume_food(float f);
    void metabolize(float e);

    static Agent create_live(const float genome[7]);
    static Agent create_live_default();
    static void  mutate_genome(const float in[7], float out[7],
                               float mutation_rate, std::mt19937& rng);
    static Agent reproduce_asexually(const Agent& parent,
                                     float mutation_rate, std::mt19937& rng);
    // Maps genome to RGB via HSV — related genomes share hues.
    static void  genome_to_color(const float genome[7], float color[3]);
};
