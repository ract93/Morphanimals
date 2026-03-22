#pragma once
#include <cmath>
#include <random>

// An Agent occupies one cell of the grid. Dead agents remain as zeroed
// placeholders so the grid can be indexed directly without pointer chasing.
struct Agent {
    bool  alive             = false;
    float genome[9]         = {0,0,0,0,0,0,0,0,0};
    float color[3]          = {0,0,0};  // RGB [0,1], derived from genome
    float age               = 0.0f;
    float energy_reserves   = 0.0f;
    float genetic_distance  = 0.0f;  // Euclidean distance from ANCESTOR genome
    int   species           = 0;
    int   last_step_acted   = -1;    // prevents double-acting after moving into a
                                     // cell still present in the step snapshot

    // Decoded traits — mirrors genome indices 0-8.
    // Updated by decode_genome() after any genome write.
    float lifespan               = 0.0f;  // [0] shifts the sigmoid death curve rightward
    float hardiness              = 0.0f;  // [1] must exceed terrain difficulty to survive; also predation resistance
    float strength               = 0.0f;  // [2] wins contested cells; also predation resistance
    float metabolism             = 0.0f;  // [3] burn rate AND food extraction rate
    float reproduction_threshold = 0.0f;  // [4] minimum energy needed to reproduce
    float speed                  = 0.0f;  // [5] random-walk steps per turn; 0 = sessile
    float trophism               = 0.0f;  // [6] genome[6]/100;      0=autotroph, 1=predator
    float kin_attraction         = 0.0f;  // [7] (genome[7]-50)/50;  -1=flee kin,  0=neutral, +1=seek kin
    float threat_response        = 0.0f;  // [8] (genome[8]-50)/50;  scales by (self.trophism - neighbor.trophism):
                                          //   +1 = seek lower-trophism (prey pursuit) / flee higher-trophism (predator avoidance)
                                          //   -1 = inverted (maladaptive; purged by selection)

    // All trials begin with a single agent carrying these values.
    // kin_attraction=50 and threat_response=50 both decode to 0 — neutral random walk.
    static constexpr float ANCESTOR[9] = {20.0f, 10.0f, 5.0f, 5.0f, 3.0f, 0.0f, 0.0f, 50.0f, 50.0f};

    Agent() = default;

    void decode_genome();
    void calculate_genetic_distance();
    void kill();
    void age_step(bool enable_aging, std::mt19937& rng);
    void consume_food(float f);
    void metabolize(float e);

    static Agent create_live(const float genome[9]);
    static Agent create_live_default();
    static void  mutate_genome(const float in[9], float out[9],
                               float mutation_rate, std::mt19937& rng);
    static Agent reproduce_asexually(const Agent& parent,
                                     float mutation_rate, std::mt19937& rng);
    // Maps genome to RGB via HSV — related genomes share hues.
    static void  genome_to_color(const float genome[9], float color[3]);
};
