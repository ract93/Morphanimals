import numpy as np


# Speciation Functions
def classify_species(agent_matrix, live_agents, threshold):
    species_counter = 1
    rep_genomes = []   # parallel arrays — avoids attribute lookup inside hot loop
    rep_labels = []

    for (i, j) in live_agents:
        agent = agent_matrix[i][j]
        assigned = False
        if rep_genomes:
            rep_arr = np.stack(rep_genomes)                          # (S, 5)
            dists = np.linalg.norm(rep_arr - agent.genome, axis=1)  # (S,) — one call
            idx = int(np.argmin(dists))
            if dists[idx] < threshold:
                agent.species = rep_labels[idx]
                assigned = True

        if not assigned:
            agent.species = species_counter
            rep_genomes.append(agent.genome)
            rep_labels.append(species_counter)
            species_counter += 1
