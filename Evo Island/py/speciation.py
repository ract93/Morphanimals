import numpy as np


# Greedy nearest-representative clustering in genome space.
# Each agent joins the closest existing species if within `threshold`; otherwise
# it founds a new one. Order-dependent but fast — mirrors the C++ implementation.
def classify_species(agent_matrix, live_agents, threshold):
    species_counter = 1
    rep_genomes = []
    rep_labels  = []

    for (i, j) in live_agents:
        agent = agent_matrix[i][j]
        assigned = False
        if rep_genomes:
            rep_arr = np.stack(rep_genomes)
            dists   = np.linalg.norm(rep_arr - agent.genome, axis=1)
            idx     = int(np.argmin(dists))
            if dists[idx] < threshold:
                agent.species = rep_labels[idx]
                assigned = True

        if not assigned:
            agent.species = species_counter
            rep_genomes.append(agent.genome)
            rep_labels.append(species_counter)
            species_counter += 1
