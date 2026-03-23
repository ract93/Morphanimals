from genes import GENES


class SimulationMetrics:
    """Accumulates per-step simulation statistics for terminal display.

    Death counts are running totals that never reset. Trait totals and
    population_count reset each step after averages are calculated.
    """

    def __init__(self):
        self.species_counts = 0
        self.population_count = 0
        self.cumulative_deaths = 0
        self.deaths_from_aging = 0
        self.death_from_competition = 0
        self.deaths_from_starvation = 0
        self.deaths_from_exposure = 0
        self.deaths_from_predation = 0

        # Age is a runtime property, not a gene — tracked separately
        self.total_age = 0
        self.average_age = 0

        # Per-step totals and averages — auto-generated from GENES registry
        for attr, *_ in GENES:
            setattr(self, f"total_{attr}", 0)
            setattr(self, f"average_{attr}", 0)

    def calculate_averages(self):
        population = max(self.population_count, 1)
        self.average_age = self.total_age / population
        for attr, *_ in GENES:
            setattr(self, f"average_{attr}", getattr(self, f"total_{attr}") / population)

    def reset_averages(self):
        """Clear per-step accumulators. Call after calculate_averages() each step."""
        self.total_age = 0
        self.population_count = 0
        for attr, *_ in GENES:
            setattr(self, f"total_{attr}", 0)

    def get_state_string(self, trial_num, step, total_steps):
        gene_parts = " ".join(
            f"{attr[:4].title()}:{getattr(self, f'average_{attr}'):.2f}"
            for attr, *_ in GENES
        )
        return (
            f"Trial {trial_num} | Step {step}/{total_steps} | "
            f"Pop:{self.population_count} | "
            f"Deaths:{self.cumulative_deaths}"
            f"(Age:{self.deaths_from_aging} Comp:{self.death_from_competition} "
            f"Starv:{self.deaths_from_starvation} Exp:{self.deaths_from_exposure} "
            f"Pred:{self.deaths_from_predation}) | "
            f"Age:{self.average_age:.1f} {gene_parts} | "
            f"Species:{self.species_counts}"
        )
