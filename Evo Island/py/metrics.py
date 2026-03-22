import csv


class SimulationMetrics:
    """Accumulates per-step simulation statistics.

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

        self.total_age = 0
        self.total_lifespan = 0
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0
        self.total_speed = 0
        self.total_trophism = 0

        self.average_age = 0
        self.average_lifespan = 0
        self.average_strength = 0
        self.average_hardiness = 0
        self.average_metabolism = 0
        self.average_reproduction_threshold = 0
        self.average_speed = 0
        self.average_trophism = 0

        self.csv_logging_enabled = False

    def enable_csv_logging(self, filepath):
        self.csv_logging_enabled = True
        self.filepath = filepath
        self.fields = [
            "Timestep", "Population Count", "Cumulative Deaths",
            "Deaths from Aging", "Deaths from Competition",
            "Deaths from Starvation", "Deaths from Exposure", "Deaths from Predation",
            "Average Age", "Average Lifespan", "Average Strength",
            "Average Hardiness", "Average Metabolism",
            "Average Reproduction Threshold", "Average Speed", "Average Trophism",
            "Number of Species",
        ]
        # Line-buffered so data reaches disk incrementally, not all at the end.
        self.csv_file = open(self.filepath, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        self.writer.writeheader()

    def log_metrics(self, timestep):
        if not self.csv_logging_enabled:
            return
        row = {
            "Timestep": timestep,
            "Population Count": self.population_count,
            "Cumulative Deaths": self.cumulative_deaths,
            "Deaths from Aging": self.deaths_from_aging,
            "Deaths from Competition": self.death_from_competition,
            "Deaths from Starvation": self.deaths_from_starvation,
            "Deaths from Exposure": self.deaths_from_exposure,
            "Deaths from Predation": self.deaths_from_predation,
            "Average Age": self.average_age,
            "Average Lifespan": self.average_lifespan,
            "Average Strength": self.average_strength,
            "Average Hardiness": self.average_hardiness,
            "Average Metabolism": self.average_metabolism,
            "Average Reproduction Threshold": self.average_reproduction_threshold,
            "Average Speed": self.average_speed,
            "Average Trophism": self.average_trophism,
            "Number of Species": self.species_counts,
        }
        self.writer.writerow(row)

    def close_csv_logging(self):
        if self.csv_logging_enabled:
            self.csv_file.close()

    def update_agent_metrics(self, agent_matrix, live_agents):
        """Python-only path — not used by the C++ core."""
        species_set = set()
        for (i, j) in live_agents:
            agent = agent_matrix[i][j]
            self.total_age += agent.age
            self.total_lifespan += agent.lifespan
            self.total_strength += agent.strength
            self.total_hardiness += agent.hardiness
            self.total_metabolism += agent.metabolism
            self.total_reproduction_threshold += agent.reproduction_threshold
            self.population_count += 1
            species_set.add(agent.species)
        self.species_counts = len(species_set)

    def calculate_averages(self):
        population = max(self.population_count, 1)
        self.average_age = self.total_age / population
        self.average_lifespan = self.total_lifespan / population
        self.average_strength = self.total_strength / population
        self.average_hardiness = self.total_hardiness / population
        self.average_metabolism = self.total_metabolism / population
        self.average_reproduction_threshold = self.total_reproduction_threshold / population
        self.average_speed = self.total_speed / population
        self.average_trophism = self.total_trophism / population

    def reset_averages(self):
        """Clear per-step accumulators. Call after calculate_averages() each step."""
        self.total_age = 0
        self.total_lifespan = 0
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0
        self.total_speed = 0
        self.total_trophism = 0
        self.population_count = 0

    def get_state_string(self, trial_num, step, total_steps):
        return (
            f"Trial {trial_num} | Step {step}/{total_steps} | "
            f"Pop:{self.population_count} | "
            f"Deaths:{self.cumulative_deaths}(Age:{self.deaths_from_aging} Comp:{self.death_from_competition} Starv:{self.deaths_from_starvation} Exp:{self.deaths_from_exposure} Pred:{self.deaths_from_predation}) | "
            f"Age:{self.average_age:.1f} Life:{self.average_lifespan:.1f} Str:{self.average_strength:.1f} "
            f"Hard:{self.average_hardiness:.1f} Metab:{self.average_metabolism:.1f} "
            f"Repr:{self.average_reproduction_threshold:.1f} Spd:{self.average_speed:.1f} "
            f"Troph:{self.average_trophism:.2f} | "
            f"Species:{self.species_counts}"
        )
