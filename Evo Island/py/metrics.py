import csv


class SimulationMetrics:
    def __init__(self):
        self.species_counts = 0
        self.population_count = 0  # Keep track of the population to calculate averages
        self.cumulative_deaths = 0
        self.deaths_from_aging = 0
        self.death_from_competition = 0
        self.deaths_from_starvation = 0
        self.deaths_from_exposure = 0

        self.total_age = 0  # Use this to calculate average_age
        self.total_lifespan = 0  # Use this for average_lifespan
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0

        self.average_age = 0
        self.average_lifespan = 0
        self.average_strength = 0
        self.average_hardiness = 0
        self.average_metabolism = 0
        self.average_reproduction_threshold = 0

        # CSV logging fields
        self.csv_logging_enabled = False

    def enable_csv_logging(self, filepath):
        self.csv_logging_enabled = True
        self.filepath = filepath
        self.fields = [
            "Timestep",
            "Population Count",
            "Cumulative Deaths",
            "Deaths from Aging",
            "Deaths from Competition",
            "Deaths from Starvation",
            "Deaths from Exposure",
            "Average Age",
            "Average Lifespan",
            "Average Strength",
            "Average Hardiness",
            "Average Metabolism",
            "Average Reproduction Threshold",
            "Number of Species",
        ]
        self.csv_file = open(
            self.filepath, "w", newline="", buffering=1
        )  # Line buffering
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
            "Average Age": self.average_age,
            "Average Lifespan": self.average_lifespan,
            "Average Strength": self.average_strength,
            "Average Hardiness": self.average_hardiness,
            "Average Metabolism": self.average_metabolism,
            "Average Reproduction Threshold": self.average_reproduction_threshold,
            "Number of Species": self.species_counts,
        }
        self.writer.writerow(row)

    def close_csv_logging(self):
        if self.csv_logging_enabled:
            self.csv_file.close()

    def update_agent_metrics(self, agent_matrix, live_agents):
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
        # Ensure division by zero is handled
        population = max(self.population_count, 1)
        self.average_age = self.total_age / population
        self.average_lifespan = self.total_lifespan / population
        self.average_strength = self.total_strength / population
        self.average_hardiness = self.total_hardiness / population
        self.average_metabolism = self.total_metabolism / population
        self.average_reproduction_threshold = (
            self.total_reproduction_threshold / population
        )

    def reset_averages(self):
        # Reset total stats (but not cumulative ones) for the next calculation step
        self.total_age = 0
        self.total_lifespan = 0
        self.total_strength = 0
        self.total_hardiness = 0
        self.total_metabolism = 0
        self.total_reproduction_threshold = 0
        self.population_count = 0

    def get_state_string(self, trial_num, step, total_steps):
        return (
            f"Trial {trial_num} | Step {step}/{total_steps} | "
            f"Pop:{self.population_count} | "
            f"Deaths:{self.cumulative_deaths}(Age:{self.deaths_from_aging} Comp:{self.death_from_competition} Starv:{self.deaths_from_starvation} Exp:{self.deaths_from_exposure}) | "
            f"Age:{self.average_age:.1f} Life:{self.average_lifespan:.1f} Str:{self.average_strength:.1f} "
            f"Hard:{self.average_hardiness:.1f} Metab:{self.average_metabolism:.1f} Repr:{self.average_reproduction_threshold:.1f} | "
            f"Species:{self.species_counts}"
        )
