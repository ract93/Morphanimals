import json
import os
import warnings

warnings.filterwarnings("ignore", ".*missing an id field.*")

from experiment import run_experiment


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.json", "r") as f:
        config = json.load(f)
    run_experiment(config)


if __name__ == "__main__":
    main()
