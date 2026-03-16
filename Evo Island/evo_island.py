import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", ".*missing an id field.*")

# Add py/ to the path so all module imports inside py/ resolve without changes.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py"))

from experiment import run_experiment


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.json", "r") as f:
        config = json.load(f)
    run_experiment(config)


if __name__ == "__main__":
    main()
