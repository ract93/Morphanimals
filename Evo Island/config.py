import json
import os

# Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("config.json", "r") as config_file:
    config = json.load(config_file)
