"""
This script attach the different exploration metric guess to info file of experiment
(override past metric computation)
"""

#tyro cly dependencies
from dataclasses import dataclass
from dataclasses import field
import tyro

import numpy as np
import json

@dataclass
class Args:
    experiment_file: str = "None"
    """the experiment file"""


if __name__ == "__main__":

    args = tyro.cli(Args)

    with open(args.experiment_file+"json", 'r') as json_file:
        simulation_dict = json.load(json_file)