""" 
The goal of this script is to align xl_sindy algorithm with the Mujoco environment.
The script takes in input :
- the mujoco environment .xml file
- the lagrangian to be identified
- the way forces function should be created
- some optional hyperparameters
"""

from dataclasses import dataclass
import tyro


@dataclass
class Args: