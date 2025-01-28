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
import mujoco
import mujoco.viewer


@dataclass
class Args:
    mujoco_xml: str
    """the path of the mujoco environment .xml file"""
    lagrangian: str = None
    """the lagrangian to be identified using simpy formalism and symbol matrix s_m[d,i] (d : derivative order (starting from 1, 0 is index for external forces) and i : index ) formalism"""
    mujoco_angle_offset:float = -3.14
    """angle offset to be applied to the mujoco environment"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    mujoco_model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    mujoco_data = mujoco.MjData(mujoco_model)

    #print(args)