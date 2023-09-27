import math

import numpy as np
from gym import utils

from trp_env.envs.two_resource_env import TwoResourceEnv
from trp_env.envs.ant_trp_env import MyAntEnv


class MyLowGearAntEnv(MyAntEnv, utils.EzPickle):
    # TODO: MaKe low-gear version as an option in AntGather
    FILE = "low_gear_ratio_ant.xml"


class LowGearAntTwoResourceEnv(TwoResourceEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MyLowGearAntEnv
    ORI_IND = 3


class LowGearAntSmallTwoResourceEnv(TwoResourceEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MyLowGearAntEnv
    ORI_IND = 3

    def __init__(self,
                 n_blue=6,
                 n_red=4,
                 activity_range=6.,
                 n_bins=20,
                 sensor_range=16,
                 *args, **kwargs):
        super().__init__(
            n_blue=n_blue,
            n_red=n_red,
            n_bins=n_bins,
            activity_range=activity_range,
            sensor_range=sensor_range,
            *args, **kwargs
        )
