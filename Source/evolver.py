#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We are expecting all numpy related objects
# Default ints/floats are numpy.int32 and numpy.float32
#
# Python 3.12.4: conda activate star-epi-pre
#####################################################################################################

import numpy as np
import selection
from typeguard import typechecked
from typing import List
import numpy.typing as npt
import copy as cp
import pandas as pd
import sys
# import nsga_tool_box as nsga

class Pipeline:
    def __init__(self) -> None:
        pass

    def printer(self) -> None:
        print("Pipeline")

@typechecked # for debugging purposes
class EA:
    def __init__(self, seed: np.int32, pop_size: np.int32, branch_cnt_max: np.int32, selector: selection.Selection):
        # arguments needed to run
        self.seed = seed
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.branch_cnt_max = branch_cnt_max

        # instantiate the population as empty
        self.population = np.array([])

        # instnatiate the selector passed by user
        self.selector = selector