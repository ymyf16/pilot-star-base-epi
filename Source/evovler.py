#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We are expecting all numpy related objects
# Default ints/floats are numpy.int32 and numpy.float32
#
# Python 3.12.4: conda activate star-epi-pre
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List
import numpy.typing as npt
import copy as cp
import pandas as pd
import sys


@typechecked # for debugging purposes
class EA:
    def __init__(self, seed: np.int32, pop_size: np.int32, branch_cnt_max: np.int32):
        # arguments needed to run
        self.seed = seed
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.branch_cnt_max = branch_cnt_max

        # instantiate the population as empty
        self.population = np.array([])


    # Run the EA
    # EA runs until a max_eval evaluations are met
    def Evolve(self, max_evals: np.int32) -> None:
        pass


    #####################
    # EVALUATION STUFF
    #####################

    # evaluate the set of pipelines
    def Evaluation(self):
        pass

    # identify parent pipelines for reproduction
    def ParentSelection(self) -> List[int]:
        pass

    # indentify surviving pipelines for next generation
    def SurvivialSelection(self) -> List[int]:
        pass

    # create the initial population of pipelines
    def InitializePopulation(self) -> None:
        pass
