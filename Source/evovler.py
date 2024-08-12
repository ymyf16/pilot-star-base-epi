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
import nsga_tool_box as nsga

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


    # Run NSGA-II for a specified number of generations
    def Evolve(self, max_gens: np.int32) -> None:

        # create the initial population
        self.InitializePopulation()

        # for each generation
        for gen in range(max_gens):

            # evaluate the initial population
            # transform pipeline into scikit learn pipeline
            pop_obj_scores = self.Evaluation(self.population)


            # select parent pipelines
            parents = self.ParentSelection(pop_obj_scores)

            # create offspring pipelines
            offspring = self.Reproduction(parents)

            # evaluate the offspring
            off_obj_scores = self.Evaluation(offspring)

            # select surviving pipelines
            # TODO: make sure that the np,concatenate is correctly stacking the scores
            self.SurvivialSelection(np.concatenate((pop_obj_scores, off_obj_scores), axis=None))

    # evaluate the set of pipelines
    def Evaluation(self, pipelines: npt.NDArray) -> npt.NDArray[np.float32]:
        assert pipelines.shape[0] > 0
        assert isinstance(pipelines[0], Pipeline)

        scores = np.zeros((pipelines.shape[0], 2), dtype=np.float32)

        return scores

    # identify parent pipelines for reproduction
    def ParentSelection(self, obj_scores: npt.NDArray[np.float32], pnt_cnt: np.uint16) -> List[np.uint16]:
        # quick check to make sure that elements in scores are numpy arrays with np.float32
        assert all(isinstance(x, np.ndarray) for x in obj_scores)
        assert isinstance(obj_scores[0][0], np.float32)

        # get nondominated fronts from current population objective scores
        ranks = self.selector.NonDominatedSorting(obj_scores=obj_scores)
        # get crowding distance for each solution in each front
        distances = np.zeros(self.pop_size, dtype=np.float32)
        for front in ranks:
            distances[front] = self.selector.CrowdingDistance(obj_scores=obj_scores[front], count=2)

        # select parents
        parents = []
        for _ in range(pnt_cnt):
            winner = self.selector.NonDominatedBinaryTournament(ranks=ranks, distances=distances,rng=self.rng)
            parents.append(np.uint16(winner))

        return parents

    # indentify surviving pipelines for next generation
    def SurvivialSelection(sel, pipelines: npt.NDArray) -> npt.NDArray[np.uint16]:
        assert pipelines.shape[0] > 0
        assert isinstance(pipelines[0], Pipeline)


        return

    # create the initial population of pipelines
    def InitializePopulation(self) -> None:

        for _ in range(self.pop_size):
            pipeline = Pipeline()
            pipeline.genrate_random_pipeline(self.branch_cnt_max)
            self.population = np.append(self.population, pipeline)
            pop = GenrateRandomPipeline(self.pop_size, self.branch_cnt_max)
