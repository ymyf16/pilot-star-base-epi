#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We use the NSGA-II algorithm to evolve pipelines.
# Pipelines consist of a set of epistatic interactions, feature selector, and a final regressor.
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
import ray
from pipeline import Pipeline
import nsga_toolbox as nsga
from sklearn.utils import check_X_y, check_array, check_consistent_length


@typechecked # for debugging purposes
class EA:
    def __init__(self, seed: np.uint16, pop_size: np.uint16, epi_cnt_max: np.uint16, cores: np.uint8,
                 mut_ran_p: np.float32, mut_smt_p: np.float32, mut_non_p: np.float32,
                 smt_in_in_p: np.float32, smt_in_out_p: np.float32, smt_out_out_p: np.float32) -> None:
        """
        Main class for the evolutionary algorithm.

        Parameters:
        seed: np.uint16
            Seed for the random number generator.
        pop_size: np.uint16
            Population size.
        epi_cnt_max: np.uint16
            Maximum number of epistatic interactions (nodes).
        mut_ran_p: np.float32
            Probability for random mutation.
        mut_smt_p: np.float32
            Probability for smart mutation.
        mut_non_p: np.float32
            Probability for no mutation.
        smt_in_in_p: np.float32
            Probability for smart mutation, new snp comes from the same chromosome and assigned bin.
        smt_in_out_p: np.float32
            Probability for smart mutation, new snp comes from the same chromosome but different bin.
        smt_out_out_p: np.float32
            Probability for smart mutation, new snp comes from a different chromosome (different bin by definition).
        """

        # arguments needed to run
        self.seed = seed
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed) # random number generator to be passed to all other stocastic functions
        self.epi_cnt_max = epi_cnt_max
        self.mut_ran_p = mut_ran_p
        self.mut_smt_p = mut_smt_p
        self.mut_non_p = mut_non_p
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p

        # Initialize Ray: Will have to specify when running on hpc
        context = ray.init(num_cpus=cores, include_dashboard=True)
        print('dashboard:', context.dashboard_url)

    # data loader
    def DataLoader(self, path: str) -> pd.DataFrame:
        """
        Function to load data from a file.

        Parameters:
        path: str
            Path to the file.
        """

        pass

    # data checker
    # Function to check for validity of dataset
    def _check_dataset(self, features, target, sample_weight=None):
        """
        Check if a dataset has a valid feature set and labels.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples} or None
            List of class labels for prediction
        sample_weight: array-like {n_samples} (optional)
            List of weights indicating relative importance
        Returns
        -------
        (features, target)
        """
        # Check sample_weight
        if sample_weight is not None:
            try:
                sample_weight = np.array(sample_weight, dtype=np.float32)
            except ValueError as e:
                raise ValueError(
                    "sample_weight could not be converted to numpy.float32: %s" % e
                )
            if np.any(np.isnan(sample_weight)):
                raise ValueError("sample_weight contained NaN values.")
            try:
                check_consistent_length(sample_weight, target)
            except ValueError as e:
                raise ValueError(
                    "sample_weight dimensions did not match target: %s" % e
                )
        # check for features
        if isinstance(features, np.ndarray):
                if np.any(np.isnan(features)):
                    self._imputed = True
        elif isinstance(features, pd.DataFrame):
                if features.isnull().values.any():
                    self._imputed = True
        if self._imputed:
                features = self._impute_values(features)
        # check for target
        try:
            if target is not None:
                X, y = check_X_y(features, target, accept_sparse=True, dtype=None)
                if self._imputed:
                    return X, y
                else:
                    return features, target
            else:
                X = check_array(features, accept_sparse=True, dtype=None)
                if self._imputed:
                    return X
                else:
                    return features
        except (AssertionError, ValueError):
            raise ValueError(
                "Error: Input data is not in a valid format. Please confirm "
                "that the input data is scikit-learn compatible. For example, "
                "the features must be a 2-D array and target labels must be a "
                "1-D array."
            )

    # Run NSGA-II for a specified number of generations
    def Evolve(self, gens: np.uint16, X_train, y_train, X_validate, y_validate) -> None:
        """
        Function to evovle pipelines using the NSGA-II algorithm for a user specified number of generations.
        We also take in the training and validation data to evaluate the pipelines.

        Parameters:
        g

        """

        # create the initial population
        self.InitializePopulation()

        # for each generation
        for gen in range(gens):

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