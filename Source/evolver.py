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
import sys, os
import ray
# from pipeline import Pipeline
import nsga_toolbox as nsga
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.model_selection import train_test_split
import time
from geno_hub import GenoHub

class Pipeline:
    def __init__(self) -> None:
        pass

@typechecked # for debugging purposes
class EA:
    def __init__(self, seed: np.uint16, pop_size: np.uint16, epi_cnt_max: np.uint16, cores: int,
                 mut_ran_p: np.float32 = np.float32(.45), mut_smt_p: np.float32 = np.float32(.45), mut_non_p: np.float32 = np.float32(.1),
                 smt_in_in_p: np.float32 = np.float32(.1), smt_in_out_p: np.float32 = np.float32(.45), smt_out_out_p: np.float32 = np.float32(.45)) -> None:
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
    def DataLoader(self, path: str, target_label: str = "y", split: float = 0.50) -> None:
        """
        Function to load data from a csv file into a pandas dataframe.
        We assume that the target label is 'y', unless otherwise specified.
        At the end of the function, we partition the data into training and validation sets.
        Additionally, we load the data into the ray object store and intialize hubs.

        Parameters:
        path: str
            Path to the file.
        target_label: str
            Name of the target label.
        """

        print('Loading data...')

        # save target variable
        self.target_label = target_label

        # check if the path is valid
        if os.path.isfile(path) == False:
            # load the data
            exit('Error: The path provided is not valid. Please provide a valid path to the data file.', -1)

        # get pandas dataframe snp names without loading all data
        self.snp_labels = pd.read_csv(path, nrows=0, index_col=0).columns.tolist()

        # check if the target label is valid
        if target_label not in self.snp_labels:
            exit('Error: The target label provided is not valid. Please provide a valid target label.', -1)

        # remove target label from snp labels
        self.snp_labels.remove(target_label)

        # load the data
        all_x = pd.read_csv(filepath_or_buffer=path, usecols=self.snp_labels)
        all_y = pd.read_csv(filepath_or_buffer=path, usecols=[self.target_label]).values.ravel()

        # check if the data was loaded correctly
        all_x, all_y = self.CheckDataset(all_x, all_y)
        print('X_data.shape:', all_x.shape)
        print('y_data.shape:', all_y.shape)
        print()

        # partition data based splits
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(all_x, all_y, test_size=split, random_state=self.seed)

        # check if the data was partitioned correctly
        self.X_train, self.y_train = self.CheckDataset(self.X_train, self.y_train)
        self.X_val, self.y_val = self.CheckDataset(self.X_val, self.y_val)

        # load data into ray object store
        self.X_train_id = ray.put(self.X_train)
        self.y_train_id = ray.put(self.y_train)
        self.X_val_id = ray.put(self.X_val)
        self.y_val_id = ray.put(self.y_val)

        print('X_train.shape:', self.X_train.shape)
        print('y_train.shape:', self.y_train.shape)
        print('X_val.shape:', self.X_val.shape)
        print('y_val.shape:', self.y_val.shape)
        print()

        print('Data loaded successfully.')

    # data checker to check for validity of dataset
    def CheckDataset(self, features, target):
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
        """
        # check for missing values
        if isinstance(features, pd.DataFrame):
                if features.isnull().values.any():
                    exit('Error: Input data contains missing values. Please impute the missing values before running.', -1)

        # check for target
        try:
            if target is not None:
                X, y = check_X_y(features, target, accept_sparse=True, dtype=None)
                # if self._imputed:
                #     return X, y
                # else:
                return features, target
            else:
                X = check_array(features, accept_sparse=True, dtype=None)
            #     if self._imputed:
            #         return X
            #     else:
                return features
        except (AssertionError, ValueError):
            raise ValueError(
                "Error: Input data is not in a valid format. Please confirm "
                "that the input data is scikit-learn compatible. For example, "
                "the features must be a 2-D array and target labels must be a "
                "1-D array."
            )

    def IntializeHubs(self, bin_size:int) -> None:
        """
        Function to initialize the hubs for the evolutionary algorithm.
        This function must be called after the data has been loaded.

        Parameters
        ----------
        hub_size: int
            Number of hubs to initialize.
        """
        # initialize the hubs
        self.hubs = GenoHub(snps=self.snp_labels, bin_size=np.uint16(bin_size))



    # Run NSGA-II for a specified number of generations
    # All functions below are EA specific

    def Evolve(self, gens: np.uint16) -> None:
        """
        Function to evovle pipelines using the NSGA-II algorithm for a user specified number of generations.
        We also take in the training and validation data to evaluate the pipelines.

        Parameters:
        gens: np.uint16
            Number of generations to run the algorithm.

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

    def InitializePopulation(self):
        pass

    def Evaluation(self, pop: List[Pipeline]) -> npt.NDArray[np.float32]:
        pass

    def ParentSelection(self, pop_scores: npt.NDArray[np.float32]) -> List[Pipeline]:
        pass

    def Reproduction(self, parents: List[Pipeline]) -> List[Pipeline]:
        pass

    def SurvivialSelection(self, scores: npt.NDArray[np.float32]) -> None:
        pass


def main():

    ea_config = {'seed': np.uint16(0),
                 'pop_size': np.uint16(100),
                 'epi_cnt_max': np.uint16(100),
                 'cores': 10,
                 'mut_ran_p':np.float32(.45),
                 'mut_smt_p': np.float32(.45),
                 'mut_non_p': np.float32(.1),
                 'smt_in_in_p': np.float32(.1),
                 'smt_in_out_p': np.float32(.45),
                 'smt_out_out_p': np.float32(.45)}

    ea = EA(**ea_config)
    ea.DataLoader('/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/pruned_ratdata_bmitail_onSNPnum.csv')
    ea.IntializeHubs(10)
    # time.sleep(20)

    ray.shutdown()



if __name__ == "__main__":
    main()