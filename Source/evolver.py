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
# import nsga_toolbox as nsga
from sklearn.model_selection import train_test_split
import time
from geno_hub import GenoHub
# from pipeline_builder import PipelineBuilder
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict, Set
# import reproduction_toolbox as repo_tool
# from epi_nodes import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node, EpiNode
# from scikit_nodes import ScikitNode, VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree
# from scikit_nodes import SequentialFeatureSelectorNode, LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode
# from scikit_nodes import ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode
from sklearn.pipeline import Pipeline as SklearnPipeline

# import linear regression from scikit-learn
from sklearn.linear_model import LinearRegression


@typechecked # for debugging purposes
class EA:
    def __init__(self,
                 seed: np.uint16,
                 pop_size: np.uint16,
                 epi_cnt_max: np.uint16,
                 cores: int,
                 mut_prob: np.float32 = np.float32(.5),
                 cross_prob: np.float32 = np.float32(.5),
                 mut_selector_p: np.float32 = np.float32(.5),
                 mut_regressor_p: np.float32 = np.float32(.5),
                 mut_ran_p: np.float32 = np.float32(.45),
                 mut_non_p: np.float32 = np.float32(.1),
                 mut_smt_p: np.float32 = np.float32(.45),
                 smt_in_in_p: np.float32 = np.float32(.1),
                 smt_in_out_p: np.float32 = np.float32(.45),
                 smt_out_out_p: np.float32 = np.float32(.45)) -> None:
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
        self.mut_non_p = mut_non_p
        self.mut_smt_p = mut_smt_p
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p
        self.population = [] # will hold all the pipelines

        # Initialize Ray: Will have to specify when running on hpc
        context = ray.init(num_cpus=cores, include_dashboard=True)
        print('dashboard:', context.dashboard_url)

    # data loader
    def data_loader(self, path: str, target_label: str = "y", split: float = 0.50) -> None:
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


        # check if the path is valid
        if os.path.isfile(path) == False:
            # load the data
            exit('Error: The path provided is not valid. Please provide a valid path to the data file.', -1)

        # get pandas dataframe snp names without loading all data
        self.snp_labels = pd.read_csv(path, nrows=0).columns.tolist()

        # check if the target label is valid
        if target_label not in self.snp_labels:
            exit('Error: The target label provided is not valid. Please provide a valid target label.', -1)

        # remove target label from snp labels
        self.snp_labels.remove(target_label)

        # convert python strings into numpy strings
        self.snp_labels = np.array(self.snp_labels, dtype=np.str_)
        self.target_label = np.str_(target_label)

        # load the data
        all_x = pd.read_csv(filepath_or_buffer=path, usecols=self.snp_labels)
        all_y = pd.read_csv(filepath_or_buffer=path, usecols=[self.target_label]).values.ravel()

        # check if the data was loaded correctly
        all_x, all_y = self.check_dataset(all_x, all_y)
        print('X_data.shape:', all_x.shape)
        print('y_data.shape:', all_y.shape)
        print()

        # partition data based splits
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(all_x, all_y, test_size=split, random_state=self.seed)

        # check if the data was partitioned correctly
        self.X_train, self.y_train = self.check_dataset(self.X_train, self.y_train)
        self.X_val, self.y_val = self.check_dataset(self.X_val, self.y_val)

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
    def check_dataset(self, features, target):
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
                # X, y = check_X_y(features, target, accept_sparse=True, dtype=None)
                # if self._imputed:
                #     return X, y
                # else:
                return features, target
            else:
                # X = check_array(features, accept_sparse=True, dtype=None)
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

    def initialize_hubs(self, bin_size:int) -> None:
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

    def evolve(self, gens: int) -> None:
        """
        Function to evovle pipelines using the NSGA-II algorithm for a user specified number of generations.
        We also take in the training and validation data to evaluate the pipelines.

        Parameters:
        gens: np.uint16
            Number of generations to run the algorithm.

        """

        # create the initial population
        self.initialize_population()


    def initialize_population(self) -> None:
        """
        Function to initialize the population of pipelines.
        We start by creating a random set of interactions and determine their best lo.
        We then create a epi_node for each interaction and lo.
        Once we have all epi_nodes, we create a pipeline with a random selector and regressor.
        """
        pop_epi_interactions = []

        # create the initial population
        for _ in range(self.pop_size):
            # how many epi nodes to create
            # lower bound will be half of what the user specified
            # upper bound will be the actual user specified mas
            epi_cnt = self.rng.integers(int(self.epi_cnt_max * 0.5), self.epi_cnt_max)

            # holds all interactions we are doing
            # set to make sure we don't have duplicates
            interactions = set()

            # while we have not reached the max number of epi nodes
            while len(interactions) < epi_cnt:
                # get random snp1 and snp2 and add to interactions
                interactions.add(self.hubs.get_ran_interaction(self.rng))

            # add to the population
            pop_epi_interactions.append(interactions)

        # find all unique interactions
        # set() = {(snp1_name, snp2_name),...}
        unique_interactions = set()
        for interactions in pop_epi_interactions:
            unique_interactions.update(interactions)



def main():
    # set experiemnt configurations
    ea_config = {'seed': np.uint16(42),
                 'pop_size': np.uint16(100),
                 'epi_cnt_max': np.uint16(100),
                 'cores': 10,
                 'mut_selector_p': np.float32(1.0),
                 'mut_regressor_p': np.float32(.5),
                 'mut_ran_p':np.float32(.45),
                 'mut_smt_p': np.float32(.45),
                 'mut_non_p': np.float32(.1),
                 'smt_in_in_p': np.float32(.1),
                 'smt_in_out_p': np.float32(.45),
                 'smt_out_out_p': np.float32(.45),
                 'mut_prob': np.float32(.5),
                 'cross_prob': np.float32(.5)}

    ea = EA(**ea_config)
    # need to update the path to the data file
    data_dir = '/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/pruned_ratdata_bmitail_onSNPnum.csv'
    # data_dir = '/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/18qtl_pruned_BMIres.csv'
    ea.data_loader(data_dir)
    ea.initialize_hubs(20)
    ea.evolve(1)


    ray.shutdown()

if __name__ == "__main__":
    main()