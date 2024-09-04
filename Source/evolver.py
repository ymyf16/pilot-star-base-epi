#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We use the NSGA-II algorithm to evolve pipelines.
# Pipelines consist of a set of epistatic interactions, feature selector, and a final regressor.
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List
import pandas as pd
import os
import ray
from pipeline import Pipeline
# import nsga_toolbox as nsga
from sklearn.model_selection import train_test_split
import time
from geno_hub import GenoHub
from typing import List, Tuple, Dict, Set
from epi_node import EpiNode
from epi_node import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node, EpiNode
from scikit_node import ScikitNode
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from reproduction import Reproduction
import numpy.typing as npt
import nsga_tool as nsga
import logging
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning

# snp name type
snp_name_t = np.str_
# snp hub position type
snp_hub_pos_t = np.uint32
# epi node list type
epi_node_list_t = List[EpiNode]
# probability type: needed to avoid rounding errors with probabilities
prob_t = np.float64

@ray.remote
def ray_lo_eval(x_train,
                y_train,
                x_val,
                y_val,
                snp1_name: snp_name_t,
                snp2_name: snp_name_t,
                snp1_pos: snp_hub_pos_t,
                snp2_pos: snp_hub_pos_t) -> Tuple[np.float32, np.str_, np.str_, np.str_]:
    # hold results
    best_epi = ''
    best_res = -1.0

    # holds all lo's we are going to evaluate
    epis = {np.str_('cartesian'): EpiCartesianNode,
            np.str_('xor'): EpiXORNode,
            np.str_('pager'): EpiPAGERNode,
            np.str_('rr'): EpiRRNode,
            np.str_('rd'): EpiRDNode,
            np.str_('t'): EpiTNode,
            np.str_('mod'): EpiModNode,
            np.str_('dd'): EpiDDNode,
            np.str_('m78'): EpiM78Node
            }

    # iterate over the epi node types and create sklearn pipeline
    for lo, epi in epis.items():
        steps = []
        # create the epi node
        epi_node = epi(name=lo, snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos)
        steps.append((lo, epi_node))

        # add random forrest regressor
        steps.append(('regressor', LinearRegression()))

        # create the pipeline
        skl_pipeline = SklearnPipeline(steps=steps)

        # Fit the pipeline
        skl_pipeline_fitted = skl_pipeline.fit(x_train, y_train)

        # get score
        r2 = skl_pipeline_fitted.score(x_val, y_val)

        # check if this is the best lo
        if r2 > best_res:
            best_res = r2
            best_epi = lo

    return np.float32(best_res), np.str_(best_epi), snp1_name, snp2_name

@ray.remote
def ray_eval_pipeline(x_train,
                      y_train,
                      x_val,
                      y_val,
                      epi_nodes: epi_node_list_t,
                      selector_node: ScikitNode,
                      root_node: ScikitNode,
                      pop_id: np.int16) -> Tuple[np.float32, np.uint16, np.int16]:
    # create the pipeline
    steps = []
    # combine the epi nodes into a sklearn union
    steps.append(('epi_union', FeatureUnion([(epi_node.name, epi_node) for epi_node in epi_nodes])))
    # add the selector node
    steps.append(('selector', selector_node))
    # add the root node
    steps.append(('root', root_node))
    # transform internal pipeline representation into sklearn pipeline with PipelineBuilder class
    pipeline = SklearnPipeline(steps=steps)

    # attempt to fit the pipeline
    try:
        # Fit the pipeline with warnings captured as exceptions
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            pipeline_fitted = pipeline.fit(x_train, y_train)
    except ConvergenceWarning as cw:
        logging.error(f"ConvergenceWarning while fitting model: {cw}")
        logging.error(f"selector_node: {selector_node.name}")
        logging.error(f"selector_node.params: {selector_node.params}")
        logging.error(f"epi_nodes: {len(epi_nodes)}")
        return -1.0, 0, 0
    except NotFittedError as nfe:
        logging.error(f"NotFittedError occurred: {nfe}")
        return -1.0, 0, 0
    except Exception as e:
        # Catch all other exceptions and log error with relevant context
        logging.error(f"Exception while fitting model: {e}")
        logging.error(f"selector_node: {selector_node.name}")
        logging.error(f"selector_node.params: {selector_node.params}")
        logging.error(f"epi_nodes: {len(epi_nodes)}")
        logging.error(f"Shapes -> X_train: {x_train.shape}, Y_train: {y_train.shape}")
        return -1.0, 0, 0

    try:
        r2_score = pipeline_fitted.score(x_val, y_val)
        feature_count = pipeline_fitted.named_steps['selector'].get_feature_count()
    except Exception as e:
        logging.error(f"Error while scoring or getting feature count: {e}")
        return -1.0, 0, 0

    # return the pipeline
    return np.float32(r2_score), np.uint16(feature_count), pop_id

@typechecked # for debugging purposes
class EA:
    def __init__(self,
                 seed: np.uint16,
                 pop_size: np.uint16,
                 epi_cnt_max: np.uint16,
                 cores: int,
                 mut_prob: prob_t = prob_t(.5),
                 cross_prob: prob_t = prob_t(.5),
                 mut_selector_p: prob_t = prob_t(.5),
                 mut_regressor_p: prob_t = prob_t(.5),
                 mut_ran_p: prob_t = prob_t(.45),
                 mut_non_p: prob_t = prob_t(.1),
                 mut_smt_p: prob_t = prob_t(.45),
                 smt_in_in_p: prob_t = prob_t(.1),
                 smt_in_out_p: prob_t = prob_t(.45),
                 smt_out_out_p: prob_t = prob_t(.45)) -> None:
        """
        Main class for the evolutionary algorithm.

        Parameters:
        seed: np.uint16
            Seed for the random number generator.
        pop_size: np.uint16
            Population size.
        epi_cnt_max: np.uint16
            Maximum number of epistatic interactions (nodes).
        mut_ran_p: prob_t
            Probability for random mutation.
        mut_smt_p: prob_t
            Probability for smart mutation.
        mut_non_p: prob_t
            Probability for no mutation.
        smt_in_in_p: prob_t
            Probability for smart mutation, new snp comes from the same chromosome and assigned bin.
        smt_in_out_p: prob_t
            Probability for smart mutation, new snp comes from the same chromosome but different bin.
        smt_out_out_p: prob_t
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
        self.repoduction = Reproduction(mut_prob=mut_prob,
                                 cross_prob=cross_prob,
                                 mut_selector_p=mut_selector_p,
                                 mut_regressor_p=mut_regressor_p,
                                 mut_ran_p=mut_ran_p,
                                 mut_non_p=mut_non_p,
                                 mut_smt_p=mut_smt_p,
                                 smt_in_in_p=smt_in_in_p,
                                 smt_in_out_p=smt_in_out_p,
                                 smt_out_out_p=smt_out_out_p)

        # Initialize Ray: Will have to specify when running on hpc
        context = ray.init(num_cpus=cores, include_dashboard=True)
        print()

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
                return features, target
            else:
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

    # will evolve a population of pipelines for a specified number of generations
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

        # print out the population for checks
        # self.print_population()

        # run the algorithm for the specified number of generations
        for g in range(gens):
            # make sure we have the correct number of pipelines
            assert(0 < len(self.population) <= self.pop_size)

            print('Generation:', g)

            # how many extra pipeline offspring do we need to fill up considered solutions
            extra_offspring = self.pop_size - len(self.population)
            # get order of mutation/crossover to do with the extra offspring
            var_order, parent_cnt = self.repoduction.variation_order(self.rng, np.uint16(extra_offspring + self.pop_size))
            # get the parent scores by position
            parent_ids = self.parent_selection(np.array([[pipeline.get_trait_r2(), pipeline.get_trait_feature_cnt()] for pipeline in self.population], dtype=np.float32), parent_cnt)

            # generate offspring
            offspring = self.repoduction.produce_offspring(rng = self.rng,
                                                           hub = self.hubs,
                                                           offspring_cnt=np.uint16(extra_offspring + self.pop_size),
                                                           parent_ids=parent_ids,
                                                           population=self.population,
                                                           order=var_order,
                                                           seed=int(self.seed))
            # make sure we have the correct number of competing solutions
            assert len(offspring) + len(self.population) == 2 * self.pop_size

            # process offspring: evaluation interactions and remove bad interactions
            offspring = self.process_offspring(offspring)

            # evaluate the offspring
            offspring_scores = self.evaluation(offspring)
            assert(len(offspring_scores) == len(offspring))

            # update offspring with scores
            for score, pipeline in zip(offspring_scores, offspring):
                pipeline.set_traits([score[0], np.uint16(score[1])])

            # make sure the correct number of solutions are competing prior to negative r2 filtering
            assert len(offspring_scores) + len(self.population) == 2 * self.pop_size

            # survival selection
            self.population = self.survival_selection(self.population, offspring, offspring_scores)

            # make sure we have the correct number of pipelines
            assert len(self.population) == self.pop_size

    # survival selection
    def survival_selection(self,
                           current_pop: List[Pipeline],
                           offspring: List[Pipeline],
                           offspring_scores: List[Tuple[np.float32, np.uint16, np.int16]]) -> List[Pipeline]:
        """
        Function to select the survivors from the current population and offspring.

        Parameters:
        current_pop: List[Pipeline]
            Current population of pipelines.
        offspring: List[Pipeline]
            Offspring population of pipelines.
        """
        # subset the offspring to only include pipelines with positive r2 scores
        offspring = [offspring[offspring_scores[i][2]] for i in range(len(offspring_scores)) if offspring_scores[i][0] > 0.0]
        # subset the scores to only include pipelines with positive r2 scores: [(r2, complexity),...]
        positive_offspring_scores = [(offspring_scores[i][0], offspring_scores[i][1]) for i in range(len(offspring_scores)) if offspring_scores[i][0] > 0.0]
        # get all scores from the current population
        population_scores = [(pipeline.get_trait_r2(), pipeline.get_trait_feature_cnt()) for pipeline in self.population]
        # make sure all population scores are positive
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in self.population)

        # combine both the population and offspring scores
        all_scores = np.array(np.concatenate((population_scores, positive_offspring_scores), axis=0), dtype=np.float32)
        assert len(all_scores) == len(self.population) + len(positive_offspring_scores)
        assert len(all_scores) >= self.pop_size

        # get the fronts and rank
        fronts, ranks = nsga.non_dominated_sorting(all_scores)

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(all_scores, np.int32(2))

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
        # make sure that the number of survivors is correct
        assert len(survivor_ids) == self.pop_size

        # combine the population and offspring
        candidates = self.population + offspring

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(candidates)
            new_pop.append(candidates[i])

        return new_pop

    # initialize the starting population
    def initialize_population(self) -> None:
        """
        Initialize the population of pipelines with their set of epistatic interactions.
        We start by creating (2 * pop_size) random set of interactions and find their best lo from the 9 we use.
        After, we remove bad interactions (negative r2) for a given set and create pipelines from the good interactions.
        We then evaluate the pipelines and keep only the pipelines with positive r2 scores.

        If the number of pipelines with positive r2 scores is less than the population size, we keep the same population.
        If the number of pipelines with positive r2 scores is greater than the population size, we use NSGA-II to get the pareto front.
        """
        # will hold sets of interactions for each pipeline in the population
        pop_epi_interactions = []
        # will hold the unseen interactions -- interactions not found in the Genohub
        unseen_interactions = set()

        # create the initial population
        # we create double the population size to account for bad interactions
        # which are are extremely common given the nature of epistatic interactions
        for _ in range(self.pop_size * 2):
            # how many epi nodes to create
            # lower bound will be half of what the user specified
            # upper bound will be the actual user specified mas
            epi_cnt = self.rng.integers(int(self.epi_cnt_max * 0.5), self.epi_cnt_max)

            # holds all interactions we are doing
            # set to make sure we don't have duplicates
            interactions = set()

            # while we have not reached the max number of epi interactions
            while len(interactions) < epi_cnt:
                # get random snp1 and snp2 and add to interactions
                snp1, snp2 = self.hubs.get_ran_interaction(self.rng)
                # add interaction to the set
                interactions.add((snp1, snp2))

                # check if we have seen this interaction before
                if self.hubs.is_interaction_in_hub(snp1, snp2) == False:
                    unseen_interactions.add((snp1, snp2))

            # add to the population
            pop_epi_interactions.append(interactions)

        # make sure we have the correct number of interactions
        assert len(pop_epi_interactions) == 2 * self.pop_size

        # evaluate all unseen interactions
        self.evaluate_unseen_interactions(unseen_interactions)

        # remove bad interactions for each pipeline's set of interactions
        for interactions in pop_epi_interactions:
            good_interactions = self.remove_bad_interactions(interactions)

            # make sure we have the correct number of good interactions
            assert len(good_interactions) <= len(interactions)

            # create pipeline and add to the population
            self.population.append(self.repoduction.generate_random_pipeline(self.rng, good_interactions, int(self.seed)))

        # make sure we have the correct number of pipelines
        assert len(self.population) ==  2 * self.pop_size

        # evaluate the initial population
        scores =  self.evaluation(self.population)

        # subset the population to only include pipelines with positive r2 scores
        self.population = [self.population[scores[i][2]] for i in range(len(scores)) if scores[i][0] > 0.0]
        # subset the scores to only include pipelines with positive r2 scores: [(r2, complexity),...]
        positive_scores = np.array([(scores[i][0], scores[i][1]) for i in range(len(scores)) if scores[i][0] > 0.0], dtype=np.float32)
        # make sure that the size of scores matches the population size
        assert len(positive_scores) == len(self.population)

        # assign traits to each pipeline
        for score, pipeline in zip(positive_scores, self.population):
            pipeline.set_traits([score[0], np.uint16(score[1])])

        # if size positive_scores is less than the population size, we keep the same population
        if len(positive_scores) < self.pop_size:
            return
        # else we use nsga to get the pareto front from all the pipelines in the population
        else:
            # get the fronts and rank
            fronts, ranks = nsga.non_dominated_sorting(positive_scores)
            # make sure that the number of fronts is correct
            assert sum([len(f) for f in fronts]) == len(ranks)

            # get crowding distance for each solution
            crowding_distance = nsga.crowding_distance(positive_scores, np.int32(2))

            # truncate the population to the population size with nsga ii
            survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
            self.population = [self.population[i] for i in survivor_ids]
            return

    # evaluate all unseed interactions and update the GenoHub
    def evaluate_unseen_interactions(self, unseen_interactions: Set[Tuple]) -> None:
        """
        Function to evaluate all unseen interactions.
        We will evaluate all unseen interactions and add them to the GenoHub.

        Parameters:
        unseen_interactions: Set[Tuple]
            Set of unseen interactions to evaluate.
        """
        # collect all parallel jobs
        ray_jobs = []

        # collect all ray jobs for evaluation
        for snp1_name, snp2_name in unseen_interactions:
            ray_jobs.append(ray_lo_eval.remote(x_train = self.X_train_id,
                                                y_train = self.y_train_id,
                                                x_val = self.X_val_id,
                                                y_val = self.y_val_id,
                                                snp1_name = snp1_name,
                                                snp2_name = snp2_name,
                                                snp1_pos = self.hubs.get_snp_pos(snp1_name),
                                                snp2_pos = self.hubs.get_snp_pos(snp2_name)))
        # run all ray jobs
        ray_results = ray.get(ray_jobs)

        # make sure we have the correct number of results
        assert len(ray_results) == len(unseen_interactions)

        # add all results to the GenoHub
        for r in ray_results:
            r2, lo, snp1_name, snp2_name = r
            self.hubs.update_epi_n_snp_hub(snp1_name, snp2_name, r2, lo)

    # remove bad interactions for a given set of interactions
    def remove_bad_interactions(self, interactions: Set[Tuple[snp_name_t,snp_name_t]]) -> Set[Tuple[snp_name_t,snp_name_t]]:
        """
        Function to remove bad interactions for a given set of interactions

        Parameters:
        interactions: List[Tuple[snp_name_t,snp_name_t]]
            List of epistatic interactions represented by Tuples.
        """
        # iterate through each interaction
        good_interactions = set()
        for snp1_name, snp2_name in interactions:
            # check if r2 is positive
            if self.hubs.get_interaction_res(snp1_name, snp2_name) > np.float32(0.0):
                # add to good interactions
                good_interactions.add((snp1_name, snp2_name))

        # return the good interactions
        return good_interactions

    # print the population
    def print_population(self) -> None:
        """
        Function to print the population.
        """
        print('Population:')
        for p in self.population:
            p.print_pipeline()

    # evaluate the population                                   # r2 , feature count, pop_id
    def evaluation(self, pop: List[Pipeline]) -> List[Tuple[np.float32, np.uint16, np.int16]]:
        # collect all parallel jobs
        ray_jobs = []
        # go through each pipeline in the population and evaluate
        for i, pipeline in enumerate(pop):
            # add ray job
            ray_jobs.append(ray_eval_pipeline.remote(self.X_train_id,
                                                     self.y_train_id,
                                                     self.X_val_id,
                                                     self.y_val_id,
                                                     self.construct_epi_nodes(pipeline.get_epi_pairs()),
                                                     pipeline.get_selector_node(),
                                                     pipeline.get_root_node(),
                                                     np.int16(i)))
        # run all ray jobs
        results = ray.get(ray_jobs)

        return results

    # construct epi_nodes for a pipeline's set of epi_pairs
    def construct_epi_nodes(self, epi_pairs: Set[Tuple[snp_name_t,snp_name_t]]) -> epi_node_list_t:
        """
        Function to construct epi nodes for a pipeline's set of epi_pairs.

        Parameters:
        epi_pairs: Set[Tuple[snp_name_t,snp_name_t]]
            Set of epistatic interactions represented by Tuples.
        """
        epi_nodes = []
        id = 0
        for snp1_name, snp2_name in epi_pairs:
            # get the epi node
            epi_lo = self.hubs.get_interaction_lo(snp1_name, snp2_name)
            # get each snps position in the hub
            snp1_pos = self.hubs.get_snp_pos(snp1_name)
            snp2_pos = self.hubs.get_snp_pos(snp2_name)

            if epi_lo == np.str_('cartesian'):
                epi_nodes.append(EpiCartesianNode(name=f"EpiCartesianNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('xor'):
                epi_nodes.append(EpiXORNode(name=f"EpiXORNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('rr'):
                epi_nodes.append(EpiRRNode(name=f"EpiRRNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('rd'):
                epi_nodes.append(EpiRDNode(name=f"EpiRDNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('t'):
                epi_nodes.append(EpiTNode(name=f"EpiTNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('mod'):
                epi_nodes.append(EpiModNode(name=f"EpiModNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('dd'):
                epi_nodes.append(EpiDDNode(name=f"EpiDDNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('m78'):
                epi_nodes.append(EpiM78Node(name=f"EpiM78Node_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            elif epi_lo == np.str_('pager'):
                epi_nodes.append(EpiPAGERNode(name=f"EpiPAGERNode_{id}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos))
            else:
                exit('Error: The interaction lo is not valid. Please provide a valid interaction lo.', -1)

            id += 1
        # return the list of epi nodes
        return epi_nodes

    # parent selection
    def parent_selection(self, scores: npt.NDArray[np.float32],
                         parent_cnt: np.uint16) -> List[np.uint16]:
        """
        Function to return a specified number of parent ids based on the scores of the pipelines.

        Parameters:
        scores: List[Tuple[np.float32, np.uint16, np.int16]] # r2 , feature count, pop_id
            Scores of the pipelines (scores: np.array([r2,complexity])).
        """
        # make sure that the size of scores matches the population size
        assert len(scores) == len(self.population)

        # will hold the parent ids
        parent_ids = []

        # get the fronts and rank
        fronts, ranks = nsga.non_dominated_sorting(scores)
        # make sure that the number of fronts is correct
        assert sum([len(f) for f in fronts]) == len(ranks)

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(scores, np.int32(2))

        # get parent_cnt number of parents
        for _ in range(parent_cnt):
            parent_ids.append(nsga.non_dominated_binary_tournament(rng=self.rng, ranks=ranks, distances=crowding_distance))
        # make sure that the number of parents is correct
        assert len(parent_ids) == parent_cnt

        return parent_ids

    # process offspring
    def process_offspring(self, pipelines: List[Pipeline]) -> List[Pipeline]:

        # get unseen interactions
        unseen_interactions = self.get_unseen_interactions(pipelines)

        # evaluate all unseen interactions
        self.evaluate_unseen_interactions(unseen_interactions)

        # remove bad interactions for each pipeline's set of interactions
        updated_pipelines = []
        for pipeline in pipelines:
            good_interactions = self.remove_bad_interactions(pipeline.get_epi_pairs())
            updated_pipelines.append(Pipeline(good_interactions, pipeline.get_selector_node(), pipeline.get_root_node(), []))

        return updated_pipelines

    # get unseen intreactions from offspring pipelines
    def get_unseen_interactions(self, pipelines: List[Pipeline]) -> Set[Tuple[snp_name_t,snp_name_t]]:
        """
        Function to get unseen interactions from offspring pipelines.

        Parameters:
        pipelines: List[Pipeline]
            List of pipelines.
        """
        unseen_interactions = set()
        for pipeline in pipelines:
            for snp1_name, snp2_name in pipeline.get_epi_pairs():
                if self.hubs.is_interaction_in_hub(snp1_name, snp2_name) == False:
                    unseen_interactions.add((snp1_name, snp2_name))

        return unseen_interactions

def main():
    # set experiemnt configurations
    ea_config = {'seed': np.uint16(0),
                 'pop_size': np.uint16(300),
                 'epi_cnt_max': np.uint16(300),
                 'cores': 10,
                 'mut_selector_p': np.float64(1.0),
                 'mut_regressor_p': np.float64(.5),
                 'mut_ran_p':np.float64(.45),
                 'mut_smt_p': np.float64(.45),
                 'mut_non_p': np.float64(.1),
                 'smt_in_in_p': np.float64(.10),
                 'smt_in_out_p': np.float64(.450),
                 'smt_out_out_p': np.float64(.450),
                 'mut_prob': np.float64(.5),
                 'cross_prob': np.float64(.5)}

    ea = EA(**ea_config)
    # need to update the path to the data file
    data_dir = '/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/pruned_ratdata_bmitail_onSNPnum.csv'
    # data_dir = '/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/18qtl_pruned_BMIres.csv'
    ea.data_loader(data_dir)
    ea.initialize_hubs(20)
    ea.evolve(100)


    ray.shutdown()

if __name__ == "__main__":
    main()