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
from pipeline_uni import Pipeline
# import nsga_toolbox as nsga
from sklearn.model_selection import train_test_split
import time
from geno_hub_uni import GenoHub
from typing import List, Tuple, Dict, Set
from epi_node import EpiNode
from epi_node import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node, EpiNode

from uni_node import UniNode
from uni_node import UniDominantNode, UniRecessiveNode, UniHeterosisNode, UniUnderDominantNode, UniSubadditiveNode, UniSuperadditiveNode, UniPAGERNode

from scikit_node import ScikitNode
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from reproduction_uni import Reproduction
import numpy.typing as npt
import nsga_tool as nsga
import logging
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning
import matplotlib.pyplot as plt


# snp name type
snp_name_t = np.str_
# snp hub position type
snp_hub_pos_t = np.uint32
# epi node list type
epi_node_list_t = List[EpiNode]
#YF uni node list type
uni_node_list_t = List[UniNode]
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

#YF? Same feature union or separate ones?
@ray.remote
def ray_eval_pipeline(x_train,
                      y_train,
                      x_val,
                      y_val,
                      epi_nodes: epi_node_list_t,
                      uni_nodes: uni_node_list_t,
                      selector_node: ScikitNode,
                      root_node: ScikitNode,
                      pop_id: np.int16) -> Tuple[np.float32, np.uint16, np.int16]:
    # create the pipeline
    steps = []
    #YFupdate epi & uni nodes into one sklearn union
    steps.append(('feature_union', FeatureUnion([(epi_node.name, epi_node) for epi_node in epi_nodes] + [(uni_node.name, uni_node) for uni_node in uni_nodes])))
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
        return np.float32(-1.0), np.uint16(0), pop_id
    except NotFittedError as nfe:
        logging.error(f"NotFittedError occurred: {nfe}")
        return np.float32(-1.0), np.uint16(0), pop_id
    except Exception as e:
        # Catch all other exceptions and log error with relevant context
        logging.error(f"Exception while fitting model: {e}")
        logging.error(f"selector_node: {selector_node.name}")
        logging.error(f"selector_node.params: {selector_node.params}")
        logging.error(f"epi_nodes: {len(epi_nodes)}")
        logging.error(f"Shapes -> X_train: {x_train.shape}, Y_train: {y_train.shape}")
        return np.float32(-1.0), np.uint16(0), pop_id

    try:
        r2_score = pipeline_fitted.score(x_val, y_val)
        feature_count = pipeline_fitted.named_steps['selector'].get_feature_count()
    except Exception as e:
        logging.error(f"Error while scoring or getting feature count: {e}")
        return np.float32(-1.0), np.uint16(0), pop_id

    # return the pipeline
    return np.float32(r2_score), np.uint16(feature_count), pop_id

####YF
@ray.remote
def ray_uni_eval(x_train,
                y_train,
                x_val,
                y_val,
                snp_name: snp_name_t,
                snp_pos: snp_hub_pos_t) -> Tuple[np.float32, np.str_, np.str_]:
    # hold results
    best_uni = ''
    best_res = -1.0

    # holds all lo's we are going to evaluate
    unis = {np.str_('dominant'): UniDominantNode,
            np.str_('recessive'): UniRecessiveNode,
            np.str_('heterosis'): UniHeterosisNode,
            np.str_('underdominant'): UniUnderDominantNode,
            np.str_('subadd'): UniSubadditiveNode,
            np.str_('superadd'): UniSuperadditiveNode,
            np.str_('pager'): UniPAGERNode,
            }

    # iterate over the uni node types and create sklearn pipeline
    for lo, uni in unis.items():
        steps = []
        # create the epi node
        uni_node = uni(name=lo, snp_name=snp_name, snp_pos=snp_pos)
        steps.append((lo, uni_node))

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
            best_uni = lo

    return np.float32(best_res), np.str_(best_uni), snp_name

    

@typechecked # for debugging purposes
class EA:
    def __init__(self,
                 seed: np.uint16,
                 pop_size: np.uint16,
                 epi_cnt_max: np.uint16,
                 epi_cnt_min: np.uint16,
                 uni_cnt_max: np.uint16,
                 uni_cnt_min: np.uint16,
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
                 smt_out_out_p: prob_t = prob_t(.45),
                 num_add_interactions: np.uint16 = np.uint16(10),
                 num_del_interactions: np.uint16 = np.uint16(10)) -> None:
        """
        Main class for the evolutionary algorithm of .

        Parameters:
        seed: np.uint16
            Seed for the random number generator.
        pop_size: np.uint16
            Population size.
        epi_cnt_max: np.uint16
            Maximum number of epistatic interactions (nodes).

        #YF
        uni_cnt_max: np.uint16
            Maximum number of univariate snps (nodes).

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
        mut_prob: prob_t
            Probability for mutation ocurring.
        cross_prob: prob_t
            Probability for crossover ocurring.
        num_add_interactions: np.uint16
            Number of interactions to add within a pipeline.
        num_del_interactions: np.uint16
            Number of interactions to delete within a pipeline.
        """

        # arguments needed to run
        self.seed = seed
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed) # random number generator to be passed to all other stocastic functions
        self.epi_cnt_max = epi_cnt_max
        self.epi_cnt_min = epi_cnt_min

        #YF added arguments for uni
        self.uni_cnt_max = uni_cnt_max
        self.uni_cnt_min = uni_cnt_min

        self.mut_ran_p = mut_ran_p
        self.mut_non_p = mut_non_p
        self.mut_smt_p = mut_smt_p
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p
        self.num_add_interactions = num_add_interactions
        self.num_del_interactions = num_del_interactions
        self.population = [] # will hold all the pipelines
        self.repoduction = Reproduction(epi_cnt_max=epi_cnt_max,
                                        epi_cnt_min=epi_cnt_min,
                                        uni_cnt_max=uni_cnt_max,
                                        uni_cnt_min=uni_cnt_min,
                                        mut_prob=mut_prob,
                                        cross_prob=cross_prob,
                                        mut_selector_p=mut_selector_p,
                                        mut_regressor_p=mut_regressor_p,
                                        mut_ran_p=mut_ran_p,
                                        mut_non_p=mut_non_p,
                                        mut_smt_p=mut_smt_p,
                                        smt_in_in_p=smt_in_in_p,
                                        smt_in_out_p=smt_in_out_p,
                                        smt_out_out_p=smt_out_out_p,
                                        num_add_interactions=num_add_interactions,
                                        num_del_interactions=num_del_interactions)

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
        print('Path:', path)

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

        #YF change all the 1 in all_x to 0.5, all 2 to 1 in all_x - changing the additive encoding from 0,1,2 to 0,0.5,1
        all_x = all_x.replace(1, 0.5)
        all_x = all_x.replace(2, 1)

        # checking the encoding
        print("Genotype data: ", all_x)
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
        print()
        print('X_val.shape:', self.X_val.shape)
        print('y_val.shape:', self.y_val.shape)
        print()
        print('Data loaded successfully.')
        print()
        return

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
        print("Initialized population")

        # print out the population for checks
        # self.print_population()

        # run the algorithm for the specified number of generations
        for g in range(gens):
            # make sure we have the correct number of pipelines
            assert(0 < len(self.population) <= self.pop_size)

            print('Generation:', g)

            # self.plot_pareto_front()

            # how many extra pipeline offspring do we need to fill up considered solutions
            print("calculating extra offspring")
            extra_offspring = self.pop_size - len(self.population)
            # get order of mutation/crossover to do with the extra offspring
            var_order, parent_cnt = self.repoduction.variation_order(self.rng, np.uint16(extra_offspring + self.pop_size))
            # get the parent scores by position
            parent_ids = self.parent_selection(np.array([[pipeline.get_trait_r2(), pipeline.get_trait_feature_cnt()] for pipeline in self.population], dtype=np.float32), parent_cnt)

            print("generating offspring")
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
            print("evaluating offspring")
            self.evaluation(offspring)
            print("offspring eval done")


            # subset the population to only include pipelines with positive r2 scores
            off = []
            for pipeline in offspring:
                if pipeline.get_trait_r2() > 0.0:
                    off.append(pipeline)
            offspring = off

            offspring_scores = self.get_pipeline_scores(offspring)
            assert(len(offspring_scores) == len(offspring))

            # make sure the correct number of solutions are competing prior to negative r2 filtering
            assert len(offspring_scores) + len(self.population) <= 2 * self.pop_size

            # survival selection
            self.population = self.survival_selection(self.population,
                                                      self.get_pipeline_scores(self.population),
                                                      offspring,
                                                      offspring_scores)

            # make sure we have the correct number of pipelines
            assert len(self.population) == self.pop_size
        
    #YFTODO call save_epihub thing 
    # save the epi_hub and snp_hub to a file
    def save_hubs(self, epi_file: str, snp_file: str) -> None:
        # Save epi hub with headers
        epi_data = []
        for k, v in self.hubs.epi_hub.hub.items(): #YF2
            epi_data.append([k[0], k[1], v[0], v[1]])

        # Sort epi_data by the third column (R2)
        epi_data.sort(key=lambda x: x[2], reverse=True)  # reverse=True for descending order

        # Write epi hub to file
        with open(epi_file, 'w') as f:
            # Write the headers for the epi_file
            f.write("SNP1,SNP2,R2,LO\n")
            for row in epi_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

        # Save snp hub with headers
        snp_data = []
        for k, v in self.hubs.snp_hub.hub.items(): #YF2
            snp_data.append([k, v[0], v[1], v[2], v[3], v[4], v[5]])

        # Sort snp_data by the second column (AVG_R2)
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order

        # Write snp hub to file
        with open(snp_file, 'w') as f:
            # Write the headers for the snp_file
            f.write("SNP,AVG_R2,FREQUENCY,BIN_ID,HUB_POSITION,BEST_INDIV_R2,BEST_ENCOD\n")
            for row in snp_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[5]}\n")

        return

    # get list of pipeline scores (r2, complexity) by position
    def get_pipeline_scores(self, pipelines: List[Pipeline]) -> List[Tuple[np.float32, np.int16]]:
        """
        Function to get the pipeline scores (r2, complexity) by position.

        Need to multiply the feature count by -1 to ensure that we are minimizing the feature count.
        """
        return [(pipeline.get_trait_r2(), np.int16(pipeline.get_trait_feature_cnt())) for pipeline in pipelines]

    # survival selection
    def survival_selection(self,
                           pop1: List[Pipeline],
                           pop1_scores: List[Tuple[np.float32, np.int16]],
                           pop2: List[Pipeline],
                           pop2_scores: List[Tuple[np.float32, np.int16]]) -> List[Pipeline]:
        """
        Function to select the survivors from the current population and offspring.

        Parameters:
        pop1: List[Pipeline]
            First list of pipelines
        pop1_scores: List[Tuple[np.float32, np.uint16]]
            First list of pipeline scores
        pop2: List[Pipeline]
            Second list of pipelines
        pop2_scores: List[Tuple[np.float32, np.uint16]]
            Second list of pipeline scores
        """
        # make sure all population scores are positive
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in pop1)
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in pop2)

        # combine both the population and offspring scores
        all_scores = np.array(np.concatenate((pop1_scores, pop2_scores), axis=0), dtype=np.float32)
        assert len(all_scores) == len(pop1) + len(pop2)
        assert len(all_scores) >= self.pop_size

        # get the fronts and rank
        fronts, _ = nsga.non_dominated_sorting(obj_scores=all_scores, weights=np.array([1.0, -1.0], dtype=np.float32))

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(all_scores, np.int32(2))

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
        # make sure that the number of survivors is correct
        assert len(survivor_ids) == self.pop_size

        # combine the population and offspring
        candidates = pop1 + pop2

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(candidates)
            new_pop.append(candidates[i])

        return new_pop

    ##YF
    # initialize the starting population
    def initialize_population(self) -> None:
        """
        Initialize the population of pipelines with their set of epistatic interactions and univariate snps.
        We start by creating (2 * pop_size) random set of interactions and find their best lo from the 9 we use.
        We will also create (2 * pop_size) random snps and find their best encoder type and r2.
        After, we remove bad interactions and snps (negative r2) for a given set and create pipelines from the good interactions and snps.
        We then evaluate the pipelines and keep only the pipelines with positive r2 scores.

        If the number of pipelines with positive r2 scores is less than the population size, we keep the same population.
        If the number of pipelines with positive r2 scores is greater than the population size, we use NSGA-II to get the pareto front.
        """
        # will hold sets of interactions for each pipeline in the population
        pop_epi_interactions = []
        # will hold the unseen interactions -- interactions not found in the Genohub
        unseen_interactions = set()

        #YF 
        # will hold snps for each pipeline in the population
        pop_uni = []
        # will hold unseen snps -- snps whose best encoder type is empty
        unseen_snps = set() # to avoid duplicates



        # create the initial population
        # we create double the population size to account for bad interactions
        # which are are extremely common given the nature of epistatic interactions
        for _ in range(self.pop_size * 2):
            # holds all interactions we are doing
            # set to make sure we don't have duplicates
            interactions = set()
            snps = set()

            # while we have not reached the max number of epi interactions
            while len(interactions) <= self.epi_cnt_max:
                # get random snp1 and snp2 and add to interactions
                snp1, snp2 = self.hubs.get_ran_interaction(self.rng)
                # add interaction to the set
                interactions.add((snp1, snp2))

            new_unseen = set(interaction for interaction in interactions 
                            if not self.hubs.is_interaction_in_hub(*interaction))
            unseen_interactions.update(new_unseen)
            # add to the population
            pop_epi_interactions.append(interactions)

            # while haven't reached the max number of uni snps
            while len(snps) <= self.uni_cnt_max:
                # get random snp and add to snps
                snp = self.hubs.get_ran_snp(self.rng)
                # add snp to the snps set
                snps.add(snp)

            new_snp = set(snp for snp in snps 
                            if not self.hubs.is_encoder_in_hub(snp))
            unseen_snps.update(new_snp)
            # add to the population
            pop_uni.append(snps)

            print("initial pop generated, now starts eval")



        # make sure we have the correct number of interactions
        assert len(pop_epi_interactions) == 2 * self.pop_size
        assert len(pop_uni) == 2 * self.pop_size #YF? Is this what we want?

        # evaluate all unseen interactions
        print("evaluating initial unseens")
        self.evaluate_unseen_interactions(unseen_interactions)
        self.evaluate_unseen_snps(unseen_snps)
        print("unseen eval done")

        ##YF remove bad interactions for each pipeline's set of interactions and snps together 
        for snps, interactions in zip(pop_uni, pop_epi_interactions):
            good_snps = self.remove_bad_snps(snps)
            good_interactions = self.remove_bad_interactions(interactions)

            # make sure we have the correct number of good interactions and snps
            assert len(good_snps) <= len(snps)
            assert len(good_interactions) <= len(interactions)

            # create pipeline and add to the population
            self.population.append(self.repoduction.generate_random_pipeline(self.rng, good_snps, good_interactions, int(self.seed)))

        "removed bad things"

        # make sure we have the correct number of pipelines
        assert len(self.population) ==  2 * self.pop_size

        # evaluate the initial population
        self.evaluation(self.population)
        # scores = self.get_pipeline_scores(self.population)
        # assert len(scores) == len(self.population)

        # subset the population to only include pipelines with positive r2 scores
        pop = []
        for pipeline in self.population:
            if pipeline.get_trait_r2() > 0.0:
                pop.append(pipeline)
        self.population = pop

        # get scores from the trimmed population
        #YF2 change positive_scores to ndarray to match requirement for non_dominated_sorting
        positive_scores = np.array(self.get_pipeline_scores(self.population))

        # make sure that the size of scores matches the population size
        assert len(positive_scores) == len(self.population)

        # if size positive_scores is less than the population size, we keep the same population
        if len(positive_scores) < self.pop_size:
            return
        # else we use nsga to get the pareto front from all the pipelines in the population
        else:
            # get the fronts and rank
            fronts, ranks = nsga.non_dominated_sorting(obj_scores=positive_scores, weights=np.array([1.0, -1.0], dtype=np.float32))
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
        All of this is done in asyncronous parallel jobs.
        We update the GenoHub with the results as they come in.

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
        assert len(ray_jobs) == len(unseen_interactions)

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, lo, snp1_name, snp2_name = ray.get(finished)[0]
            self.hubs.update_epi_n_snp_hub(snp1_name, snp2_name, r2, lo)

    ##YF? evaluate all unevaluated snps and update 
    def evaluate_unseen_snps(self, unseen_snps: Set) -> None:
        """
        Function to evaluate all unseen snps and add their best R2 and Encoder type to the GenoHub.
        All of this should be done in asyncronous parallel jobs.
        We update the GenoHub with the results as they come in.

        Parameters:
        unseen_snps: Set
            Unseen snps in a set to evaluate.
        """
        ray_jobs = []
        # collect all ray jobs for evaluation
        for snp_name in unseen_snps:
            ray_jobs.append(ray_uni_eval.remote(x_train = self.X_train_id,
                                                y_train = self.y_train_id,
                                                x_val = self.X_val_id,
                                                y_val = self.y_val_id,
                                                snp_name = snp_name,
                                                snp_pos = self.hubs.get_snp_pos(snp_name) ))
        assert len(ray_jobs) == len(unseen_snps)

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, type, snp_name = ray.get(finished)[0]
            self.hubs.update_snp_uni_hub(snp_name, r2, type)
    

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
    
    #YF
    def remove_bad_snps(self, snps: Set) -> Set:
        """
        Function to remove bad snps with r2<0 for a given set of snps

        Parameters:
        snps: Set of snps 
        """
        good_snps = set()
        for snp_name in snps:
            # check if r2 is positive
            if self.hubs.get_uni_res(snp_name) > np.float32(0.0):
                # add to good snps
                good_snps.add(snp_name)
        # return the good snps
        return good_snps

    # print the population
    def print_population(self) -> None:
        """
        Function to print the population.
        """
        print('Population:')
        for p in self.population:
            p.print_pipeline()


    #YF
    # evaluate the population                                   # r2 , feature count, pop_id
    def evaluation(self, pop: List[Pipeline]) -> None:
        """
        #YF edited
        Function to evaluate entire pipelines after upating all unseen interactions and snps to the GenoHub.
        So the evaluation here only calls GenoHub to get required positions and values since the lo and type for interactions and snps are already pre-determined. 
        
        Parameters:
        pop: List[Pipeline]
            List of Sklearn pipelines constructed in the EA process. 
        """

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
                                                     self.construct_uni_nodes(pipeline.get_uni_snps()),
                                                     pipeline.get_selector_node(),
                                                     pipeline.get_root_node(),
                                                     np.int16(i)))
        assert len(ray_jobs) == len(pop)

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, feature_count, pop_id = ray.get(finished)[0]
            # update the pipeline
            pop[pop_id].set_traits([r2, feature_count])
        return

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
    
    #YF
    # construct uni_nodes for a pipeline's set of individual snps
    def construct_uni_nodes(self, uni_snps: Set) -> uni_node_list_t:
        """
        Function to construct uni nodes for a pipeline's set of univariates.

        Parameters:
        uni_snps: Set
            Set of snps.
        """
        uni_nodes = []
        id = 0
        for snp_name in uni_snps:
            # get the uni snp type
            uni_type = self.hubs.get_uni_type(snp_name)
            # get each snps position in the hub
            snp_pos = self.hubs.get_snp_pos(snp_name)

            if uni_type == np.str_('dominant'):
                uni_nodes.append(UniDominantNode(name=f"UniDominantNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('recessive'):
                uni_nodes.append(UniRecessiveNode(name=f"UniRecessiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('heterosis'):
                uni_nodes.append(UniHeterosisNode(name=f"UniHeterosisNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('underdominant'):
                uni_nodes.append(UniUnderDominantNode(name=f"UniUnderDominantNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('subadd'):
                uni_nodes.append(UniSubadditiveNode(name=f"UniSubadditiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('superadd'):
                uni_nodes.append(UniSuperadditiveNode(name=f"UniSuperadditiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('pager'):
                uni_nodes.append(UniPAGERNode(name=f"UniPAGERNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            else:
                exit('Error: The univariate snp type is not valid. Please provide a valid type.', -1)

            id += 1
        # return the list of epi nodes
        return uni_nodes

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
        fronts, ranks = nsga.non_dominated_sorting(obj_scores=scores, weights=np.array([1.0, -1.0], dtype=np.float32))
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
        # print('processing offspring unseen interactions:', len(unseen_interactions))

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

    # plot the current pareto front from the population with complexity and r2 scores
    def plot_pareto_front(self) -> None:
        """
        Function to plot the current pareto front from the population with complexity and r2 scores.
        """

        # get all scores from the current population
        pop_scores = np.array(self.get_pipeline_scores(self.population), dtype=np.float32)

        # get the fronts and rank
        fronts, rank = nsga.non_dominated_sorting(obj_scores=pop_scores, weights=np.array([1.0, -1.0], dtype=np.float32))

        # remove scores that are not of rank 0
        pareto_front = pop_scores[rank == 0]

        # plot the pareto front
        plt.scatter(pareto_front[:, 1], pareto_front[:, 0])
        plt.xlabel('Feature Count')
        plt.ylabel('R2 Score')

        # show grid
        plt.grid(True)

        # show plot
        plt.show()


def main():
    # set experiemnt configurations
    ea_config = {'seed': np.uint16(0),
                 'pop_size': np.uint16(100),
                 'epi_cnt_max': np.uint16(150),
                 'epi_cnt_min': np.uint16(10),
                 'uni_cnt_max': np.uint16(150),
                 'uni_cnt_min': np.uint16(10),
                 'cores': 8,
                 'mut_selector_p': np.float64(1.0),
                 'mut_regressor_p': np.float64(.5),
                 'mut_ran_p':np.float64(.45),
                 'mut_smt_p': np.float64(.45),
                 'mut_non_p': np.float64(.1),
                 'smt_in_in_p': np.float64(.10),
                 'smt_in_out_p': np.float64(.450),
                 'smt_out_out_p': np.float64(.450),
                 'mut_prob': np.float64(.5),
                 'cross_prob': np.float64(.5),
                 'num_add_interactions': np.uint16(10),
                 'num_del_interactions': np.uint16(10)}

    ea = EA(**ea_config)
    # need to update the path to the data file
    data_dir = './test_data/BMIwTail_qtl_10.csv'
    # data_dir = '/Users/hernandezj45/Desktop/Repositories/pilot-star-base-epi/18qtl_pruned_BMIres.csv'
    ea.data_loader(data_dir)
    ea.initialize_hubs(20) #try with increased numbers
    ea.evolve(20)
    ea.save_hubs(epi_file='./epi_file.txt', snp_file='./snp_file.txt')


    ray.shutdown()

if __name__ == "__main__":
    main()