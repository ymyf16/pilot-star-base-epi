#####################################################################################################
#
# Reproduction class that generating new pipelines.
# We use a combination of mutation and crossover to generate new pipelines (both user specified).
# We also initialize the first population of solutions (this is reproduction w/out parents).
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Tuple, Set
import numpy.typing as npt

from pipeline import Pipeline
from geno_hub import GenoHub

import copy as cp

# feature selectors
from scikit_node import VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode
# regressors
from scikit_node import LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode

rng_t = np.random.Generator
pop_size_t = np.uint16
epi_interactions_t = Set[Tuple[np.str_, np.str_]]
# probability type
prob_t = np.float64
# snp type
snp_t = np.str_

@typechecked
class Reproduction:
    def __init__(self,
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

        # save all the variables
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.mut_selector_p = mut_selector_p
        self.mut_regressor_p = mut_regressor_p
        self.mut_ran_p = mut_ran_p
        self.mut_non_p = mut_non_p
        self.mut_smt_p = mut_smt_p
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p

        return

    def generate_random_pipeline(self, rng: rng_t, interactions: epi_interactions_t, seed: int) -> Pipeline:
        # quick checks
        assert len(interactions) > 0

        # randomly select selector nodes and root nodes
        selector_node = rng.choice([VarianceThresholdNode(rng=rng, seed=seed),
                                    SelectPercentileNode(rng=rng, seed=seed),
                                    SelectFweNode(rng=rng, seed=seed),
                                    SelectFromModelLasso(rng=rng, seed=seed),
                                    SelectFromModelTree(rng=rng, seed=seed),
                                    SequentialFeatureSelectorNode(rng=rng, seed=seed)
                                ])
        root_node = rng.choice([LinearRegressionNode(rng=rng, seed=seed),
                                RandomForestRegressorNode(rng=rng, seed=seed),
                                SGDRegressorNode(rng=rng, seed=seed),
                                DecisionTreeRegressorNode(rng=rng, seed=seed),
                                ElasticNetNode(rng=rng, seed=seed),
                                SVRNode(rng=rng, seed=seed),
                                GradientBoostingRegressorNode(rng=rng, seed=seed),
                                MLPRegressorNode(rng=rng, seed=seed)
                            ])
        # create the pipeline
        return Pipeline(selector_node=selector_node, root_node=root_node, epi_pairs=interactions, traits=[])

    def variation_order(self, rng: rng_t, offpring_cnt: pop_size_t) -> Tuple[List[str], pop_size_t]:
        """
        Function to generate the order of variation operators to be applied to generate offspring.
        The order is determined by the probabilities of mutation and crossover.
        We return a list with the names of the operators in the order they should be applied.
        E.g.: ['m', 'c', 'm', 'c', ...]

        Crossover means two parents are required
        Mutation means one parent is required

        Parameters:
        rng (rng_t): A numpy random number generator from the evolver
        offpring_cnt (pop_size_t): The number of offspring to generate

        Returns:
        List[np.str_]: A list of strings representing the order of variation operators to be applied
        pop_size_t: The number of parents needed to generate the offspring
        """
        # parents needed by variantion operators
        parent_count = {'m': 1, 'c': 2}

        # generate half of the operators
        # we do this so that in the worse case (all crossover) we don't need to generate more than needed
        order = rng.choice(['m', 'c'], offpring_cnt // 2, p=[self.mut_prob, self.cross_prob]).tolist()

        # how many more offspring do we need
        left = offpring_cnt -  np.uint16(sum(parent_count[op] for op in order))

        # get that many more operators
        while left > 0:
            op = rng.choice(['m', 'c'], 1, p=[self.mut_prob, self.cross_prob])[0]

            # check if we can add the operator
            if parent_count[op] <= left:
                order.append(op)
                left -= parent_count[op]

            # if left == 1, we can only add mutation
            elif left == 1:
                order.append('m')
                left -= 1

        # make sure we have the right number of offspring
        assert sum(parent_count[op] for op in order) == offpring_cnt

        # return the order and number of parents needed
        return order, pop_size_t(sum(parent_count[op] for op in order))

    # method to generate offspring
    def produce_offspring(self,
                          rng: rng_t,
                          hub: GenoHub,
                          offspring_cnt: pop_size_t,
                          population: List[Pipeline],
                          parent_ids: List[np.uint16],
                          order: List[str],
                          seed: int) -> List[Pipeline]:
        # quick checks
        assert len(parent_ids) > 0
        assert len(population) > 0
        assert offspring_cnt > 0

        # list to store the offspring
        offspring = []

        # go through the order of operators
        p_id = 0
        for op in order:
            # mutation
            if op == 'm':
                offspring.append(self.mutate(rng, population[parent_ids[p_id]], hub))
                p_id += 1
            elif op == 'c':
                off1, off2 = self.crossover(rng, population[parent_ids[p_id]], population[parent_ids[p_id+1]], hub)
                offspring.append(off1)
                offspring.append(off2)
                p_id += 2
            else:
                raise ValueError(f"Unknown operator: {op}")
        # make sure we have the right number of offspring
        assert len(offspring) == offspring_cnt
        assert p_id == len(parent_ids)

        # return the offspring
        return offspring

    # what mutation are we applying to the pipeline
    def mutate(self,
               rng: np.random.Generator,
               parent: Pipeline,
               hub: GenoHub) -> Pipeline:

        # clone the pipeline
        offspring = Pipeline(epi_pairs=set(),
                            selector_node=parent.get_selector_node(),
                            root_node=parent.get_root_node(),
                            traits=[])

        # get parent epi pairs
        parent_epi_pairs = cp.deepcopy(parent.get_epi_pairs())

        # print offspring pipeline
        # print('*'*100)
        # print("Offspring pipeline before:")
        # parent.print_pipeline()
        # print('*'*100)

        # mutate the offspring
        epi_pairs = set()

        # go through the epi branches and mutate if needed
        for interaction in parent_epi_pairs:
            # coin flip to determine if we mutate
            if rng.choice([True, False], p=[self.mut_non_p, 1.0-self.mut_non_p]):
                # coin flip to determine the type of mutation
                if rng.choice([True, False], p=[self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p), self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p)]):
                    # smart mutation
                    epi_pairs.add(self.mutate_epi_node_smrt(rng, hub, interaction[0], interaction[1]))
                else:
                    # smt mutation
                    epi_pairs.add(self.mutate_epi_node_rand(rng, hub, interaction[0], interaction[1]))
            else:
                # no mutation
                epi_pairs.add(interaction)

        # update the epi pairs
        offspring.set_epi_pairs(epi_pairs)

        # mutate the selector node
        if rng.choice([True, False], p=[self.mut_selector_p, 1.0-self.mut_selector_p]):
            offspring.mutate_selector_node(rng)

        # mutate the regressor node
        if rng.choice([True, False], p=[self.mut_regressor_p, 1.0-self.mut_regressor_p]):
            offspring.mutate_root_node(rng)

        # print offspring pipeline
        # print("Offspring pipeline after:")
        # offspring.print_pipeline()
        # print('*'*100)
        # print()

        return offspring


    # execute a smart mutation on the epi_node
    def mutate_epi_node_smrt(self,
                             rng: rng_t,
                             hub: GenoHub,
                             snp1_name: snp_t,
                             snp2_name: snp_t) -> Tuple[snp_t, snp_t]:
        # will hold new interactions
        new_snp1_name, new_snp2_name = None, None

        # randomly select one of the SNPs
        new_snp1_name = rng.choice([snp1_name, snp2_name])

        # randomly select one of the mutations to perform
        mut_fun = rng.choice([0,1,2], p=[self.smt_in_in_p, self.smt_in_out_p, self.smt_out_out_p])

        if mut_fun == 0:
            # in chromosome and in bin
            new_snp2_name = hub.get_smt_snp_in_bin(snp=new_snp1_name, rng=rng)
        elif mut_fun == 1:
            # in chromosome and out of bin
            new_snp2_name = hub.get_smt_snp_in_chrm(snp=new_snp1_name, rng=rng)
        elif mut_fun == 2:
            # out of chromosome
            new_snp2_name = hub.get_smt_snp_out_chrm(snp=new_snp1_name, rng=rng)
        else:
            exit("Unknown mutation function", -1)

        # make sure the new snps are set
        assert new_snp1_name != None and new_snp2_name != None

        # return new snps in order by size
        if new_snp1_name > new_snp2_name:
            return new_snp2_name, new_snp1_name
        return new_snp1_name, new_snp2_name

    # execute a random mutation on the epi_node
    def mutate_epi_node_rand(self,
                             rng: rng_t,
                             hub: GenoHub,
                             snp1_name: snp_t,
                             snp2_name: snp_t) -> Tuple[snp_t, snp_t]:
        # will hold new interactions
        new_snp1_name, new_snp2_name = None, None

        # randomly select one of the SNPs
        new_snp1_name = rng.choice([snp1_name, snp2_name])

        # randomly select another SNP
        new_snp2_name = hub.get_ran_snp(rng)

        # make new snps are different
        while new_snp2_name == new_snp1_name:
            new_snp2_name = hub.get_ran_snp(rng)

        # make sure the new snps are set
        assert new_snp1_name != None and new_snp2_name != None

        # return new snps in order by size
        if new_snp1_name > new_snp2_name:
            return new_snp2_name, new_snp1_name

        return new_snp1_name, new_snp2_name

    # execute a crossover between two pipelines
    def crossover(self,
                  rng: np.random.Generator,
                  parent1: Pipeline,
                  parent2: Pipeline,
                  hub: GenoHub) -> Tuple[Pipeline, Pipeline]:
        # get the epi branches from the parents
        p1_epi_pairs = list(parent1.get_epi_pairs())
        p2_epi_pairs = list(parent2.get_epi_pairs())

        # get smallest half length from both
        half_len = min(len(p1_epi_pairs), len(p2_epi_pairs)) // 2

        # randomly select indecies from both parents epi branches
        p1_idx = rng.choice(len(p1_epi_pairs), half_len, replace=False)
        p2_idx = rng.choice(len(p2_epi_pairs), half_len, replace=False)

        # swap elements between parents
        for i1, i2 in zip(p1_idx, p2_idx):
            p1_epi_pairs[i1], p2_epi_pairs[i2] = p2_epi_pairs[i2], p1_epi_pairs[i1]

        # create one offspring per parent
        offspring_1 = Pipeline(epi_pairs=set(p1_epi_pairs),
                               selector_node=parent1.get_selector_node(),
                               root_node=parent1.get_root_node(),
                               traits=[])

        offsprint_2 = Pipeline(epi_pairs=set(p2_epi_pairs),
                               selector_node=parent2.get_selector_node(),
                               root_node=parent2.get_root_node(),
                               traits=[])

        return offspring_1, offsprint_2