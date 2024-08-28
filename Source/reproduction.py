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

# feature selectors
from scikit_node import VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode
# regressors
from scikit_node import LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode

rng_t = np.random.Generator
pop_size_t = np.uint16
epi_interactions_t = Set[Tuple[np.str_, np.str_]]

@typechecked
class Reproduction:
    def __init__(self,
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

    def generate_random_pipeline(self, rng: rng_t, interactions: epi_interactions_t) -> Pipeline:
        # quick checks
        assert len(interactions) > 0

        # randomly select selector nodes and root nodes
        selector_node = rng.choice([VarianceThresholdNode(rng=rng),
                                    SelectPercentileNode(rng=rng),
                                    SelectFweNode(rng=rng),
                                    SelectFromModelLasso(rng=rng),
                                    SelectFromModelTree(rng=rng),
                                    SequentialFeatureSelectorNode(rng=rng)
                                ])
        root_node = rng.choice([LinearRegressionNode(rng=rng),
                                RandomForestRegressorNode(rng=rng),
                                SGDRegressorNode(rng=rng),
                                DecisionTreeRegressorNode(rng=rng),
                                ElasticNetNode(rng=rng),
                                SVRNode(rng=rng),
                                GradientBoostingRegressorNode(rng=rng),
                                # MLPRegressorNode(rng=rng)
                            ])
        # create the pipeline
        return Pipeline(selector_node=selector_node, root_node=root_node, epi_pairs=interactions, traits=[])

    def variation_order(self, rng: rng_t, offpring_cnt: pop_size_t) -> Tuple[List[np.str_], pop_size_t]:
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
