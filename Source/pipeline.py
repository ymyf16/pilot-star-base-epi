# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

from sklearn.base import BaseEstimator
from typing import List, Dict, Tuple, Set
from scikit_node import ScikitNode
import numpy as np
from typeguard import typechecked
import numpy.typing as npt
from typing_extensions import Self
import copy as cp

# numpy random number generator type
gen_rng_t = np.random.Generator
gen_snp_arr_t = npt.NDArray[np.str_]
epi_pairs_t = Set[Tuple]
traits_t = List

@typechecked # for debugging purposes
class Pipeline:
    def __init__(self,
                 epi_pairs: epi_pairs_t,
                 selector_node: ScikitNode,
                 root_node: ScikitNode,
                 traits: traits_t) -> None:

        # need to make a deep copy to keep independent
        self.epi_pairs = cp.deepcopy(epi_pairs) # will contain the interacting_features
        self.traits = cp.deepcopy(traits) # will contain the r2 (traits[0]) and feature_cnt (traits[1])
        self.selector_node = cp.deepcopy(selector_node) # will contain the selector node
        self.root_node = cp.deepcopy(root_node) # will contain the regressor node

    # method to set the epi_pairs
    def set_epi_pairs(self, epi_pairs: epi_pairs_t) -> None:
        # check that internal epi_pairs is empty
        assert len(self.epi_pairs) == 0
        # make sure that the epi_pairs are not empty
        assert len(epi_pairs) > 0

        # update the epi_pairs
        self.epi_pairs = cp.deepcopy(epi_pairs)
        return

    # set traits
    def set_traits(self, traits: traits_t) -> None:
        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 2

        # update the traits
        self.traits = cp.deepcopy(traits)
        return

    def get_trait_r2(self) -> np.float32:
        assert len(self.traits) == 2
        return self.traits[0]

    def get_trait_feature_cnt(self) -> np.uint16:
        assert len(self.traits) == 2
        return self.traits[1]

    def get_traits(self) -> traits_t:
        return self.traits

    def get_epi_pairs(self) -> Set[Tuple]:
        return self.epi_pairs

    # method to get the number of nodes in the pipeline
    def get_branch_count(self):
        return len(self.epi_pairs)

    # method to get the number of features/SNPs in the pipeline
    def get_feature_count(self):
        return self.selector_node.get_feature_count()

    # method to get the selector node
    def get_selector_node(self):
        return self.selector_node

    # method to get the root node
    def get_root_node(self):
        return self.root_node

    # print the pipeline
    def print_pipeline(self) -> None:
        print("Pipeline:")
        print("EpiNodes:")
        print('len(self.epi_pairs):', len(self.epi_pairs))
        for epi_node in self.epi_pairs:
            print(epi_node)
        print("Selector Node:")
        print(self.selector_node.selector)
        print("Root Node:")
        print(self.root_node.regressor)
        return

    def mutate_selector_node(self, rng: gen_rng_t) -> None:
        self.selector_node.mutate(rng)

    def mutate_root_node(self, rng: gen_rng_t) -> None:
        self.root_node.mutate(rng)