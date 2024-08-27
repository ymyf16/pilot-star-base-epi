# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

from sklearn.base import BaseEstimator
from epi_nodes import EpiNode, EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
from typing import List, Dict, Tuple, Set
from scikit_nodes import ScikitNode, VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode, LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode
import numpy as np
from typeguard import typechecked
import numpy.typing as npt
from typing_extensions import Self
import copy as cp

@typechecked # for debugging purposes
class Pipeline:
    def __init__(self, # TODO: get ride of all varaiables except epi_pairs and traits
                 epi_pairs: Set[Tuple],
                 epi_branches: List[EpiNode], # each branch consists of one EpiNode
                 selector_node: ScikitNode | None,
                 root_node: ScikitNode | None,
                 traits: Dict[int, float],
                 ):

        # need to make a deep copy to keep independent
        self.epi_pairs = cp.deepcopy(epi_pairs) # will contain the interacting_features from each EpiNode/branch
        self.epi_branches = cp.deepcopy(epi_branches) # each branch consists of one EpiNode
        self.selector_node = cp.deepcopy(selector_node)
        self.root_node = cp.deepcopy(root_node)
        self.traits = cp.deepcopy(traits) # no of features going in the final regressor and the R^2 value from the regressor

    def get_trait_r2(self) -> np.float32:
        assert 'r2' in self.traits
        return self.traits['r2']

    def get_trait_feature_cnt(self) -> np.uint16:
        assert 'feature_cnt' in self.traits
        return self.traits['feature_cnt']

    def set_traits(self, traits: Dict[int, float]) -> None:
        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 2
        # make sure that the traits are a dictionary
        assert 'r2' in traits and 'feature_cnt' in traits

        # update the traits
        self.traits.update(traits)
        return

    def get_traits(self) -> Dict[int, float]:
        return self.traits

    def get_epi_pairs(self) -> Set[Tuple]:
        return self.epi_pairs

    def get_epi_branches(self) -> List[EpiNode]:
        return self.epi_branches

    # method to get the number of nodes in the pipeline
    def get_branch_count(self):
        return len(self.epi_branches)

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
        print('len(self.epi_branches):', len(self.epi_branches))
        for epi_node in self.epi_branches:
            print(epi_node)
        print("Selector Node:")
        print(self.selector_node)
        print("Root Node:")
        print(self.root_node)
        return

    def mutate_pair_snpwise(self, rng):
        # mutate a pair of nodes
        pass

    def mutate_selector_node(self, rng):
        print('pipeline::mutate_selector_node')
        self.selector_node.mutate(rng)

    def mutate_root_node(self, rng):
        self.root_node.mutate(rng)