# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

from sklearn.base import BaseEstimator
from epi_nodes import EpiNode, EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
from typing import List, Dict, Tuple
from scikit_nodes import ScikitNode, VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode, LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode
import numpy as np
from typeguard import typechecked
import numpy.typing as npt
from typing_extensions import Self

@typechecked # for debugging purposes
class Pipeline:
    def __init__(self,
                 epi_pairs: List[Tuple],
                 epi_branches: List[EpiNode], # each branch consists of one EpiNode
                 selector_node: ScikitNode | None,
                 root_node: ScikitNode | None,
                 traits: Dict[int, float],
                 max_feature_count: np.uint16,
                 clone: bool = False):

        self.epi_pairs = epi_pairs # will contain the interacting_features from each EpiNode/branch
        self.epi_branches = epi_branches # each branch consists of one EpiNode
        self.selector_node = selector_node
        self.root_node = root_node
        self.traits = traits # no of features going in the final regressor and the R^2 value from the regressor
        self.clone = clone # true if no mutations applied to the pipeline
        self.max_feature_count = max_feature_count # maximum number of features/SNPs in the pipeline

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
    def get_root_nood(self):
        return self.root_node

    # generate a random pipeline
    def generate_random_pipeline(self, rng, header_list: npt.NDArray[np.str_]) -> Self:
        # randomly select a selector node
        selector_node = rng.choice([VarianceThresholdNode(rng=rng),
                                    SelectPercentileNode(rng=rng),
                                    SelectFweNode(rng=rng),
                                    SelectFromModelLasso(rng=rng),
                                    SelectFromModelTree(rng=rng),
                                    SequentialFeatureSelectorNode(rng=rng)
                                ])

        # randomly select a regressor node
        root_node = rng.choice([LinearRegressionNode(rng=rng),
                                RandomForestRegressorNode(rng=rng),
                                SGDRegressorNode(rng=rng),
                                DecisionTreeRegressorNode(rng=rng),
                                ElasticNetNode(rng=rng),
                                SVRNode(rng=rng),
                                GradientBoostingRegressorNode(rng=rng),
                                MLPRegressorNode(rng=rng)
                            ])

        # randomly select how many EpiNodes to create - upper bound to be provided by user.
        num_epi_nodes = rng.integers(2, self.max_feature_count)
        epi_branches = []
        epi_pairs = []

        # Generate EpiNodes with random interacting pairs (numpy array indexes)
        for _ in range(num_epi_nodes):
            # select two snps randomly from the header_list
            snp1_name, snp2_name = rng.choice(header_list, size=2, replace=False)

            # make sure snp1 < snp2 or else swap
            if snp1_name > snp2_name:
                snp1_name, snp2_name = snp2_name, snp1_name

            # find the index of the selected snps
            snp1_index = np.uint32(np.where(header_list == snp1_name)[0][0])
            snp2_index = np.uint32(np.where(header_list == snp2_name)[0][0])

            # randomly select an EpiNode class
            epi_node_class = rng.choice([EpiCartesianNode(name=f"EpiCartesianNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiXORNode(name=f"EpiXORNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiRRNode(name=f"EpiRRNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiRDNode(name=f"EpiRDNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiTNode(name=f"EpiTNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiModNode(name=f"EpiModNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiDDNode(name=f"EpiDDNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiM78Node(name=f"EpiM78Node_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                        EpiPAGERNode(name=f"EpiPAGERNode_{_}", snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_index, snp2_pos=snp2_index),
                                    ])
            epi_branches.append(epi_node_class)
            epi_pairs.append((epi_node_class.get_snp1_name(), epi_node_class.get_snp2_name()))

        # Create and return the pipeline
        return Pipeline(epi_pairs=epi_pairs,
                            epi_branches=epi_branches,
                            selector_node=selector_node,
                            root_node=root_node,
                            max_feature_count=self.max_feature_count,
                            clone=False,
                            traits={})

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
        # mutate the selector node
        pass

    def mutate_root_node(self, rng):
        # mutate the root node
        pass