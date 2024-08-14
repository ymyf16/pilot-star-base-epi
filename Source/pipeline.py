# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

from sklearn.base import BaseEstimator
from epi_nodes import EpiNode, EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
from typing import List, Dict
from scikit_nodes import ScikitNode, VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode, LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode, MLPRegressorNode
import numpy as np

class Pipeline:
    def __init__(self,
                 epi_pairs: List[List], 
                 epi_branches: List[EpiNode], # each branch consists of one EpiNode
                 selector_node: ScikitNode,
                 root_node: ScikitNode, 
                 traits: Dict[int, float], 
                 clone: bool = False,
                 max_feature_count: int = 10):    
        
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
    
    # predict method
    def predict(self, X):
        # todo: implement the predict method
        pass

    def generate_random_pipeline(self, rng, header_list): # pass upper bound no of epi nodes.
        # generate a random pipeline
        # 1. Define lists of possible selector and regressor nodes
        print("Random Seed: ", rng)
        selector_nodes = [
            VarianceThresholdNode(rng=rng),
            SelectPercentileNode(rng=rng),
            SelectFweNode(rng=rng),
            SelectFromModelLasso(rng=rng),
            SelectFromModelTree(rng=rng),
            SequentialFeatureSelectorNode(rng=rng)
        ]
    
        regressor_nodes = [
        LinearRegressionNode(rng=rng),
        RandomForestRegressorNode(rng=rng),
        SGDRegressorNode(rng=rng),
        DecisionTreeRegressorNode(rng=rng),
        ElasticNetNode(rng=rng),
        SVRNode(rng=rng),
        GradientBoostingRegressorNode(rng=rng),
        MLPRegressorNode(rng=rng)
        ]
    
        # 2. Randomly select a selector node
        selector_node = rng.choice(selector_nodes)
    
        # 3. Randomly select a regressor node
        root_node = rng.choice(regressor_nodes)
    
        # 4. Generate multiple EpiNodes
        epi_nodes = [
            EpiCartesianNode(name="EpiCartesianNode", rng=rng, header_list=header_list),
            EpiXORNode(name="EpiXORNode", rng=rng, header_list=header_list),
            EpiRRNode(name="EpiRRNode", rng=rng, header_list=header_list),
            EpiRDNode(name="EpiRDNode", rng=rng, header_list=header_list),
            EpiTNode(name="EpiTNode", rng=rng, header_list=header_list),
            EpiModNode(name="EpiModNode", rng=rng, header_list=header_list),
            EpiDDNode(name="EpiDDNode", rng=rng, header_list=header_list),
            EpiM78Node(name="EpiM78Node", rng=rng, header_list=header_list),
            EpiPAGERNode(name="EpiPAGERNode", rng=rng, header_list=header_list)
        ]
    
        # Generate EpiNodes with random interacting pairs (numpy array indexes)
        num_epi_nodes = rng.integers(2, self.max_feature_count)  # Randomly select how many EpiNodes to create - upper bound to be provided by user.
        epi_branches = []
        epi_pairs = []
        for _ in range(num_epi_nodes):
            selected_epi_node_class = rng.choice(epi_nodes)
            #num_snps = 2  # for 2-way interactions
            #interacting_features = [rng.randint(0, 1000, size=num_snps)]  # Random numpy array indexes
            epi_node_instance = selected_epi_node_class.__class__(name=f"{selected_epi_node_class.__class__.__name__}_{_}", rng=rng, header_list = header_list) 
            epi_branches.append(epi_node_instance)
            # epi_pairs.append((epi_node_instance.snp1_name, epi_node_instance.snp2_name))

        # Create the pipeline using the generated nodes
        pipeline = Pipeline(epi_pairs=epi_pairs,
                            epi_branches=epi_branches,
                            selector_node=selector_node,
                            root_node=root_node,
                            traits={})

        # Create and return the pipeline
        return pipeline


    def mutate_pair_snpwise(self, rng):
        # mutate a pair of nodes
        pass    

    def mutate_selector_node(self, rng):
        # mutate the selector node
        pass

    def mutate_root_node(self, rng):
        # mutate the root node
        pass    


    # def AddRandomSelectorNode(self, rng):
    #     classs = rng.choice([SELEC, RandomForestClassifier, LinearRegression])
    #     self.selector_node = classs(rng)

 
# # create a default pipeline
# pipeline = Pipeline(epi_pairs=[], epi_branches=[], selector_node=None, root_node=None, traits={}, clone=False, max_feature_count=10)
# pipeline = pipeline.generate_random_pipeline(np.random.default_rng(0))
# print(pipeline.get_branch_count())
# #print(pipeline.get_feature_count())
# print(pipeline.get_selector_node())
# print(pipeline.get_root_nood())