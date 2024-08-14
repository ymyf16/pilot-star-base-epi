import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')
from epi_nodes import EpiCartesianNode, EpiXORNode
from pipeline import Pipeline

def test_pipeline():
    # Example epi_pairs and epi_branches
    # read a sample dataset
    data = pd.read_csv("/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/3XOR_20features.csv")
    genotypes = data.iloc[:, 0:-1]
    phenotype = data.iloc[:, -1]
    # split the data into training_testing and validation sets
    genotypes_train_test, genotypes_validation, phenotype_train_test, phenotype_validation = train_test_split(genotypes, phenotype, test_size=0.2)

    # split the training_testing set into training and testing sets
    genotypes_train, genotypes_test, phenotype_train, phenotype_test = train_test_split(genotypes_train_test, phenotype_train_test, test_size=0.5)

    snp1_train = genotypes_train.iloc[:, 0].values
    snp2_train = genotypes_train.iloc[:, 1].values

    snp1_test = genotypes_test.iloc[:, 0].values
    snp2_test = genotypes_test.iloc[:, 1].values
    
    features_test = [snp1_test, snp2_test]
    epi_pairs = [(genotypes_test.iloc[:, 0].values, genotypes_test.iloc[:, 1].values), (genotypes_test.iloc[:, 3].values, genotypes_test.iloc[:, 4].values)]

    epi_branches = [EpiCartesianNode("CartesianNode1", features=epi_pairs[0]), 
                    EpiCartesianNode("CartesianNode2", features=epi_pairs[1]),
                    EpiXORNode("XORNode1", features=epi_pairs[0])]
    
    # Example selector_node and root_node
    selector_node = RandomForestRegressor()
    root_node = RandomForestRegressor()
    
    # Example traits
    traits = {'scorer': 0.95}
    
    pipeline = Pipeline(epi_pairs, epi_branches, selector_node, root_node, traits)
    
    print("Total number of nodes:", pipeline.getNodeCount())
    print("Total number of features in feature sets:", pipeline.getFeatureCount())

if __name__ == "__main__":
    test_pipeline()
