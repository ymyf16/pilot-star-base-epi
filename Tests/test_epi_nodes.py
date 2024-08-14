import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')

from epi_nodes import EpiCartesianNode, EpiXORNode, EpiPAGERNode

def test_epi_nodes():
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

    # Test EpiCartesianNode
    cartesian_node = EpiCartesianNode("CartesianNode", features_test)
    cartesian_node.print_data()
    result = cartesian_node.process_data()
    print("EpiCartesianNode Result:", result)
    print("EpiCartesianNode Result type:", type(result))
    
    # Test EpiXORNode
    xor_node = EpiXORNode("XORNode", features_test)
    xor_node.print_data()
    result = xor_node.process_data()
    print("EpiXORNode Result:", result)
    print("EpiXORNode Result type:", type(result))
    
    # Test EpiPAGERNode
    features_train = [snp1_train, snp2_train]
    pager_node = EpiPAGERNode("PAGERNode", features_test, features_train, phenotype_train)
    pager_node.print_data()
    result = pager_node.process_data()
    print("EpiPAGERNode Result:", result)
    print("EpiPAGERNode Result type:", type(result))

if __name__ == "__main__":
    test_epi_nodes()
