# This script will test the scikit_nodes.py file
import sys
sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')
import unittest
from unittest import TestCase
import numpy as np
import pandas as pd
from scikit_nodes import VarianceThresholdNode, SelectPercentileNode, LinearRegressionNode, SVRNode

class TestScikitNodes(TestCase):
    def setUp(self):
        # Load the dataset
        data = pd.read_csv("/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/3XOR_20features.csv")
        self.genotypes = data.iloc[:, :-1].values  # Features
        self.phenotype = data.iloc[:, -1].values  # Target

    def test_variance_threshold_node(self):
        # Create a VarianceThresholdNode object
        var_node = VarianceThresholdNode(name='VarianceThreshold', threshold=0.1)
        # Fit the VarianceThresholdNode object
        var_node.fit(self.genotypes)
        # Transform the data
        transformed_X = var_node.transform(self.genotypes)
        # Check if the transformed numpy array has the expected shape
        self.assertTrue(transformed_X.shape[1] <= self.genotypes.shape[1])
        # Additional checks can be added as needed

    def test_select_percentile(self):
        # Create a SelectPercentileNode object
        select_node = SelectPercentileNode(name='SelectPercentile', percentile=10)
        # Fit the SelectPercentileNode object
        select_node.fit(self.genotypes, self.phenotype)
        # Transform the data
        transformed_X = select_node.transform(self.genotypes)
        # Check if the transformed numpy array has the expected shape
        self.assertTrue(transformed_X.shape[1] <= self.genotypes.shape[1])

    def test_linear_regression(self):
        # Create a LinearRegressionNode object
        lin_reg = LinearRegressionNode(name='LinearRegression')
        # Fit the LinearRegressionNode object
        lin_reg.fit(self.genotypes, self.phenotype)
        # Predict the target values
        y_pred = lin_reg.predict(self.genotypes)
        # Check if the predicted values have the expected shape
        self.assertEqual(y_pred.shape, self.phenotype.shape)

    def test_svr(self):
        # Create a SVRNode object
        svr = SVRNode(name='SVR')
        # Fit the SVRNode object
        svr.fit(self.genotypes, self.phenotype)
        # Predict the target values
        y_pred = svr.predict(self.genotypes)
        # Check if the predicted values have the expected shape
        self.assertEqual(y_pred.shape, self.phenotype.shape)

if __name__ == '__main__':
    unittest.main()
