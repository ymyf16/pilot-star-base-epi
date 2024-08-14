# function to read in a Pipeline object and convert it to a scikit learn pipeline object

from sklearn.pipeline import Pipeline as SklearnPipeline
from pipeline import Pipeline # our custom pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict
import numpy as np

class Pipeline_Builder:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def build_sklearn_pipeline(self) -> SklearnPipeline:
        # Create a dictionary to hold the steps for the scikit-learn pipeline
        steps = []

        # Add EpiNodes to FeatureUnion
        epi_transformers = {}
        for epi_node in self.pipeline.epi_branches:
            epi_transformers[epi_node.name] = epi_node
        
        feature_union = FeatureUnion(transformer_list=[
            (name, transformer) for name, transformer in epi_transformers.items()
        ])
        
        # Add FeatureUnion to the steps
        steps.append(('epi_features', feature_union))

        # Add selector node to the steps
        steps.append(('selector', self.pipeline.selector_node))

        # Add regressor node to the steps
        steps.append(('regressor', self.pipeline.root_node))

        # Create and return the scikit-learn pipeline
        return SklearnPipeline(steps=steps)
    
    def evaluate_pipeline(self, X, y):
        # Convert the custom pipeline to a scikit-learn pipeline
        skl_pipeline = self.build_sklearn_pipeline()
        
        # Fit the pipeline
        skl_pipeline_fitted = skl_pipeline.fit(X, y)
        
        # # Make predictions
        # predictions = skl_pipeline.predict(X)
        
        return skl_pipeline_fitted
