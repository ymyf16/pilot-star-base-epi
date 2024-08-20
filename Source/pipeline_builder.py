# function to read in a Pipeline object and convert it to a scikit learn pipeline object

from sklearn.pipeline import Pipeline as SklearnPipeline
from pipeline import Pipeline # our custom pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict
import numpy as np
from typeguard import typechecked

@typechecked # for debugging purposes
class PipelineBuilder:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.fitted_pipeline = None

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
        self.fitted_pipeline = skl_pipeline_fitted

        # Collect SNP pairs from each epi node after fitting [(snp1, lo, snp2),...]
        self.pipeline.epi_pairs = [(epi_node.snp1_name, epi_node.logical_operator, epi_node.snp2_name) for epi_node in self.pipeline.epi_branches]

        # Debugging information
        # print("Epi pairs:", self.pipeline.epi_pairs)

        return skl_pipeline_fitted

    def fit(self, X, y):
        # fit the pipeline
        skl_pipeline_fitted = self.evaluate_pipeline(X, y)
        return skl_pipeline_fitted

    def predict(self, X):
        # assume that the sklearn pipeline is already built and fitted
        if self.fitted_pipeline is None:
            raise ValueError("Pipeline is not fitted yet.")

        # Make predictions
        predictions = self.predict(X)

        return predictions

    def score(self, X, y):
        # assume that the sklearn pipeline is already built and fitted
        if self.fitted_pipeline is None:
            exit("Pipeline is not fitted yet.", 0)

        # Score the pipeline
        r2 = self.fitted_pipeline.score(X, y)

        # get the total number of features in the pipeline
        feature_count = self.pipeline.get_feature_count()
        final_fetaure_names = self.pipeline.get_selector_node().selector.get_feature_names_out()
        # print("Final Feature Names:", final_fetaure_names)

        # set the pipeline's traits
        self.pipeline.traits = {"r2_score": r2, "feature_count": feature_count}

        # print("Pipeline Traits:", self.pipeline.traits)


        return r2, feature_count

    def get_final_epi_pairs(self):
        if self.fitted_pipeline is None:
            exit("Pipeline is not fitted yet.", 0)

        # Access the selector node from the fitted pipeline
        selector_node = self.fitted_pipeline.named_steps.get('selector', None)

        if selector_node is None:
            exit("Selector node is not found in the pipeline.", 0)

        # Get selected feature indices from the selector
        selected_indices = selector_node.selector.get_support(indices=True)
        print("Selected Indices:", selected_indices)
        print("No of branches after selection:", len(selected_indices))

        # Retrieve final epi pairs based on selected indices
        final_epi_pairs = []
        for index in selected_indices:
            final_epi_pairs.append(self.pipeline.epi_pairs[index])

        return final_epi_pairs