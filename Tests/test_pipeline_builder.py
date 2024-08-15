import sys
sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')
from pipeline_builder import PipelineBuilder
from pipeline import Pipeline
import numpy as np
import pandas as pd

# read a sample dataset
data = pd.read_csv("/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/3XOR_20features.csv")
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# convert X and y to numpy arrays
X = X.to_numpy()
y = y.to_numpy()

print(np.array(data.columns[:-1]))

# Create a random pipeline
pipeline = Pipeline(epi_pairs=[], epi_branches=[], selector_node=None, root_node=None, traits={}, max_feature_count=len(X)/10).generate_random_pipeline(rng = np.random.default_rng(42), header_list=np.array(data.columns[:-1]))

print("Branch Count:", pipeline.get_branch_count())
#print("Epi pairs:", pipeline.epi_pairs)
# Build a scikit-learn pipeline
pipeline_builder = PipelineBuilder(pipeline)
#sklearn_pipeline = pipeline_builder.build_sklearn_pipeline()

# # Evaluate the pipeline
# skl_pipeline_fitted = pipeline_builder.evaluate_pipeline(X, y) # fit the pipeline
skl_pipeline_fitted = pipeline_builder.fit(X, y)  # fit the pipeline
#skl_pipeline_fitted = pipeline_builder.evaluate_pipeline(X, y)

# get the traits
score, feature_count = pipeline_builder.score(X, y)

print("Pipeline built successfully!")
print(pipeline)
#print(sklearn_pipeline)
print(skl_pipeline_fitted)
print("Pipeline Score:", score)
print("Feature Count:", feature_count)

print("Epi SNP Pairs:", pipeline.epi_pairs)

# # for all the branches after selector, print the results of get_interactions()
# for branch in pipeline.epi_branches:
#     print(branch.get_feature_names_out())

final_epi_pairs = pipeline_builder.get_final_epi_pairs()
print("Final Epi Pairs:", final_epi_pairs)