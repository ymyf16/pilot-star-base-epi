# test the rat qtl dataset
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')
from pipeline_builder import PipelineBuilder
from sklearn.pipeline import Pipeline as SklearnPipeline
import numpy as np
import pandas as pd
from evovler import EA
from epi_nodes import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
from itertools import combinations
import ray
from sklearn.model_selection import train_test_split


# read a sample dataset
data = pd.read_csv("/Users/ghosha/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/StarBASE-GP/Benchmarking/18qtl_pruned_BMIres.csv")

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# convert X and y to numpy arrays
X = X.to_numpy()
y = y.to_numpy()

# split the data into training and testing
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)


# holds all LO's (epistatic node types) we are going to evaluate
epis = {
    'cartesian': EpiCartesianNode,
    'xor': EpiXORNode,
    'pager': EpiPAGERNode,
    'rr': EpiRRNode,
    'rd': EpiRDNode,
    't': EpiTNode,
    'mod': EpiModNode,
    'dd': EpiDDNode,
    'm78': EpiM78Node
}

# Initialize a list to hold results for the CSV
results = []

# Iterate over all pairwise combinations of features
for snp1_idx, snp2_idx in combinations(range(X_train.shape[1]), 2):
    snp1_name = data.iloc[:, snp1_idx].name
    snp2_name = data.iloc[:, snp2_idx].name
    print(f"Processing SNPs {snp1_name} and {snp2_name}")

    # iterate over the epi node types and create sklearn pipeline for each pair of features
    for lo, epi in epis.items():
        steps = []
        
        # create the epi node with feature pair
        epi_node = epi(name=lo, snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_idx, snp2_pos=snp2_idx)

        # model = LinearRegression()
        # model.fit(epi_feature_train, y_train)
        # r2 = model.score(epi_feature_test, y_val)
        # print(f"SNP pair {snp1_name} and {snp2_name} with LO {lo} has R² score {r2}")

        steps.append((lo, epi_node))

        # add random forest regressor
        steps.append(('regressor', LinearRegression()))
        #steps.append(('regressor', RandomForestRegressor(random_state=42)))


        # create the pipeline
        skl_pipeline = SklearnPipeline(steps=steps)

        # Fit the pipeline
        skl_pipeline_fitted = skl_pipeline.fit(X_train, y_train)
    
        # get score
        r2 = skl_pipeline_fitted.score(X_val, y_val)

        # # Fit the pipeline
        # skl_pipeline_fitted = skl_pipeline.fit(X_train, y_train)

        # # get score
        # r2 = skl_pipeline_fitted.score(X_val, y_val)

        # add the results to the list
        results.append({
            'snp1': snp1_name,
            'snp2': snp2_name,
            'lo': lo,
            'r2_score': r2
        })

    print(results)
# Convert the results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('18qtl_epistatic_results.csv', index=False)

print("CSV file with best results has been created.")


# # test the rat qtl dataset
# import sys

# from sklearn.ensemble import RandomForestRegressor
# sys.path.append('/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source')
# from pipeline_builder import PipelineBuilder
# from sklearn.pipeline import Pipeline as SklearnPipeline
# import numpy as np
# import pandas as pd
# from evovler import EA
# from epi_nodes import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
# from itertools import combinations
# import ray

# # initialize ray
# ray.init(runtime_env={"working_dir": "/Users/ghosha/Documents/VSCode Projects/pilot-star-base-epi/Source"})

# # ea_config = {'seed': np.uint16(0),
# #                  'pop_size': np.uint16(100),
# #                  'epi_cnt_max': np.uint16(250),
# #                  'cores': 10,
# #                  'mut_ran_p':np.float32(.45),
# #                  'mut_smt_p': np.float32(.45),
# #                  'mut_non_p': np.float32(.1),
# #                  'smt_in_in_p': np.float32(.1),
# #                  'smt_in_out_p': np.float32(.45),
# #                  'smt_out_out_p': np.float32(.45)}

# # ea = EA(**ea_config)
# # # need to update the path to the data file
# # data_dir = '/Users/ghosha/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/StarBASE-GP/Benchmarking/18qtl_pruned_BMIres.csv'
# # ea.data_loader(data_dir)
# # ea.initialize_hubs(100)
# # ea.evolve(1)

# # read a sample dataset
# data = pd.read_csv("/Users/ghosha/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/StarBASE-GP/Benchmarking/18qtl_pruned_BMIres.csv")
# X = data.iloc[:, 0:-1]
# y = data.iloc[:, -1]

# # convert X and y to numpy arrays
# X = X.to_numpy()
# y = y.to_numpy()

# # split the data into training and testing
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# # # hold results
# # best_epi = ''
# # best_res = -1.0

# epi_objects = {
#     'cartesian': EpiCartesianNode,
#     'xor': EpiXORNode,
#     'pager': EpiPAGERNode,
#     'rr': EpiRRNode,
#     'rd': EpiRDNode,
#     't': EpiTNode,
#     'mod': EpiModNode,
#     'dd': EpiDDNode,
#     'm78': EpiM78Node
# }

# # Put the objects into the Ray object store
# epi_objects_id = ray.put(epi_objects)

# # Define a Ray remote function for evaluating a single pair of SNPs
# @ray.remote
# def evaluate_snp_pair(snp1_idx, snp2_idx, X_train, X_val, y_train, y_val, data, epi_objects):
#     snp1_name = data.columns[snp1_idx]
#     print("SNP1 Name:", snp1_name)
#     snp2_name = data.columns[snp2_idx]
#     print("SNP2 Name:", snp2_name)
    
#     best_epi = ''
#     best_res = -1.0

#     #epi_objects = ray.get(epi_objects_id)

#     # # Iterate over the epi node types and create sklearn pipeline for each pair of features
#     # for lo, epi in epi_objects.items():
#     #     #steps = []
#     #     epi_node = epi(name=lo, snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_idx, snp2_pos=snp2_idx)
#     #     #steps.append((lo, epi_node))
#     #     #steps.append(('regressor', RandomForestRegressor(random_state=42)))
#     #     epi_node.fit(X_train, y_train)
#     #     epi_feature_train = epi_node.transform(X_train)
#     #     epi_feature_test = epi_node.transform(X_val)

#     #     #skl_pipeline = SklearnPipeline(steps=steps)

#     #     # # Ensure you are using NumPy arrays for slicing
#     #     # X_train_subset = X_train[:, [snp1_idx, snp2_idx]]
#     #     # X_val_subset = X_val[:, [snp1_idx, snp2_idx]]

#     #     # # Convert X_train_subset and X_val_subset to DataFrames with proper column names
#     #     # X_train_subset = pd.DataFrame(X_train[:, [snp1_idx, snp2_idx]], columns=[snp1_name, snp2_name])
#     #     # X_val_subset = pd.DataFrame(X_val[:, [snp1_idx, snp2_idx]], columns=[snp1_name, snp2_name])


#     #     # Fit the pipeline
#     #     #skl_pipeline_fitted = skl_pipeline.fit(X_train, y_train)

#     #     # Get R² score
#     #     #r2 = skl_pipeline_fitted.score(X_val, y_val)
#     #     model = RandomForestRegressor(random_state=42)
#     #     model.fit(epi_feature_train, y_train)
#     #     r2 = model.score(epi_feature_test, y_val)

#     #     #print(f"SNP pair {snp1_name} and {snp2_name} with LO {lo} has R² score {r2}")

#     #     if r2 > best_res:
#     #         best_res = r2
#     #         best_epi = lo

#     # return {'snp1': snp1_name, 'snp2': snp2_name, 'best_epi': best_epi, 'best_res': best_res}
#     results_list = []

#     # Iterate over the epi node types and create sklearn pipeline for each pair of features
#     for lo, epi in epi_objects.items():
#         epi_node = epi(name=lo, snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_idx, snp2_pos=snp2_idx)
#         epi_node.fit(X_train, y_train)
#         epi_feature_train = epi_node.transform(X_train)
#         epi_feature_test = epi_node.transform(X_val)

        
#         # if the snp1_name is '1.281788173' and snp2_name is '3.136492861' and lo is xor then save the epi_feature_train and epi_feature_test
#         if snp1_name == '1.281788173' and snp2_name == '3.136492861' and lo == 'xor':
#             epi_feature_train_1 = epi_feature_train
#             epi_feature_test_1 = epi_feature_test
        
#             # save epi_feature_train_1 and epif_feature_test_1 as csv files
#             pd.DataFrame(epi_feature_train_1).to_csv('epi_feature_train_1.csv', index=False)
#             pd.DataFrame(epi_feature_test_1).to_csv('epi_feature_test_1.csv', index=False)

#             # save X_train and X_val as csv files
#             pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
#             pd.DataFrame(X_val).to_csv('X_val.csv', index=False)


#         model = RandomForestRegressor(random_state=42)
#         model.fit(epi_feature_train, y_train)
#         r2 = model.score(epi_feature_test, y_val)

#         # Append the results to the list for each epistatic interaction
#         results_list.append({
#             'snp1': snp1_name,
#             'snp2': snp2_name,
#             'lo': lo,
#             'r2_score': r2
#         })

#     return results_list


# # List to store Ray object references (futures)
# futures = []

# # Iterate over all pairwise combinations of features and submit Ray tasks
# for snp1_idx, snp2_idx in combinations(range(X_train.shape[1]), 2):
#     print(f"Processing SNPs {snp1_idx} and {snp2_idx}")
#     futures.append(evaluate_snp_pair.remote(snp1_idx, snp2_idx, X_train, X_val, y_train, y_val, data, epi_objects))

# # # Gather the results once all tasks are complete
# # results = ray.get(futures)

# # # Convert the results to a DataFrame and save to CSV
# # results_df = pd.DataFrame(results)
# # results_df.to_csv('best_epistatic_results_ray.csv', index=False)
# # Gather the results once all tasks are complete
# results = ray.get(futures)
# # Flatten the list of lists
# flattened_results = [item for sublist in results for item in sublist]

# # Convert the flattened results to a DataFrame and save to CSV
# results_df = pd.DataFrame(flattened_results)
# results_df.to_csv('all_epistatic_results_ray.csv', index=False)

# print("CSV file with best results has been created.")

# # Shutdown Ray
# ray.shutdown()