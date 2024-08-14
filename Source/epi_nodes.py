# # make an abstract class called EpiNodes a list variable called interacting_features, a variable called name, a function called print_data and a function called process_data

# from abc import ABC, abstractmethod
# import numpy as np
# from itertools import product
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# class EpiNode(ABC):
#     interacting_features = []
#     name = ""
#     epi_feature = np.ndarray
#     logical_operator = "" 
    
#     def __init__(self, name, interacting_features=None, logical_operator=None, epi_feature=None):
#         self.name = name
#         if interacting_features is not None:
#             self.interacting_features = [np.array(f) for f in interacting_features]
#         else:
#             self.interacting_features = []
#         self.logical_operator = logical_operator
#         self.epi_feature = epi_feature
    
#     @abstractmethod
#     def print_data(self):
#         pass
    
#     @abstractmethod
#     def process_data(self):
#         pass

# class EpiCartesianNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         epi_feature = np.multiply(snp1, snp2)
#         # Min-Max Normalization of the epistatic feature
#         epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
#         self.epi_feature = epi_feature
#         self.logical_operator = "cartesian"
#         return epi_feature.astype(np.float32)
    
# class EpiXORNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         epi_feature = (snp1%2 + snp2%2)%2
#         self.epi_feature = epi_feature
#         self.logical_operator = "xor"
#         return epi_feature.astype(np.float32)
    
# class EpiRRNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if both snp1 and snp2 are 2, otherwise it is 0
#         epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
#         self.epi_feature = epi_feature
#         self.logical_operator = "rr"
#         return epi_feature.astype(np.float32)
    
# class EpiRDNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if (snp1 is 2 and snp2 is 1) or (both snp1 and snp2 are 2), otherwise it is 0
#         epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
#         self.epi_feature = epi_feature
#         self.logical_operator = "rd"
#         return epi_feature.astype(np.float32)
    
# class EpiTNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if:
#         # (snp1 is 2 and snp2 is 1) or 
#         # (both snp1 and snp2 are 2) or 
#         # (snp1 is 1 and snp2 is 2), otherwise it is 0
#         epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | 
#                                ((snp1 == 2) & (snp2 == 2)) | 
#                                ((snp1 == 1) & (snp2 == 2)), 1, 0)
#         self.epi_feature = epi_feature
#         self.logical_operator = "t"
#         return epi_feature.astype(np.float32)
    
# class EpiModNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if:
#         # (snp1 is 2 and snp2 is 1) or 
#         # (both snp1 and snp2 are 2) or 
#         # (snp1 is 1 and snp2 is 2) or
#         # (snp1 is 2 and snp2 is 0), otherwise it is 0
#         epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
#         self.epi_feature = epi_feature
#         self.logical_operator = "mod"
#         return epi_feature.astype(np.float32)
    
# class EpiDDNode(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if:
#         # (snp1 is 1 and snp2 is 1) or 
#         # (both snp1 and snp2 are 2) or 
#         # (snp1 is 1 and snp2 is 2) or
#         # (snp1 is 2 and snp2 is 1), otherwise it is 0
#         condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
#         condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
#         condition3 = (snp1 == 2) & (snp2 == 2)
#         epi_feature = condition1 | condition2 | condition3
#         self.epi_feature = epi_feature
#         self.logical_operator = "dd"
#         return epi_feature.astype(np.float32)
    
# class EpiM78Node(EpiNode):
#     def __init__(self, name, interacting_features=None):
#         super().__init__(name, interacting_features)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
    
#     def process_data(self):
#         snp1 = self.interacting_features[0]
#         snp2 = self.interacting_features[1]
#         # The epistatic term is 1 if:
#         # (snp1 is 0 and snp2 is 2) or 
#         # (snp1 is 1 and snp2 is 2) or 
#         # (snp1 is 2 and snp2 is 0) or
#         # (snp1 is 2 and snp2 is 1), otherwise it is 0
#         condition1 = (snp1 == 0) & (snp2 == 2)
#         condition2 = (snp1 == 1) & (snp2 == 2)
#         condition3 = (snp1 == 2) & (snp2 == 0)
#         condition4 = (snp1 == 2) & (snp2 == 1)
#         epi_feature = condition1 | condition2 | condition3 | condition4
#         self.epi_feature = epi_feature
#         self.logical_operator = "m78"
#         return epi_feature.astype(np.float32)
    
# class EpiPAGERNode(EpiNode):
#     def __init__(self, name, interacting_features= None, train_interacting_features = None, train_phenotype = None):
#         super().__init__(name, interacting_features)
#         self.train_interacting_features = [np.array(f) for f in train_interacting_features]
#         self.train_phenotype = np.array(train_phenotype)
    
#     def print_data(self):
#         print(self.name, self.interacting_features)
#         print("Train interacting_features:", self.train_interacting_features)
#         print("Test interacting_features:", self.interacting_features)
#         print("Train Phenotype:", self.train_phenotype)
    
#     def process_data(self):
#         snp1_train = self.train_interacting_features[0] # testing genotype SNP1
#         snp2_train = self.train_interacting_features[1] # testing genotype SNP2
#         snp1_test = self.interacting_features[0] # testing genotype SNP1
#         snp2_test = self.interacting_features[1] # testing genotype SNP2

#         # Combine SNP arrays into a single DataFrame for train and test sets - using pandas dataframes for easier manipulation
#         train_data = pd.DataFrame({
#             'snp1': snp1_train,
#             'snp2': snp2_train,
#             'Phenotype': self.train_phenotype
#         })
#         test_data = pd.DataFrame({
#             'snp1': snp1_test,
#             'snp2': snp2_test
#         })

      
#         # Calculate mean phenotype per SNP combination for the training set
#         geno_aggregations_train = train_data.groupby(['snp1', 'snp2']).agg(
#             mean_phenotype=('Phenotype', 'mean')
#         ).reset_index()

#         # Calculate relative distance from the anchor mean phenotype
#         #anchor_mean = geno_aggregations_train['mean_phenotype'].min()
#         anchor_mean = geno_aggregations_train['mean_phenotype'].iloc[
#             geno_aggregations_train['mean_phenotype'].first_valid_index()]
#         geno_aggregations_train['rel_dist'] = geno_aggregations_train['mean_phenotype'] - anchor_mean

#         # Normalize relative distance
#         scaler_train = MinMaxScaler()
#         geno_aggregations_train['normalized_rel_dist'] = scaler_train.fit_transform(
#             geno_aggregations_train['rel_dist'].values.reshape(-1, 1)
#         )


#         # Map the combinations to normalized_rel_dist values for test set
#         test_data = pd.merge(test_data, geno_aggregations_train, on=['snp1', 'snp2'], how='left')
#         pager_encoded_interactions_test = test_data['normalized_rel_dist'].values

#         # Replace NaN values with 0.5 in the test interactions
#         epi_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5)
#         self.epi_feature = epi_feature
#         self.logical_operator = "pager"
#         return epi_feature.astype(np.float32)
           

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class EpiNode(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, name, rng, header_list): # take in the header of the data and randomly select two columns 
        self.name = name
        self.rng = rng
        self.header_list = header_list
        self.snp1_name = None
        self.snp2_name = None
        self.epi_feature = None # store the output of process_data, the actual epistatic feature
        self.logical_operator = None
        self.mapping = None 
        self.epi_feature_name = None
        self.fit_flag = False
    
    def fit(self, X, y=None):
        # Abstract method for fitting process
        self.process_data(X, y)
        return self

    def transform(self, X):
        # The output of process_data should be stored in self.epi_feature
        return self.epi_feature.reshape(-1, 1)  # Ensure the output is 2D for scikit-learn compatibility
    
    @abstractmethod
    def process_data(self, X, y=None):
        pass

    @abstractmethod
    def get_interaction(self):
        pass

class EpiCartesianNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        
        epi_feature = np.multiply(snp1, snp2)
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "cartesian"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _cart_ {self.snp2_name}"
        return self.epi_feature_name

class EpiXORNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "xor"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _xor_ {self.snp2_name}"
        return self.epi_feature_name

class EpiRRNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "rr"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _rr_ {self.snp2_name}"
        return self.epi_feature_name

class EpiRDNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "rd"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _rd_ {self.snp2_name}"
        return self.epi_feature_name

class EpiTNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "t"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _t_ {self.snp2_name}"
        return self.epi_feature_name

class EpiModNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "mod"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _mod_ {self.snp2_name}"
        return self.epi_feature_name

class EpiDDNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "dd"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _dd_ {self.snp2_name}"
        return self.epi_feature_name

class EpiM78Node(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "m78"
        self.fit_flag = True

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _m78_ {self.snp2_name}"
        return self.epi_feature_name

# Need to change the PAGER implementation due to training/testing data
class EpiPAGERNode(EpiNode):
    def process_data(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1_train, snp2_train = X[:, snp1_index], X[:, snp2_index]

        # snp1_test, snp2_test = snp1_train, snp2_train # need to change this to actual test data

        train_data = pd.DataFrame({
            'snp1': snp1_train,
            'snp2': snp2_train,
            'Phenotype': y
        })
        # test_data = pd.DataFrame({
        #     'snp1': snp1_test,
        #     'snp2': snp2_test
        # })

        geno_aggregations_train = train_data.groupby(['snp1', 'snp2']).agg(
            mean_phenotype=('Phenotype', 'mean')
        ).reset_index()

        anchor_mean = geno_aggregations_train['mean_phenotype'].iloc[
            geno_aggregations_train['mean_phenotype'].first_valid_index()]
        geno_aggregations_train['rel_dist'] = geno_aggregations_train['mean_phenotype'] - anchor_mean

        scaler_train = MinMaxScaler()
        geno_aggregations_train['normalized_rel_dist'] = scaler_train.fit_transform(
            geno_aggregations_train['rel_dist'].values.reshape(-1, 1)
        )

        self.mapping = geno_aggregations_train['normalized_rel_dist'] # store the PAGER values generated by using the train data

        # test_data = pd.merge(test_data, geno_aggregations_train, on=['snp1', 'snp2'], how='left')
        # pager_encoded_interactions_test = test_data['normalized_rel_dist'].values

        train_data = pd.merge(train_data, geno_aggregations_train, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.epi_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)

        #self.epi_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5).astype(np.float32)
        self.logical_operator = "pager"

    def get_interaction(self):
        self.epi_feature_name = f"{self.snp1_name} _pager_ {self.snp2_name}"
        return self.epi_feature_name

        