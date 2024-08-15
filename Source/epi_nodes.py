# # make an abstract class called EpiNodes a list variable called interacting_features, a variable called name, a function called print_data and a function called process_data

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
    
    @abstractmethod
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        # The output of process_data should be stored in self.epi_feature
        return self.epi_feature.reshape(-1, 1)  # Ensure the output is 2D for scikit-learn compatibility

    @abstractmethod
    def get_feature_names_out(self): # to get the name of the epistatic feature
        pass

    @abstractmethod
    def predict(self, X):
        pass

class EpiCartesianNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        
        epi_feature = np.multiply(snp1, snp2)
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "CART"
        self.fit_flag = True
        
        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self

    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = np.multiply(snp1, snp2)
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_CART_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiXORNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "XOR"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_XOR_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiRRNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "RR"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_RR_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiRDNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "RD"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_RD_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiTNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "T"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_T_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiModNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "MOD"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_MOD_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiDDNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "DD"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_dd_{self.snp2_name}"
        return self.epi_feature_name
    

class EpiM78Node(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]

        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        self.epi_feature = epi_feature.astype(np.float32)
        self.logical_operator = "M78"
        self.fit_flag = True

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1, snp2 = X[:, snp1_index], X[:, snp2_index]
        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        return epi_feature.astype(np.float32)

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_M78_{self.snp2_name}"
        return self.epi_feature_name
    

# Need to change the PAGER implementation due to training/testing data
class EpiPAGERNode(EpiNode):
    def fit(self, X, y=None):
        # randomly select two snp names from the header list and get the corresponding columns
        snp1_name, snp2_name = self.rng.choice(list(self.header_list), 2, replace=False)
        self.snp1_name, self.snp2_name = snp1_name, snp2_name
        snp1_index = np.where(self.header_list == snp1_name)[0][0]
        snp2_index = np.where(self.header_list == snp2_name)[0][0]
        snp1_train, snp2_train = X[:, snp1_index], X[:, snp2_index]

        # snp1_test, snp2_test = snp1_train, snp2_train # need to change this to actual test data

        train_data = pd.DataFrame({
            'snp1': snp1_train,
            'snp2': snp2_train,
            'Phenotype': y
        })
        
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

        train_data = pd.merge(train_data, geno_aggregations_train, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.epi_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)

        #self.epi_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5).astype(np.float32)
        self.logical_operator = "PAGER"

        # if self.snp1_name is not None and self.snp2_name is not None:
        #     print(f"Epi pairs in fit: {self.snp1_name}, {self.snp2_name}")
        # else:
        #     print(f"Error: SNP names not set correctly.")
        return self


    def predict(self, X):
        if self.mapping is None:
            raise ValueError("Mapping not set. Please fit the model first.")
        
        snp1_index = np.where(self.header_list == self.snp1_name)[0][0]
        snp2_index = np.where(self.header_list == self.snp2_name)[0][0]
        snp1_test, snp2_test = X[:, snp1_index], X[:, snp2_index]
        test_data = pd.DataFrame({
            'snp1': snp1_test,
            'snp2': snp2_test
        })
        test_data = pd.merge(test_data, self.mapping, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_test = test_data['normalized_rel_dist'].values
        epi_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5).astype(np.float32)
        return epi_feature

    def get_feature_names_out(self):
        self.epi_feature_name = f"{self.snp1_name}_PAGER_{self.snp2_name}"
        return self.epi_feature_name
