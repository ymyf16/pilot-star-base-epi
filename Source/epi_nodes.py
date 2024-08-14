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
