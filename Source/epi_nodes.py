from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class EpiNode(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, name: np.str_, snp1_name: np.str_, snp2_name: np.str_, snp1_pos: np.uint32, snp2_pos: np.uint32):
        # self.rng = rng
        # self.header_list = header_list
        self.name = name
        self.snp1_name = snp1_name
        self.snp2_name = snp2_name
        self.snp1_pos = snp1_pos
        self.snp2_pos = snp2_pos

        self.epi_feature = None # store the output of process_data, the actual epistatic feature
        # self.logical_operator = None
        self.mapping = None
        self.epi_feature_name = None
        self.fit_flag = False

    def print_epi_feature(self):
        print(self.epi_feature)

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

    def get_interaction(self) -> np.str_:
        return np.str_(f"{self.get_snp1_name}_{self.get_lo}_{self.get_snp2_name}")

    @abstractmethod
    def get_lo(self) -> np.str_:
        pass

    def get_snp1_name(self) -> np.str_:
        return self.snp1_name

    def get_snp2_name(self) -> np.str_:
        return self.snp2_name

class EpiCartesianNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = np.multiply(snp1, snp2)
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("cartesian")

class EpiXORNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("xor")

class EpiRRNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("rr")

class EpiRDNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("rd")

class EpiTNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("t")

class EpiModNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("mod")

class EpiDDNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("dd")

class EpiM78Node(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1, snp2 = X[:, self.snp1_index], X[:, self.snp2_index]

        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

    def get_lo(self) -> np.str_:
        return np.str_("m78")

# Need to change the PAGER implementation due to training/testing data
class EpiPAGERNode(EpiNode):
    def process_data(self, X, y=None):
        # get the snp columns from the input data
        snp1_train, snp2_train = X[:, self.snp1_index], X[:, self.snp2_index]

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

    def get_lo(self) -> np.str_:
        return np.str_("pager")