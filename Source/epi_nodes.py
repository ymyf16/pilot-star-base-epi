# # make an abstract class called EpiNodes a list variable called interacting_features, a variable called name, a function called print_data and a function called process_data

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class EpiNode(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, name: np.str_, snp1_name: np.str_, snp2_name: np.str_, snp1_pos: np.uint32, snp2_pos: np.uint32):
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
    
    @abstractmethod
    def fit(self, X, y=None):
        pass

    def print_epi_feature(self):    
        print(self.epi_feature)

    def transform(self, X):
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

    @abstractmethod
    def predict(self, X):
        pass

class EpiCartesianNode(EpiNode):
    def fit(self, X, y=None):
        # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # Cartesian product for training data
        epi_feature = np.multiply(snp1, snp2)
        
        # Normalize the feature
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        
        # Store the result in self.epi_feature for training data
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
        
        # Mark the node as fitted
        self.fit_flag = True
        
        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # Cartesian product for the given data (train or validation)
        epi_feature = np.multiply(snp1, snp2)
        
        # Normalize the feature
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # Cartesian product
        epi_feature = np.multiply(snp1, snp2)
        # normalize the feature
        epi_feature = (epi_feature - np.min(epi_feature)) / (np.max(epi_feature) - np.min(epi_feature))
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("cartesian")
    

class EpiXORNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # XOR operation
        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # XOR operation for the given data (train or validation)
        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
             snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # XOR operation
        epi_feature = (snp1 % 2 + snp2 % 2) % 2
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("xor")
    

class EpiRRNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # RR operation
        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # RR operation for the given data (train or validation)
        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # RR operation
        epi_feature = np.where((snp1 == 2) & (snp2 == 2), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("rr")
    

class EpiRDNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # RD operation
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # RD operation for the given data (train or validation)
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # RD operation
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("rd")

class EpiTNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # T operation
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # T operation for the given data (train or validation)
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # T operation
        epi_feature = np.where(((snp1 == 2) & (snp2 == 1)) | ((snp1 == 2) & (snp2 == 2)) | ((snp1 == 1) & (snp2 == 2)), 1, 0)
        # cast to np.float32
        self.epi_feature = epi_feature.astype(np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("t")
    

class EpiModNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # Mod operation
        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # Mod operation for the given data (train or validation)
        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # Mod operation
        epi_feature = np.isin(snp1, [2]) & np.isin(snp2, [0, 1, 2]) | ((snp1 == 1) & (snp2 == 2))
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
     
        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("mod")
    

class EpiDDNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # DD operation
        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # DD operation for the given data (train or validation)
        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # DD operation
        condition1 = np.isin(snp1, [1, 2]) & (snp2 == 1)
        condition2 = (snp1 == 1) & np.isin(snp2, [1, 2])
        condition3 = (snp1 == 2) & (snp2 == 2)
        epi_feature = condition1 | condition2 | condition3
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("dd")

    

class EpiM78Node(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # M78 operation
        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)
        # we have fitted this node
        self.fit_flag = True

        return self
    
    def transform(self, X):
        # Always recompute the epistatic feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        # M78 operation for the given data (train or validation)
        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        
        # Return the computed feature for this dataset
        return np.array(epi_feature.astype(np.float32), dtype=np.float32).reshape(-1, 1)


    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1, snp2 = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1, snp2 = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # M78 operation
        condition1 = (snp1 == 0) & (snp2 == 2)
        condition2 = (snp1 == 1) & (snp2 == 2)
        condition3 = (snp1 == 2) & (snp2 == 0)
        condition4 = (snp1 == 2) & (snp2 == 1)
        epi_feature = condition1 | condition2 | condition3 | condition4
        # cast to np.float32
        self.epi_feature = np.array(epi_feature.astype(np.float32), dtype=np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("m78")
    

# Need to change the PAGER implementation due to training/testing data
class EpiPAGERNode(EpiNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1_train, snp2_train = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1_train, snp2_train = X[:, self.snp1_pos], X[:, self.snp2_pos]
        # create a DataFrame with the snp columns and the phenotype
        train_data = pd.DataFrame({
            'snp1': snp1_train,
            'snp2': snp2_train,
            'Phenotype': y
        })

        # calculate the mean phenotype for each snp pair
        geno_aggregations_train = train_data.groupby(['snp1', 'snp2']).agg(
            mean_phenotype=('Phenotype', 'mean')
        ).reset_index()

        # calculate the relative distance of the mean phenotype to the anchor
        anchor_mean = geno_aggregations_train['mean_phenotype'].iloc[
            geno_aggregations_train['mean_phenotype'].first_valid_index()]
        geno_aggregations_train['rel_dist'] = geno_aggregations_train['mean_phenotype'] - anchor_mean

        # normalize the relative distance
        scaler_train = MinMaxScaler()
        geno_aggregations_train['normalized_rel_dist'] = scaler_train.fit_transform(
            geno_aggregations_train['rel_dist'].values.reshape(-1, 1)
        )

        # store the PAGER values generated by using the train data
        self.mapping = geno_aggregations_train[['snp1', 'snp2', 'normalized_rel_dist']] # store the PAGER values generated by using the train data

        train_data = pd.merge(train_data, geno_aggregations_train, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.epi_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)
        self.fit_flag = True

        return self
    
    def transform(self, X):
        if self.fit_flag == False or self.mapping is None:
            raise ValueError("Mapping not set. Please fit the model first.")
        
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1_train, snp2_train = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1_train, snp2_train = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        train_data = pd.DataFrame({
            'snp1': snp1_train,
            'snp2': snp2_train
        })
        train_data = pd.merge(train_data, self.mapping, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.epi_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)

        return self.epi_feature.reshape(-1, 1)


    def predict(self, X):
        if self.fit_flag == False or self.mapping is None:
            raise ValueError("Mapping not set. Please fit the model first.")
        
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp1_test, snp2_test = X.iloc[:, self.snp1_pos], X.iloc[:, self.snp2_pos]
        else:
            snp1_test, snp2_test = X[:, self.snp1_pos], X[:, self.snp2_pos]
        
        test_data = pd.DataFrame({
            'snp1': snp1_test,
            'snp2': snp2_test
        })
        test_data = pd.merge(test_data, self.mapping, on=['snp1', 'snp2'], how='left')
        pager_encoded_interactions_test = test_data['normalized_rel_dist'].values
        self.epi_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5).astype(np.float32)

        return self.epi_feature.reshape(-1, 1)

    def get_lo(self) -> np.str_:
        return np.str_("pager")