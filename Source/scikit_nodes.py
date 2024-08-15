# This class is a wrapper for the scikit-learn library. It provides a set of nodes (regressors and feature selectors) that can be used in the pipeline.
# Will contain a abstract class and then the various regressors and feature selectors will be implemented as subclasses of this abstract class.

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFwe, SelectFromModel, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



class ScikitNode(BaseEstimator, ABC):
    def __init__(self, name, features=None):
        self.name = name
        self.features = features if features is not None else []

    @abstractmethod
    def fit(self, X, y=None):
        pass
    
    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
# check_probablity function definition
def check_probability(self, probability):
    return self.rng.uniform(0, 1) < probability
    


############################################### the feature selector classes ################################################

# variance threshold
class VarianceThresholdNode(ScikitNode, TransformerMixin):
    def __init__(self, name='VarianceThreshold', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'threshold': rng.uniform(low=0.0, high=1.0)}
        self.selector = VarianceThreshold(threshold=self.params['threshold'])
    
    def fit(self, X, y=None):
        self.selector.fit(X)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5): # ask Jose about the probability and discuss about normal/uniform distribution
            self.params['threshold'] = self.params['threshold'] + rng.uniform(low=0.0, high=1.0) # default loc=0.0, scale=1.0
        else:
            self.params['threshold'] = self.params['threshold'] - rng.uniform(low=0.0, high=1.0)

        self.selector = VarianceThreshold(threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
    
# select percentile
class SelectPercentileNode(ScikitNode, TransformerMixin):
    def __init__(self, name='SelectPercentile', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'percentile': rng.integers(low=0, high=100), 'score_func': f_regression} # should we change low for percentile to 10? 
        # if params['score_func'] == 'f_regression':
        #     params['score_func'] = f_regression
        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])       

    def fit(self, X, y):
        self.selector.fit(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['percentile'] = self.params['percentile'] + rng.integers(low=0, high=100 - self.params['percentile'])
        else:
            self.params['percentile'] = self.params['percentile'] - rng.integers(low=0, high=self.params['percentile'])
        
        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])

    def get_feature_count(self):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
    
# select fwe
class SelectFweNode(ScikitNode, TransformerMixin):
    def __init__(self, name='SelectFwe', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'alpha': rng.uniform(low=0.0, high=0.05), 'score_func': 'f_regression'}
        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

    def fit(self, X, y):
        self.selector.fit(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['alpha'] = self.params['alpha'] + rng.uniform(low=0.0, high=0.05 - self.params['alpha'])
        else:
            self.params['alpha'] = self.params['alpha'] - rng.uniform(low=0.0, high=self.params['alpha'])

        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

    def get_feature_count(self):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
 
# select from model using L1-based feature selection (model is lasso regression)
class SelectFromModelLasso(ScikitNode, TransformerMixin):
    def __init__(self, name='SelectFromLasso', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'estimator': Lasso(), 'threshold': 'mean'}
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])
    
    def fit(self, X, y):
        self.selector.fit(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['threshold'] = 'mean'
        else:
            self.params['threshold'] = 'median'

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self,):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
    
# select from model using tree-based feature selection (model is ExtraTreesRegressor)
class SelectFromModelTree(ScikitNode, TransformerMixin):
    def __init__(self, name='SelectFromExtraTrees', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'estimator': ExtraTreesRegressor(), 'threshold': 'mean'}
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])
    
    def fit(self, X, y):
        self.selector.fit(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['threshold'] = 'mean'
        else:
            self.params['threshold'] = 'median'

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
    
# sequential feature selector, model = RandomForestRegressor, need to discuss with the team
class SequentialFeatureSelectorNode(ScikitNode, TransformerMixin):
    def __init__(self, name='SequentialFeatureSelectorRF', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params ={'estimator': RandomForestRegressor(), 'tol': rng.uniform(low = 0, high = 1)} # need to discuss the hyperparameters
        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'])
    
    def fit(self, X, y):
        self.selector.fit(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['tol'] = self.params['tol'] + rng.uniform(low = 0, high = 1)
        else:
            self.params['tol'] = self.params['tol'] - rng.uniform(low = 0, high = 1)

        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'])

    def get_feature_count(self):
        return self.selector.get_support().sum()
    
    def get_feature_names_out(self):
        return self.features[self.selector.get_support()]
    
############################################### the regressor classes ################################################

# Linear regression
class LinearRegressionNode(ScikitNode, RegressorMixin):
    def __init__(self, name='LinearRegression', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = None
        self.regressor = LinearRegression()
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        pass
    
# ElasticNet regression
class ElasticNetNode(ScikitNode, RegressorMixin):
    def __init__(self, name='ElasticNet', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'alpha': rng.uniform(low = 0, high = 1), 'l1_ratio': rng.uniform(low = 0, high = 1), 'random_state': rng.integers(0,100)}
        self.regressor = ElasticNet(alpha=self.params['alpha'], l1_ratio=self.params['l1_ratio'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        if rng.check_probablity(0.5):
            self.params['alpha'] = self.params['alpha'] + rng.uniform(low = 0, high = 1)
            self.params['l1_ratio'] = self.params['l1_ratio'] + rng.uniform(low = 0, high = 1)
        else:
            self.params['alpha'] = self.params['alpha'] - rng.uniform(low = 0, high = 1)
            self.params['l1_ratio'] = self.params['l1_ratio'] - rng.uniform(low = 0, high = 1)
        
        self.regressor = ElasticNet(alpha=self.params['alpha'], l1_ratio=self.params['l1_ratio'], random_state=self.params['random_state'] + 1)

# SGD regression
class SGDRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self, name='SGDRegressor', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'alpha': rng.uniform(low = 1e-7, high = 1e-3), 'l1_ratio': rng.uniform(low = 0, high = 1), 'epsilon': rng.uniform(low = 0, high = 1), 'loss': rng.choice(['squared_loss', 'huber', 'epsilon_insensitive']), 'eta0': rng.uniform(low = 0, high = 1), 'random_state': rng.integers(0,100)} 
        self.regressor = SGDRegressor(alpha=self.params['alpha'], l1_ratio=self.params['l1_ratio'], epsilon=self.params['epsilon'], loss=self.params['loss'], eta0=self.params['eta0'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['alpha', 'l1_ratio', 'epsilon', 'loss', 'eta0'])
        if param == 'alpha':
            self.params['alpha'] = self.params['alpha'] + rng.uniform(low = 1e-7, high = 1e-3)
        elif param == 'l1_ratio':
            self.params['l1_ratio'] = self.params['l1_ratio'] + rng.uniform(low = 0, high = 1)
        elif param == 'epsilon':
            self.params['epsilon'] = self.params['epsilon'] + rng.uniform(low = 0, high = 1)
        elif param == 'loss':
            self.params['loss'] = rng.choice(['squared_loss', 'huber', 'epsilon_insensitive'])
        else:
            self.params['eta0'] = self.params['eta0'] + rng.uniform(low = 0, high = 1)

        self.regressor = SGDRegressor(alpha=self.params['alpha'], l1_ratio=self.params['l1_ratio'], epsilon=self.params['epsilon'], loss=self.params['loss'], eta0=self.params['eta0'], random_state=self.params['random_state'] + 1)
    
# SVR regression
class SVRNode(ScikitNode, RegressorMixin):
    def __init__(self, name='SVR', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'kernel': rng.choice(['linear', 'poly', 'rbf', 'sigmoid']), 'degree': rng.integers(1,5), 'gamma': rng.uniform(low = 0, high = 10), 'C': rng.uniform(low = 0.01, high = 25), 'epsilon': rng.uniform(low = 0, high = 1), 'tol': rng.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
        self.regressor = SVR(kernel=self.params['kernel'], degree=self.params['degree'], gamma=self.params['gamma'], C=self.params['C'], epsilon=self.params['epsilon'], tol=self.params['tol'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['kernel', 'degree', 'gamma', 'C', 'epsilon', 'tol'])
        if param == 'kernel':
            self.params['kernel'] = rng.choice(['linear', 'poly', 'rbf', 'sigmoid'])
        elif param == 'degree':
            self.params['degree'] = rng.integers(1,5)
        elif param == 'gamma':
            self.params['gamma'] = self.params['gamma'] + rng.uniform(low = 0, high = 10)
        elif param == 'C':
            self.params['C'] = self.params['C'] + rng.uniform(low = 0.01, high = 25)
        elif param == 'epsilon':
            self.params['epsilon'] = self.params['epsilon'] + rng.uniform(low = 0, high = 1)
        else:
            self.params['tol'] = rng.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

        self.regressor = SVR(kernel=self.params['kernel'], degree=self.params['degree'], gamma=self.params['gamma'], C=self.params['C'], epsilon=self.params['epsilon'], tol=self.params['tol'])
    
# Decision tree regression
class DecisionTreeRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self, name='DecisionTreeRegressor', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'max_depth': rng.integers(1,10), 'min_samples_split': rng.integers(1,20), 'min_samples_leaf': rng.integers(1,20), 'random_state': rng.integers(0,100)}
        self.regressor = DecisionTreeRegressor(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['max_depth', 'min_samples_split', 'min_samples_leaf'])
        if param == 'max_depth':
            self.params['max_depth'] = self.params['max_depth'] + rng.integers(1,5)
        elif param == 'min_samples_split':
            self.params['min_samples_split'] = self.params['min_samples_split'] + rng.integers(1,10)
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + rng.integers(1,10)

        self.regressor = DecisionTreeRegressor(max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], random_state=self.params['random_state'] + 1)
    
# Random forest regression
class RandomForestRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self, name='RandomForestRegressor', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'n_estimators': rng.integers(1,100), 'max_features': rng.uniform(low = 0.05, high = 1), 'criterion': rng.choice(['mse', 'mae', "friedman_mse"]), 'min_samples_split': rng.integers(2,21), 'min_samples_leaf': rng.integers(1,21), 'random_state': rng.integers(0,100)}    
        self.regressor = RandomForestRegressor(n_estimators=self.params['n_estimators'], max_features=self.params['max_features'], criterion=self.params['criterion'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['n_estimators', 'max_features', 'criterion', 'min_samples_split', 'min_samples_leaf', 'random_state'])
        if param == 'n_estimators':
            self.params['n_estimators'] = self.params['n_estimators'] + rng.integers(0,10)
        elif param == 'max_features':
            self.params['max_features'] = self.params['max_features'] + rng.uniform(low = 0.05, high = 1)
        elif param == 'criterion':
            self.params['criterion'] = rng.choice(['mse', 'mae', "friedman_mse"])
        elif param == 'min_samples_split':
            self.params['min_samples_split'] = self.params['min_samples_split'] + rng.integers(2,10)
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + rng.integers(1,10)
        
        self.regressor = RandomForestRegressor(n_estimators=self.params['n_estimators'], max_features=self.params['max_features'], criterion=self.params['criterion'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], random_state=self.params['random_state'] + 1)
    
# Gradient boosting regression
class GradientBoostingRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self, name='GradientBoostingRegressor', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'n_estimators': rng.integers(0,100), 'max_depth': rng.integers(1,10), 'min_samples_split': rng.integers(2,21), 'min_samples_leaf': rng.integers(1,21), 'loss': rng.choice(["ls", "lad", "huber", "quantile"]), 'learning_rate': rng.choice([1e-3, 1e-2, 1e-1, 0.5, 1.]), 'random_state': rng.integers(0,100)}
        self.regressor = GradientBoostingRegressor(n_estimators=self.params['n_estimators'], max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], loss=self.params['loss'], learning_rate=self.params['learning_rate'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'loss', 'learning_rate'])
        if param == 'n_estimators':
            self.params['n_estimators'] = self.params['n_estimators'] + rng.integers(0,10)
        elif param == 'max_depth':
            self.params['max_depth'] = self.params['max_depth'] + rng.integers(1,5)
        elif param == 'min_samples_split':
            self.params['min_samples_split'] = self.params['min_samples_split'] + rng.integers(2,10)
        elif param == 'min_samples_leaf':
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + rng.integers(1,10)
        elif param == 'loss':
            self.params['loss'] = rng.choice(["ls", "lad", "huber", "quantile"])
        else:
            self.params['learning_rate'] = rng.choice([1e-3, 1e-2, 1e-1, 0.5, 1.])
        
        self.regressor = GradientBoostingRegressor(n_estimators=self.params['n_estimators'], max_depth=self.params['max_depth'], min_samples_split=self.params['min_samples_split'], min_samples_leaf=self.params['min_samples_leaf'], loss=self.params['loss'], learning_rate=self.params['learning_rate'], random_state=self.params['random_state'] + 1)

# Multi-layer perceptron regression
class MLPRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self, name='MLPRegressor', params = None, rng = None):
        super().__init__(name)
        self.rng = rng
        self.params = {'n_hidden_layers': rng.integers(1,5), 'n_nodes_per_layer': rng.integers(16, 512), 'activation': rng.choice(['identity', 'logistic', 'tanh', 'relu']), 'solver': rng.choice(['lbfgs', 'sgd', 'adam']), 'alpha': rng.uniform(low = 1e-7, high = 1e-1), 'learning_rate': rng.choice(['constant', 'invscaling', 'adaptive']), 'learning_rate_init': rng.uniform(low = 1e-4, high = 1e-1 ), 'random_state': rng.integers(0,100)}
        self.regressor = MLPRegressor(hidden_layer_sizes=(self.params['n_hidden_layers']*self.params['n_nodes_per_layer'],), activation=self.params['activation'], solver=self.params['solver'], alpha=self.params['alpha'], learning_rate=self.params['learning_rate'], learning_rate_init=self.params['learning_rate_init'], random_state=self.params['random_state'])
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)
    
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)
    
    def mutate(self, rng):
        # randomly select a parameter to mutate
        param = rng.choice(['n_hidden_layers', 'n_nodes_per_layer', 'activation', 'solver', 'alpha', 'learning_rate', 'learning_rate_init'])
        if param == 'n_hidden_layers':
            self.params['n_hidden_layers'] = self.params['n_hidden_layers'] + rng.integers(1,5)
        elif param == 'n_nodes_per_layer':
            self.params['n_nodes_per_layer'] = self.params['n_nodes_per_layer'] + rng.integers(16, 128)
        elif param == 'activation':
            self.params['activation'] = rng.choice(['identity', 'logistic', 'tanh', 'relu'])
        elif param == 'solver':
            self.params['solver'] = rng.choice(['lbfgs', 'sgd', 'adam'])
        elif param == 'alpha':
            self.params['alpha'] = self.params['alpha'] + rng.uniform(low = 1e-7, high = 1e-1)
        elif param == 'learning_rate':
            self.params['learning_rate'] = rng.choice(['constant', 'invscaling', 'adaptive'])
        else:
            self.params['learning_rate_init'] = self.params['learning_rate_init'] + rng.uniform(low = 1e-4, high = 1e-1)

        self.regressor = MLPRegressor(hidden_layer_sizes=(self.params['n_hidden_layers']*self.params['n_nodes_per_layer'],), activation=self.params['activation'], solver=self.params['solver'], alpha=self.params['alpha'], learning_rate=self.params['learning_rate'], learning_rate_init=self.params['learning_rate_init'], random_state=self.params['random_state'] + 1)