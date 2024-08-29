# This class is a wrapper for the scikit-learn library. It provides a set of nodes (regressors and feature selectors) that can be used in the pipeline.
# Will contain a abstract class and then the various regressors and feature selectors will be implemented as subclasses of this abstract class.

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFwe, SelectFromModel, SequentialFeatureSelector, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from typeguard import typechecked
from typing import List, Tuple, Dict, Set

rng_t = np.random.Generator
name_t = np.str_

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

    @abstractmethod
    def mutate(self, rng):
        pass

##########################################################################################
########################## the feature selector classes ##################################
##########################################################################################

# variance threshold
@typechecked
class VarianceThresholdNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('VarianceThreshold')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'threshold': np.float32(rng.uniform(low=0.001, high=0.05))} # high is set low to allow models to have a chance to learn
        else:
            # make sure params is correct
            assert 'threshold' in params
            assert len(params) == 1
            assert isinstance(params['threshold'], np.float32)
            self.params = params

        self.selector = VarianceThreshold(threshold=self.params['threshold'])

    def fit(self, X, y=None):
        self.selector.fit(X)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng: rng_t):
        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.01))

        # check if the threshold is going to be negative
        if self.params['threshold'] + shift < np.float32(0.0):
            self.params['threshold'] = np.float32(0.0)
        # check if the threshold is going to be greater than 1
        elif self.params['threshold'] + shift > np.float32(1.0):
            self.params['threshold'] = np.float32(1.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['threshold'] = self.params['threshold'] + shift

        self.selector = VarianceThreshold(threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

# select percentile
class SelectPercentileNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SelectPercentile')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            # TODO: are the functions for the score_func correct? @attri
            self.params = {'percentile': np.int8(rng.integers(low=25, high=75)), 'score_func': rng.choice([f_regression, lambda X, y: mutual_info_regression(X, y, random_state=seed)])}
        else:
            # make sure params is correct
            assert 'percentile' in params
            assert 'score_func' in params
            assert len(params) == 2
            assert isinstance(params['percentile'], np.int8)
            assert isinstance(params['score_func'], np.ufunc)
            self.params = params

        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])
        self.seed = seed

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng: rng_t):
        # maginitude we are shfiting the percentile
        shift = np.int8(rng.integers(low=-10, high=10, endpoint=True))

        # check if the percentile is going to be less than 1
        if self.params['percentile'] + shift < np.int8(1):
            self.params['percentile'] = np.int8(1)
        # check if the percentile is going to be greater than 100
        elif self.params['percentile'] + shift > np.int8(100):
            self.params['percentile'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['percentile'] = self.params['percentile'] + shift

        # get new score_func
        self.params['score_func'] = rng.choice([f_regression, lambda X, y: mutual_info_regression(X, y, random_state=self.seed)])

        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

# select fwe
class SelectFweNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SelectFwe')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            # todo: test alpha level to get a good range for it
            self.params = {'alpha': np.float32(rng.uniform(low=0.8, high=1.0)), 'score_func': rng.choice([f_regression, lambda X, y: mutual_info_regression(X, y, random_state=seed)])}
        else:
            # make sure params is correct
            assert 'alpha' in params
            assert 'score_func' in params
            assert len(params) == 2
            assert isinstance(params['alpha'], np.float32)
            assert isinstance(params['score_func'], np.ufunc)
            self.params = params

        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])
        self.seed = seed

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng: rng_t):
        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.01))

        # check if the alpha is going to be negative
        if self.params['alpha'] + shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than .99
        elif self.params['alpha'] + shift > np.float32(0.99):
            self.params['alpha'] = np.float32(0.99)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + shift

        # get new score_func
        self.params['score_func'] = rng.choice([f_regression, lambda X, y: mutual_info_regression(X, y, random_state=self.seed)])

        # new selector configuration
        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

# select from model using L1-based feature selection (model is lasso regression)
class SelectFromModelLasso(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SelectFromLasso')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': Lasso(random_state=seed), 'threshold': rng.choice([name_t('mean'), name_t('median')])}
        else:
            # make sure params is correct
            assert 'estimator' in params
            assert 'threshold' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], Lasso)
            assert isinstance(params['threshold'], np.str_)
            self.params = params

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])
        self.seed = seed

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng):
        # randomly select threshold
        self.params['threshold'] = rng.choice([name_t('mean'), name_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self,):
        return self.selector.get_support().sum()

# select from model using tree-based feature selection (model is ExtraTreesRegressor)
class SelectFromModelTree(ScikitNode, TransformerMixin):
    # todo: do we wanna also evolve the hyperparameters of the ExtraTreesRegressor?
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SelectFromExtraTrees')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': ExtraTreesRegressor(random_state=seed), 'threshold': rng.choice([name_t('mean'), name_t('median')])}
        else:
            # make sure params is correct
            assert 'estimator' in params
            assert 'threshold' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], ExtraTreesRegressor)
            assert isinstance(params['threshold'], np.str_)
            self.params = params

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])
        self.seed = seed

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng):
        # randomly select threshold
        self.params['threshold'] = rng.choice([name_t('mean'), name_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

# sequential feature selector, model = RandomForestRegressor, need to discuss with the team
class SequentialFeatureSelectorNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SequentialFeatureSelectorRF')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': RandomForestRegressor(random_state=seed), 'tol': np.float32(rng.uniform(low=1e-5, high=0.5))}
        else:
            assert 'estimator' in params
            assert 'tol' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], RandomForestRegressor)
            assert isinstance(params['tol'], np.float32)
            self.params = params

        # todo: left cv=2 for speed ups, but we can change it to something else?
        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'], cv=2)

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng):
        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.01))

        # check if the tol is going to be less than 1e-5
        if self.params['tol'] + shift < np.float32(1e-5):
            self.params['tol'] = np.float32(1e-5)
        # check if the tol is going to be greater than 0.5
        elif self.params['tol'] + shift > np.float32(0.5):
            self.params['tol'] = np.float32(0.5)
        # if neither of the above, then we can just add the shift
        else:
            self.params['tol'] = self.params['tol'] + shift

        # new selector configuration
        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'], cv=2)

    def get_feature_count(self):
        return self.selector.get_support().sum()

##########################################################################################
############################ the regressor classes #######################################
##########################################################################################

# Linear regression
class LinearRegressionNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('LinearRegression')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'fit_intercept': rng.choice([True, False])}
        else:
            assert len(params) == 1
            assert 'fit_intercept' in params
            self.params = params

        self.regressor = LinearRegression(fit_intercept=self.params['fit_intercept'])

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng: rng_t):
        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])

        # new regressor configuration
        self.regressor = LinearRegression(fit_intercept=self.params['fit_intercept'])

# ElasticNet regression
class ElasticNetNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('LinearRegression')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            # l1_ratio should not be 0 or 1, use Lasso or Ridge instead
            self.params = {'alpha': np.float32(rng.uniform(low=0.0001, high=10.00)),
                          'l1_ratio': np.float32(rng.uniform(low=0.02, high=0.98)),
                          'fit_intercept': rng.choice([True, False]),
                          'selection': rng.choice([name_t('cyclic'), name_t('random')]),
                          'random_state': seed}
        else:
            assert len(params) == 5
            assert 'alpha' in params
            assert 'l1_ratio' in params
            assert 'fit_intercept' in params
            assert 'selection' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = ElasticNet(**self.params)

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get a random number from a normal distribution
        alpha_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the alpha is going to be less than 0.0001
        if self.params['alpha'] + alpha_shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than 10
        elif self.params['alpha'] + alpha_shift > np.float32(10.0):
            self.params['alpha'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + alpha_shift

        # get a random number from a normal distribution
        l1_ratio_shift = np.float32(rng.normal(loc=0.0, scale=0.03))

        # check if the l1_ratio is going to be less than 0.02
        if self.params['l1_ratio'] + l1_ratio_shift < np.float32(0.02):
            self.params['l1_ratio'] = np.float32(0.02)
        # check if the l1_ratio is going to be greater than 0.98
        elif self.params['l1_ratio'] + l1_ratio_shift > np.float32(0.98):
            self.params['l1_ratio'] = np.float32(0.98)
        # if neither of the above, then we can just add the shift
        else:
            self.params['l1_ratio'] = self.params['l1_ratio'] + l1_ratio_shift

        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])
        # randomly pick selection
        self.params['selection'] = rng.choice([name_t('cyclic'), name_t('random')])

        # new regressor configuration
        self.regressor = ElasticNet(**self.params)

# SGD regression
class SGDRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SGDRegressor')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'loss': rng.choice(['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),
                           'penalty': rng.choice(['l2','l1','elasticnet',None]),
                           'alpha': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'l1_ratio': np.float32(rng.uniform(low=0.02, high=0.98)),
                           'fit_intercept': rng.choice([True, False]),
                           'epsilon': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'learning_rate': rng.choice(['constant','optimal','invscaling','adaptive']),
                           'eta0': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'random_state': seed}
        else:
            assert len(params) == 9
            assert 'alpha' in params
            assert 'l1_ratio' in params
            assert 'epsilon' in params
            assert 'loss' in params
            assert 'eta0' in params
            assert 'random_state' in params
            assert 'fit_intercept' in params
            assert 'penalty' in params
            assert 'learning_rate' in params
            self.params = params

        self.regressor = SGDRegressor(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get a random number from a normal distribution
        alpha_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the alpha is going to be less than 0.0001
        if self.params['alpha'] + alpha_shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than 10
        elif self.params['alpha'] + alpha_shift > np.float32(10.0):
            self.params['alpha'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + alpha_shift

        # get a random number from a normal distribution
        l1_ratio_shift = np.float32(rng.normal(loc=0.0, scale=0.03))

        # check if the l1_ratio is going to be less than 0.02
        if self.params['l1_ratio'] + l1_ratio_shift < np.float32(0.02):
            self.params['l1_ratio'] = np.float32(0.02)
        # check if the l1_ratio is going to be greater than 0.98
        elif self.params['l1_ratio'] + l1_ratio_shift > np.float32(0.98):
            self.params['l1_ratio'] = np.float32(0.98)
        # if neither of the above, then we can just add the shift
        else:
            self.params['l1_ratio'] = self.params['l1_ratio'] + l1_ratio_shift

        # get a random number from a normal distribution
        epsilon_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the epsilon is going to be less than 0.0001
        if self.params['epsilon'] + epsilon_shift < np.float32(0.0001):
            self.params['epsilon'] = np.float32(0.0001)
        # check if the epsilon is going to be greater than 10
        elif self.params['epsilon'] + epsilon_shift > np.float32(10.0):
            self.params['epsilon'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['epsilon'] = self.params['epsilon'] + epsilon_shift

        # get a random number from a normal distribution
        eta0_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the eta0 is going to be less than 0.0001
        if self.params['eta0'] + eta0_shift < np.float32(0.0001):
            self.params['eta0'] = np.float32(0.0001)
        # check if the eta0 is going to be greater than 10
        elif self.params['eta0'] + eta0_shift > np.float32(10.0):
            self.params['eta0'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['eta0'] = self.params['eta0'] + eta0_shift

        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])
        # randomly pick loss
        self.params['loss'] = rng.choice(['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'])
        # randomly pick penalty
        self.params['penalty'] = rng.choice(['l2','l1','elasticnet',None])
        # randomly pick learning_rate
        self.params['learning_rate'] = rng.choice(['constant','optimal','invscaling','adaptive'])

        # new regressor configuration
        self.regressor = SGDRegressor(**self.params)

# SVR regression
class SVRNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng: rng_t,
                 seed: int = None,
                 params: Dict = {},
                 name: name_t = name_t('SVR')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'kernel': rng.choice(['linear', 'poly', 'rbf', 'sigmoid']),
                           'degree': np.uint8(rng.choice([1,2,3])),
                           'gamma': rng.choice(['scale', 'auto']),
                           'C': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'epsilon': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'tol': np.float32(rng.uniform(low=1e-5, high=0.5))}
        else:
            assert len(params) == 6
            assert 'kernel' in params
            assert 'degree' in params
            assert 'gamma' in params
            assert 'C' in params
            assert 'epsilon' in params
            assert 'tol' in params
            self.params = params

        self.regressor = SVR(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get a random number from a normal distribution
        C_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the C is going to be less than 0.0001
        if self.params['C'] + C_shift < np.float32(0.0001):
            self.params['C'] = np.float32(0.0001)
        # check if the C is going to be greater than 10
        elif self.params['C'] + C_shift > np.float32(10.0):
            self.params['C'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['C'] = self.params['C'] + C_shift

        # get a random number from a normal distribution
        epsilon_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the epsilon is going to be less than 0.0001
        if self.params['epsilon'] + epsilon_shift < np.float32(0.0001):
            self.params['epsilon'] = np.float32(0.0001)
        # check if the epsilon is going to be greater than 10
        elif self.params['epsilon'] + epsilon_shift > np.float32(10.0):
            self.params['epsilon'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['epsilon'] = self.params['epsilon'] + epsilon_shift

        # get a random number from a normal distribution
        tol_shift = np.float32(rng.normal(loc=0.0, scale=0.01))

        # check if the tol is going to be less than 1e-5
        if self.params['tol'] + tol_shift < np.float32(1e-5):
            self.params['tol'] = np.float32(1e-5)
        # check if the tol is going to be greater than 0.5
        elif self.params['tol'] + tol_shift > np.float32(0.5):
            self.params['tol'] = np.float32(0.5)
        # if neither of the above, then we can just add the shift
        else:
            self.params['tol'] = self.params['tol'] + tol_shift

        # randomly pick kernel
        self.params['kernel'] = rng.choice(['linear', 'poly', 'rbf', 'sigmoid'])
        # randomly pick degree
        self.params['degree'] = np.uint8(rng.choice([1,2,3]))
        # randomly pick gamma
        self.params['gamma'] = rng.choice(['scale', 'auto'])

        # new regressor configuration
        self.regressor = SVR(**self.params)

# Decision tree regression
class DecisionTreeRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng: rng_t,
                seed: int = None,
                params: Dict = {},
                name: name_t = name_t('DecisionTreeRegressor')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'criteria': rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                           'splitter': rng.choice(['best', 'random']),
                           'max_features': rng.choice([None, 'sqrt', 'log2']),
                           'max_depth': np.uint8(rng.integers(1,10)),
                           'min_samples_split': np.uint8(rng.integers(1,20)),
                           'min_samples_leaf': np.uint8(rng.integers(1,20)),
                           'random_state': seed}

        else:
            assert len(params) == 7
            assert 'criteria' in params
            assert 'splitter' in params
            assert 'max_features' in params
            assert 'max_depth' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = DecisionTreeRegressor(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # shift for max_depth up or down
        max_depth_shift = np.uint8(rng.choice([-1, 1]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.uint8(1):
            self.params['max_depth'] = np.uint8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.uint8(10):
            self.params['max_depth'] = np.uint8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.uint8(1):
            self.params['min_samples_split'] = np.uint8(1)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.uint8(20):
            self.params['min_samples_split'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.uint8(1):
            self.params['min_samples_leaf'] = np.uint8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.uint8(20):
            self.params['min_samples_leaf'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick criteria
        self.params['criteria'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        # randomly pick splitter
        self.params['splitter'] = rng.choice(['best', 'random'])
        # randomly pick max_features
        self.params['max_features'] = rng.choice([None, 'sqrt', 'log2'])

        # new regressor configuration
        self.regressor = DecisionTreeRegressor(**self.params)

# Random forest regression
class RandomForestRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng: rng_t,
                seed: int = None,
                params: Dict = {},
                name: name_t = name_t('RandomForestRegressor')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'n_estimators': np.int8(rng.integers(10,100)),
                           'criterion': rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                           'splitter': rng.choice(['best', 'random']),
                           'max_depth': np.uint8(rng.integers(1,10)),
                           'max_features': rng.choice([None, 'sqrt', 'log2']),
                           'min_samples_split': np.uint8(rng.integers(1,20)),
                           'min_samples_leaf': np.uint8(rng.integers(1,20)),
                           'random_state': seed}
        else:
            assert len(params) == 8
            assert 'n_estimators' in params
            assert 'criterion' in params
            assert 'splitter' in params
            assert 'max_depth' in params
            assert 'max_features' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = RandomForestRegressor(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get a random number from a uniform distribution
        n_estimators_shift = np.int8(rng.integers(-10, 10))

        # check if the n_estimators is going to be less than 10
        if self.params['n_estimators'] + n_estimators_shift < np.int8(10):
            self.params['n_estimators'] = np.int8(10)
        # check if the n_estimators is going to be greater than 100
        elif self.params['n_estimators'] + n_estimators_shift > np.int8(100):
            self.params['n_estimators'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['n_estimators'] = self.params['n_estimators'] + n_estimators_shift

        # shift for max_depth up or down
        max_depth_shift = np.uint8(rng.choice([-1, 1]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.uint8(1):
            self.params['max_depth'] = np.uint8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.uint8(10):
            self.params['max_depth'] = np.uint8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.uint8(1):
            self.params['min_samples_split'] = np.uint8(1)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.uint8(20):
            self.params['min_samples_split'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.uint8(1):
            self.params['min_samples_leaf'] = np.uint8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.uint8(20):
            self.params['min_samples_leaf'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick criteria
        self.params['criteria'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        # randomly pick splitter
        self.params['splitter'] = rng.choice(['best', 'random'])
        # randomly pick max_features
        self.params['max_features'] = rng.choice([None, 'sqrt', 'log2'])

        # new regressor configuration
        self.regressor = RandomForestRegressor(**self.params)

# Gradient boosting regression
class GradientBoostingRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng: rng_t,
                seed: int = None,
                params: Dict = {},
                name: name_t = name_t('RandomForestRegressor')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'loss': rng.choice(['squared_error', 'absolute_error', 'huber', 'quantile']),
                           'learning_rate': rng.uniform(low=1e-3, high=1.0),
                           'n_estimators': np.uint8(rng.integers(10,100)),
                           'criterion': rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                           'max_depth': rng.integers(1,10),
                           'min_samples_split': rng.integers(2,20),
                           'min_samples_leaf': rng.integers(1,20),
                           'random_state': seed}
        else:
            assert len(params) == 8
            assert 'loss' in params
            assert 'learning_rate' in params
            assert 'n_estimators' in params
            assert 'criterion' in params
            assert 'max_depth' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        # initialize the regressor
        self.regressor = GradientBoostingRegressor(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get random number from a normal distribution
        learning_rate_shift = rng.normal(loc=0.0, scale=0.1)

        # check if the learning_rate is going to be less than 1e-3
        if self.params['learning_rate'] + learning_rate_shift < 1e-3:
            self.params['learning_rate'] = 1e-3
        # check if the learning_rate is going to be greater than 10.0
        elif self.params['learning_rate'] + learning_rate_shift > 10.0:
            self.params['learning_rate'] = 10.0
        # if neither of the above, then we can just add the shift
        else:
            self.params['learning_rate'] = self.params['learning_rate'] + learning_rate_shift

        # shift for n_estimators up or down
        n_estimators_shift = np.int8(rng.choice([-10, 10]))

        # check if the n_estimators is going to be less than 10
        if self.params['n_estimators'] + n_estimators_shift < np.int8(10):
            self.params['n_estimators'] = np.int8(10)
        # check if the n_estimators is going to be greater than 100
        elif self.params['n_estimators'] + n_estimators_shift > np.int8(100):
            self.params['n_estimators'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['n_estimators'] = self.params['n_estimators'] + n_estimators_shift

        # shift for max_depth up or down
        max_depth_shift = np.uint8(rng.choice([-1, 1]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.uint8(1):
            self.params['max_depth'] = np.uint8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.uint8(10):
            self.params['max_depth'] = np.uint8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.uint8(1):
            self.params['min_samples_split'] = np.uint8(1)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.uint8(20):
            self.params['min_samples_split'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.uint8(rng.choice([-1, 1]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.uint8(1):
            self.params['min_samples_leaf'] = np.uint8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.uint8(20):
            self.params['min_samples_leaf'] = np.uint8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick loss
        self.params['loss'] = rng.choice(['squared_error', 'absolute_error', 'huber', 'quantile'])
        # randomly pick criterion
        self.params['criterion'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])

        # new regressor configuration
        self.regressor = GradientBoostingRegressor(**self.params)

# Multi-layer perceptron regression
class MLPRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng: rng_t,
                seed: int = None,
                params: Dict = {},
                name: name_t = name_t('RandomForestRegressor')):
        super().__init__(name)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'hidden_layer_sizes': (rng.integers(10,100),),
                           'activation': rng.choice(['identity', 'logistic', 'tanh', 'relu']),
                           'solver': rng.choice(['lbfgs', 'sgd', 'adam']),
                           'alpha': rng.uniform(low = 1e-7, high = 1e-1),
                           'learning_rate': rng.choice(['constant', 'invscaling', 'adaptive']),
                           'learning_rate_init': rng.uniform(low = 1e-4, high = 1e-1 ),
                           'random_state': seed}
        else:
            assert len(params) == 7
            assert 'hidden_layer_sizes' in params
            assert 'activation' in params
            assert 'solver' in params
            assert 'alpha' in params
            assert 'learning_rate' in params
            assert 'learning_rate_init' in params
            assert 'random_state' in params
            self.params = params

        # initialize the regressor
        self.regressor = MLPRegressor(**self.params)
        self.seed = seed

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # update hidden_layer_sizes
        self.params['hidden_layer_sizes'] = (rng.integers(10,100),)

        # get random number from a normal distribution
        alpha_shift = rng.normal(loc=0.0, scale=0.01)

        # check if the alpha is going to be less than 1e-7
        if self.params['alpha'] + alpha_shift < 1e-7:
            self.params['alpha'] = 1e-7
        # check if the alpha is going to be greater than 10.0
        elif self.params['alpha'] + alpha_shift > 10.0:
            self.params['alpha'] = 10.0
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + alpha_shift

        # get random number from a normal distribution
        learning_rate_init_shift = rng.normal(loc=0.0, scale=0.01)

        # check if the learning_rate_init is going to be less than 1e-4
        if self.params['learning_rate_init'] + learning_rate_init_shift < 1e-4:
            self.params['learning_rate_init'] = 1e-4
        # check if the learning_rate_init is going to be greater than 10.0
        elif self.params['learning_rate_init'] + learning_rate_init_shift > 10.0:
            self.params['learning_rate_init'] = 10.0
        # if neither of the above, then we can just add the shift
        else:
            self.params['learning_rate_init'] = self.params['learning_rate_init'] + learning_rate_init_shift

        # randomly pick activation
        self.params['activation'] = rng.choice(['identity', 'logistic', 'tanh', 'relu'])
        # randomly pick solver
        self.params['solver'] = rng.choice(['lbfgs', 'sgd', 'adam'])
        # randomly pick learning_rate
        self.params['learning_rate'] = rng.choice(['constant', 'invscaling', 'adaptive'])

        # new regressor configuration
        self.regressor = MLPRegressor(**self.params)