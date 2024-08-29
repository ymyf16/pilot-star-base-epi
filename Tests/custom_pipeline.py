import numpy as np
import pandas as pd
from evovler import EA
from epi_nodes import EpiCartesianNode, EpiXORNode, EpiPAGERNode, EpiRRNode, EpiRDNode, EpiTNode, EpiModNode, EpiDDNode, EpiM78Node
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        # Sequentially fit each step
        for name, step in self.steps:
            if hasattr(step, 'fit'):
                step.fit(X, y)
            if hasattr(step, 'transform'):
                X = step.transform(X)
        return self

    def predict(self, X):
        # Sequentially transform X through all steps
        for name, step in self.steps[:-1]:
            if hasattr(step, 'transform'):
                X = step.transform(X)
        # The last step is the regressor, which will do the prediction
        return self.steps[-1][1].predict(X)

    def score(self, X, y):
        # Get predictions and compute R2 score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
def ray_lo_eval(x_train, y_train, x_val, y_val, snp1_name: np.str_, snp2_name: np.str_, snp1_pos: np.uint32, snp2_pos: np.uint32):
    # hold results
    best_epi = ''
    best_res = -1.0

    # holds all lo's we are going to evaluate
    epis = {np.str_('cartesian'): EpiCartesianNode,
            np.str_('xor'): EpiXORNode,
            np.str_('pager'): EpiPAGERNode,
            np.str_('rr'): EpiRRNode,
            np.str_('rd'): EpiRDNode,
            np.str_('t'): EpiTNode,
            np.str_('mod'): EpiModNode,
            np.str_('dd'): EpiDDNode,
            np.str_('m78'): EpiM78Node
            }

    # iterate over the epi node types and create custom pipeline
    for lo, epi in epis.items():
        steps = []
        # create the epi node
        epi_node = epi(name=lo, snp1_name=snp1_name, snp2_name=snp2_name, snp1_pos=snp1_pos, snp2_pos=snp2_pos)
        steps.append((lo, epi_node))

        # add random forest regressor
        # TODO: Should we have this random_state always set to 0? or should we pass the seed?
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        steps.append(('regressor', regressor))

        # create the custom pipeline
        custom_pipeline = CustomPipeline(steps=steps)

        # Fit the pipeline
        custom_pipeline_fitted = custom_pipeline.fit(x_train, y_train)

        # get score
        r2 = custom_pipeline_fitted.score(x_val, y_val)

        # check if this is the best lo
        if r2 > best_res:
            best_res = r2
            best_epi = lo

    return np.float32(best_res), np.str_(best_epi)

# read the dataset

data = pd.read_csv("/Users/ghosha/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/StarBASE-GP/Benchmarking/18qtl_pruned_BMIres.csv")
# extract X
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

