import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import sparseoperations
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split


import os
import numpy as np
os.chdir('/home/matsumoton/common/git/geneSelect/Multiverse')
import sys; sys.path.append('/home/matsumoton/common/git/geneSelect/Multiverse')
from sparseBANN import sparseBANN
from nn_functions import Sigmoid, MSE, tanh, Relu, LeakyRelu, Linear
from utils import load_data, check_path
import datetime
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import pyarrow.feather as feather

random_state = 62
#getting rat gene dataset

file_out = feather.read_feather("/home/matsumoton/common/git/geneSelect/encoded_130k.feather")
file_out = file_out.reindex(sorted(file_out.columns), axis=1)

X_train, X_test, y_train, y_test = train_test_split(file_out.iloc[:,1:], file_out.iloc[:,0], test_size=0.20, random_state=random_state)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

del file_out

epoch = 200
no_training_epochs = 200

dataset_name = "ratbmi"

strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
no_hidden_neurons_layers = [29129, 645, 200]
loss = MSE

config = {
    'weight_decay' : [0.00001, 0.00002, 0.00004,0.0001,0.0002],
    'batch_size' : [32,64,128],
    'dropout_rate' : [0,0.1,0.2,0.3],
    'learning_rate': [0.001,0.005,0.01,0.02,0.04],
    'momentum' : [0.9,0.8],
    'epsilon' : [6,7,8,9,10,11,12,13,14,15], # set the sparsity level
    'zeta' : [0,0.1,0.2,0.3] # in [0..1]. It gives the percentage of unimportant connections which are removed and replaced with random ones after every epoch
}

import ray
ray.init()

X_train_ref = ray.put(X_train.values)
y_train_ref = ray.put(y_train)
X_test_ref = ray.put(X_test.values)
y_test_ref = ray.put(y_test)

#for i in range(50):

# Ray task
@ray.remote
def sdae(config):
    import numpy as np
    import sys; sys.path.append('/home/matsumoton/common/git/geneSelect/Multiverse')
    import sparseBANN
    import utils
    from nn_functions import Sigmoid, MSE, tanh, Relu, LeakyRelu, Linear
    #from utils import load_data, check_path

    weight_decay = np.random.choice(config['weight_decay'])
    batch_size=np.random.choice(config['batch_size'])
    dropout_rate=np.random.choice(config['dropout_rate'])
    learning_rate=np.random.choice(config['learning_rate'])
    momentum=np.random.choice(config['momentum'])
    epsilon = np.random.choice(config['epsilon'])
    zeta = np.random.choice(config['zeta'])


    filename = "/home/matsumoton/common/git/geneSelect/Multiverse/results/"+dataset_name+"/"+str(strtime)+"_seed"+str(random_state)+"/epoch_"+str(no_training_epochs)+"_epsilon_"+str(epsilon)+ \
        "_zeta_"+str(zeta)+"_wd_"+str(weight_decay)+ "_batch_"+str(batch_size)+"_dr_"+str(dropout_rate)+"_lr_"+str(learning_rate) +"_m_"+str(momentum) + "/"
    print(filename)
    utils.check_path(filename)
    print("*******************************************************************************")
    print("Dataset = ", dataset_name)
    print("epsilon = ", epsilon)
    print("zeta = ", zeta)
    start = time.time()


    # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
    set_mlp = sparseBANN((ray.get(X_train_ref).shape[1], no_hidden_neurons_layers[0], no_hidden_neurons_layers[1], no_hidden_neurons_layers[2], 1),
                        (Relu, Relu, Relu, Linear), epsilon=epsilon, classification = False, maximization = False)

    start_time = time.time()
    # train SET-MLP
    #ravel() removed on the y
    set_mlp.fit(ray.get(X_train_ref), ray.get(y_train_ref), ray.get(X_test_ref), ray.get(y_test_ref), loss=loss, epochs=no_training_epochs, batch_size=batch_size, learning_rate=learning_rate,
                momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropoutrate=dropout_rate, testing=True,
                save_filename=filename, monitor=False)

    step_time = time.time() - start_time
    print("\nTotal training time: ", step_time)

    # test SET-MLP
    accuracy, _ = set_mlp.predict(ray.get(X_test_ref), ray.get(y_test_ref), batch_size=100)

    print("\nAccuracy of the last epoch on the testing data: ", accuracy)


print(os.cpu_count())
iterations = 5
def run_remote(config):
    # Starting Ray
    start_time = time.time()
    #results = ray.get([sdae.remote(config) for _ in range(int(os.cpu_count()/5))])
    results = ray.get([sdae.remote(config) for _ in range(5)])
    duration = time.time() - start_time
    print('Remote execution time: {}'.format(duration))

for _ in range(iterations):
    run_remote(config)

ray.shutdown()