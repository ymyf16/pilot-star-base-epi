import openml
import os
import pickle
import time
import numpy as np

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import ray

# Parallelized functions
@ray.remote
def evaluate_model(model_name, model, X, y):
    scores = cross_val_score(model, X, y, cv=10, error_score="raise")
    return model_name, scores.mean(), scores.std()

class EA:
    def __init__(self):
        print('Initializing EA')
        context = ray.init(num_cpus=10, include_dashboard=True)
        print('dashboard:', context.dashboard_url)
        print('use this if you want to see the dashboard')

    def __del__(self):
        print('Deleting EA')
        ray.shutdown()

    #https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
    def load_task(self, task_id, preprocess=True):

        cached_data_path = f"./{task_id}_{preprocess}.pkl"
        print(cached_data_path)
        if os.path.exists(cached_data_path):
            d = pickle.load(open(cached_data_path, "rb"))
            X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
            return X_train, y_train, X_test, y_test

    def gen_models(self, ):
        models = {}

        for i in range(10):
            models[f"RandomForestClassifier_{i}"] = RandomForestClassifier(n_estimators=100, random_state=i)
            models[f"SVC_{i}"] = SVC(gamma='auto', random_state=i)
            models[f"KNeighborsClassifier_{i}"] = KNeighborsClassifier(n_neighbors=5)
            models[f"DecisionTreeClassifier_{i}"] = DecisionTreeClassifier(random_state=i)

        return models

    def convert_seconds(self, seconds):
        days = seconds / (24 * 3600)
        hours = seconds / 3600
        minutes = seconds / 60

        return f"{days:.2f} days, {hours:.2f} hours, {minutes:.2f} minutes, and {seconds:.2f} seconds"

    def run(self):
        # get models
        models = self.gen_models()

        # get task & load it to ray
        X_train, y_train, X_test, y_test = self.load_task(167184, preprocess=True)

        X_train_ray, y_train_ray, X_test_ray, y_test_ray = ray.put(X_train), ray.put(y_train), ray.put(X_test), ray.put(y_test)

        # Create Ray tasks for each model
        futures = [evaluate_model.remote(name, model, X_train_ray, y_train_ray) for name, model in models.items()]

        # Gather the results
        start = time.time()
        results = ray.get(futures)
        end = time.time()  - start

        # Print the results
        for result in results:
            model_name, mean_score, std_score = result
            print(f"Model: {model_name}, Mean CV Score: {mean_score:.4f}, Std CV Score: {std_score:.4f}")

        print(f"Time taken (seconds -> time_adjusted):", self.convert_seconds(end))

def main():

    # get system stats
    print('cpu cnt:',os.cpu_count())

    # Initialize class
    ea = EA()
    ea.run()

    print('Done!')

if __name__ == "__main__":
    main()