from skmultilearn.ext import Meka
from random import choices
import random
from joblib import Parallel, delayed
from joblib import effective_n_jobs
import joblib
import numpy as np
import re


class EccClassifier:
    """Ensemble Chain Classifiers
    """

    def __init__(self,
                 meka_classifier="meka.classifiers.multilabel.CC",
                 weka_classifier="weka.classifiers.trees.J48",
                 meka_classpath="",
                 java_command="/usr/bin/java",
                 n_estimators=10,
                 n_jobs=6,
                 ratio=0.9
                 ):
        self.meka_classifier = meka_classifier
        self.meka_classpath = meka_classpath
        self.weka_classifier = weka_classifier
        self.java_command = java_command
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.ratio = ratio
        self.starts = []
        self.per_jobs = []
        self.estimators = []
        self.N_class = 0

    def super_constructor(self):
        meka = Meka(
            meka_classifier=self.meka_classifier,  # Classifier Chains
            weka_classifier=self.weka_classifier, # Basic tree classifier
            meka_classpath=self.meka_classpath,  # obtained via download_meka
            java_command=self.java_command  # path to java executable
        )
        return meka

    def bootstrap(self, X, y, state):
        assert X.shape[0] == y.shape[0]

        random.seed(state)
        N = round(X.shape[0] * self.ratio)
        # With replacements
        sample_indices = choices(range(N), k=N)

        return X[sample_indices], y[sample_indices], sample_indices

    def bagging_train(self, X, y, n_estimators_range):
        estimators = []

        for n in n_estimators_range:
            X_train, y_train, _ = self.bootstrap(X, y, n)
            estimators.append(self.super_constructor().fit(X_train, y_train))

        return estimators

    # from _bagging.py of sklearn
    def _partition_estimators(self, n_estimators, n_jobs):
        """Private function used to partition estimators between jobs."""
        # Compute the number of jobs
        n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

        # Partition estimators between jobs
        n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
        n_estimators_per_job[: n_estimators % n_jobs] += 1
        starts = np.cumsum(n_estimators_per_job)

        return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

    def prepareParallel(self):
        self.n_jobs, self.per_jobs, self.starts = self._partition_estimators(self.n_estimators, self.n_jobs)

    def train(self, X_train, y_train):

        self.N_class = y_train.shape[1]

        self.prepareParallel()
        models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.bagging_train)(X_train, y_train, range(self.starts[i], self.starts[i + 1]))
            for i in range(self.n_jobs))

        estimators = []
        for model_list in models:
            estimators += model_list

        self.estimators = estimators

    def boostrap_agg(self, estimators, X):
        predictions = [estimator.predict(X) for estimator in estimators]

        return predictions

    def predict_prob(self, X):

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.boostrap_agg)(self.estimators[self.starts[i]:self.starts[i + 1]], X)
            for i in range(self.n_jobs))

        predictions = []
        for prediction_list in results:
            predictions += prediction_list

        class_probs = []

        for c in range(self.N_class):
            class_prediction = np.array([prediction[:, c].toarray() for prediction in predictions]).squeeze(
                2).transpose()
            class_prob = np.mean(class_prediction, axis=1)
            class_probs.append(class_prob)

        return np.array(class_probs).transpose()

    def binarize_results(self, class_probs, threshold):

        class_binarizes = []

        for class_prob in class_probs:
            class_binarize = np.where(class_prob > threshold, 1, 0)
            class_binarizes.append(class_binarize)

        return np.array(class_binarizes)

    def predict(self, X, threshold):

        class_probs = self.predict_prob(X)
        class_binarizes = self.binarize_results(class_probs, threshold)

        return class_probs, class_binarizes

    def save_ecc(self, path):

        if len(self.estimators) == 0:
            print("Models are not trained yet.")
            return

        for i in range(len(self.estimators)):
            filename = path + "/estimator_" + str(i) + ".joblib"
            joblib.dump(self.estimators[i], filename)

    def load_ecc(self, path, n_estimators):

        loaded_models = []
        for i in range(n_estimators):
            filename = path + "/estimator_" + str(i) + ".joblib"
            estimator = joblib.load(filename)
            loaded_models.append(estimator)
            if self.N_class == 0:
                self.extractNclass(estimator)

        self.estimators = loaded_models
        self.prepareParallel()

    def extractNclass(self, estimator):

        train_data = estimator.train_data_
        search_pattern = "(?<=traindata: -C -)[0-9]+"
        N_class = re.findall(search_pattern, train_data)
        try:
            self.N_class = int(N_class[0])
        except ValueError:
            print("Number of class cannot be parsed from training data")
            exit(1)
        except:
            exit(2)
