from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_from_arff
import pandas as pd
from ECCModel import *
from ECCMeka import *
import tempfile
import os
from joblib.parallel import Parallel
import json


class singleGen:

    def __init__(self, model, N_class, compound_list, arff_file, folds, uncovered_reaction_file, n_jobs=6):
        self.model = model
        self.N_class = N_class
        self.compound_list = compound_list
        self.arff_file = arff_file
        self.folds = folds
        self.uncovered = self.parseUncoveredReaction(uncovered_reaction_file)
        self.totalUnpredicted = 0
        self.n_jobs = n_jobs
        self.per_jobs = []
        self.starts = []

    def parseUncoveredReaction(self, uncovered_reaction_file):

        f = open(uncovered_reaction_file, "r")
        uncovered = dict()

        while True:
            line = f.readline()

            if not line:
                break

            if ">>" not in line: continue

            educts = line.split(">>")[0]
            if educts not in uncovered:
                uncovered[educts] = 1
            else:
                uncovered[educts] += 1

        return uncovered

    def evaluateFolds(self, folds):

        probs = []
        true = []
        unpredicted_counts = 0

        for fold in folds:
            class_probs_filtered, y_test_filtered, unpredicted = self.singleFoldNew(fold)
            probs += class_probs_filtered
            true += y_test_filtered
            unpredicted_counts += unpredicted

        return probs, true, unpredicted_counts

    def evaluate(self):

        probs_total = []
        true_total = []
        total_unpredicted_counts = 0

        self.prepareParallel()
        results = Parallel(n_jobs=self.n_jobs, verbose=50)(
                delayed(self.evaluateFolds)(range(self.starts[i], self.starts[i + 1]))
                for i in range(self.n_jobs))

        for result in results:
            probs_total += result[0]
            true_total += result[1]
            total_unpredicted_counts += result[2]

        self.totalUnpredicted = total_unpredicted_counts

        # for fold in range(self.folds):
        #     # print("Processing fold:", fold+1)
        #     class_probs_filtered, y_test_filtered = self.singleFoldNew(fold)
        #     probs_total += class_probs_filtered
        #     true_total += y_test_filtered

        eval_results = self.getEvalStats(probs_total, true_total)
        print(eval_results)

        return eval_results

    # from _bagging.py of sklearn
    def _partition_estimators(self, n_folds, n_jobs):
        """Private function used to partition folds between jobs."""
        # Compute the number of jobs
        n_jobs = min(effective_n_jobs(n_jobs), n_folds)

        # Partition estimators between jobs
        n_estimators_per_job = np.full(n_jobs, n_folds // n_jobs, dtype=int)
        n_estimators_per_job[: n_folds % n_jobs] += 1
        starts = np.cumsum(n_estimators_per_job)

        return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

    def prepareParallel(self):
        self.n_jobs, self.per_jobs, self.starts = self._partition_estimators(self.folds, self.n_jobs)

    def getEvalStats(self, probs_total, true_total):

        results = {"prdata":[]}
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for thred in thresholds:

            tmp_dict = dict()
            binarized_total =[int(p>=thred) for p in probs_total]

            actual_T = np.sum(np.array(true_total)) + self.totalUnpredicted
            predicted_T = np.sum(np.array(binarized_total))

            TP = np.sum(np.array(binarized_total)[np.array(true_total)== 1])

            if actual_T == 0:
                recall = 1.0
            else:
                recall = TP/actual_T

            if predicted_T == 0:
                precision = 1.0
            else:
                precision = TP/predicted_T

            tmp_dict["precision"] = precision
            tmp_dict["recall"] = recall
            tmp_dict["threshold"] = thred

            results["prdata"].append(tmp_dict)

        return results

    def singleFoldNew(self, fold):

        arff_df = self.arffToDataframe()
        arff_df["smiles"] = self.compound_list
        arff_df = arff_df.set_index("smiles")

        with open(self.arff_file, "r") as f:
            arff_data = f.read()

        arff_data_head, arff_data_tail = arff_data.split("@data\n")
        arff_data_tail_list = arff_data_tail.strip("\n").split("\n")

        train_arff_rows, test_arff_rows, compound_train, compound_test = train_test_split(arff_data_tail_list, self.compound_list, test_size=0.1, random_state=fold+1)
        y_ground_true = arff_df.loc[compound_test].reset_index().drop(columns=["smiles"]).iloc[:, :self.N_class]

        train_arff_contents = arff_data_head + "@data\n" + "\n".join(train_arff_rows)
        test_arff_contents = arff_data_head + "@data\n" + "\n".join(test_arff_rows)

        train_arff_file = tempfile.NamedTemporaryFile(delete=False)
        test_arff_file = tempfile.NamedTemporaryFile(delete=False)

        with open(train_arff_file.name+".arff", "w") as f:
            f.write(train_arff_contents)

        with open(test_arff_file.name+".arff", "w") as f:
            f.write(test_arff_contents)

        self.model.fit(train_arff_file.name+".arff")
        class_probs = self.model.predict_pro(test_arff_file.name+".arff")

        unpredicted = 0
        for compound in compound_test:
            if compound in self.uncovered:
                unpredicted += self.uncovered[compound]
        self.totalUnpredicted += unpredicted

        class_probs_flatten = class_probs.toarray().reshape(1,-1).squeeze()
        y_test_flatten = np.array(y_ground_true).reshape(1, -1).squeeze()
        class_probs_filtered = list(class_probs_flatten[y_test_flatten != "?"])
        y_test_filtered = [int(v) for v in y_test_flatten[y_test_flatten != "?"]]

        self._remove_temporary_files([train_arff_file, test_arff_file])

        return class_probs_filtered, y_test_filtered, unpredicted

    def _remove_temporary_files(self, temporary_files):
        """Internal function for cleaning temporary files"""
        for file_object in temporary_files:
            file_name = file_object.name
            file_object.close()
            if os.path.exists(file_name):
                os.remove(file_name)

            arff_file_name = file_name + '.arff'
            if os.path.exists(arff_file_name):
                os.remove(arff_file_name)

    def singleFold(self, fold):

        label_location = "start"
        arff_file_is_sparse = False

        X, y, feature_names, label_names = load_from_arff(
            self.arff_file,
            label_count=self.N_class,
            label_location=label_location,
            load_sparse=arff_file_is_sparse,
            return_attribute_definitions=True
        )

        arff_df = self.arffToDataframe()
        arff_df["smiles"] = self.compound_list
        arff_df = arff_df.set_index("smiles")

        X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y, self.compound_list, test_size=0.1, random_state=fold+1)
        y_ground_true = arff_df.loc[compound_test].reset_index().drop(columns=["smiles"]).iloc[:,:self.N_class]
        self.model.train(X_train, y_train)
        class_probs = self.model.predict_prob(X_test)

        unpredicted = 0
        for compound in compound_test:
            if compound in self.uncovered:
                unpredicted += self.uncovered[compound]
        self.totalUnpredicted += unpredicted

        class_probs_flatten = np.array(class_probs).reshape(1,-1).squeeze()
        y_test_flatten = np.array(y_ground_true).reshape(1,-1).squeeze()
        class_probs_filtered = list(class_probs_flatten[y_test_flatten != "?"])
        y_test_filtered = [int(v) for v in y_test_flatten[y_test_flatten != "?"]]

        return class_probs_filtered, y_test_filtered

    def arffToDataframe(self):
        arff_file = open(self.arff_file, 'r')
        corr_attributes = []
        trig_attributes = []
        struct_attributes = []

        for line in arff_file:
            if (line[:10] == "@attribute"):
                attr = line.split(' ')[1]
                if (attr[:5] == "CORR-"):
                    corr_attributes.append(attr)
                elif (attr[:5] == "TRIG-"):
                    trig_attributes.append(attr)
                elif (attr[:7] == "STRUCT-"):
                    struct_attributes.append(attr)
            elif (line[:5] == "@data"):
                break

        arff_file.close()

        colnames = corr_attributes + trig_attributes + struct_attributes

        arff_file = open(self.arff_file, 'r')
        rows_list = []
        for line in arff_file:
            if (line[0] == "?" or line[0] == "1" or line[0] == "0"):
                entry_dict = dict()
                data = line.split('\n')[0].split(',')
                assert len(data) == len(colnames)

                for i in range(len(data)):
                    entry_dict[colnames[i]] = data[i]

                rows_list.append(entry_dict)

        arff_df = pd.DataFrame(rows_list)

        return arff_df


if __name__ == '__main__':

    # meka_classpath = download_meka()
    # ecc = EccClassifier(meka_classpath=meka_classpath)
    from datetime import datetime

    start_time = datetime.now()

    meka_classpath = download_meka()
    meka_bagging_classifier = "meka.classifiers.multilabel.meta.BaggingML"
    meka_base_classifier = "meka.classifiers.multilabel.CC"
    weka_classifier = "weka.classifiers.trees.J48"
    java_command = '/usr/bin/java'

    meka = EccMeka(meka_classpath, meka_bagging_classifier, meka_base_classifier, weka_classifier, java_command,
                   n_estimators=10)

    compounds_file = "../compounds.txt"
    compounds = []
    with open(compounds_file, 'r') as f:
        while (True):
            line = f.readline().replace("\n", "")
            if not line:
                break

            compounds.append(line)
        f.close()

    uncovered_reaction_file = "experiment3/uncovered_reactions.txt"
    path_to_arff_file = "experiment3/train.arff"
    evaluator = singleGen(meka, 242, compounds, path_to_arff_file, 100, uncovered_reaction_file)
    results = evaluator.evaluate()

    results_file = "experiment3/singleGen_100folds_results.json"

    with open(results_file, 'w') as fp:
        json.dump(results, fp)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
