from skmultilearn.ext import Meka
from scipy import sparse
import numpy as np
import tempfile
import shlex
import subprocess
import sys
import re
from skmultilearn.ext import download_meka

class EccMeka(Meka):
    """Ensemble Chain Classifiers
    """
    def __init__(self, meka_classpath, meka_bagging_classifier, meka_base_classifier, weka_classifier, java_command, n_estimators):
        super().__init__(meka_bagging_classifier, weka_classifier, java_command, meka_classpath)
        self.meka_base_classifier = meka_base_classifier
        self.n_estimators = n_estimators

    def save(self, model_path):

        if self.classifier_dump == None or self.train_data_ == None:
            print("Model is not trained.")
            sys.exit(1)

        with open(model_path, "wb") as f:
            f.write(self.classifier_dump)

        with open(model_path+".arff", "w") as f:
            f.write(self.train_data_)

    def load(self, model_path):
        self._clean()

        with open(model_path, "rb") as fp:
            self.classifier_dump = fp.read()

        with open(model_path+".arff", "r") as f:
            self.train_data_ = f.read()

    def fit(self, train_arff):
        self._clean()

        classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)

        try:

            input_args = [
                '-I', "{}".format(self.n_estimators),
                '-verbosity', "0",
                '-split-percentage', "100",
                '-t', '"{}"'.format(train_arff),
                '-d', '"{}"'.format(classifier_dump_file.name),
            ]

            extra_args = [
                self.java_command,
                '-cp', '"{}*"'.format(self.meka_classpath),
                self.meka_classifier,
            ]

            input_args = extra_args + input_args

            weka_args = [
                '-W', self.meka_base_classifier,
                '--',
                '-W', self.weka_classifier,
            ]

            input_args += weka_args

            self._run_meka_command(input_args)
            self.train_data_ = None
            self.classifier_dump = None

            with open(train_arff, 'r') as f:
                self.train_data_ = f.read()

            search_pattern = "(?<=: -C )[0-9]+"
            N_class = re.findall(search_pattern, self.train_data_)
            try:
                self._label_count = int(N_class[0])
            except ValueError:
                print("Number of class cannot be parsed from training data")
                exit(1)

            with open(classifier_dump_file.name, 'rb') as fp:
                self.classifier_dump = fp.read()

        finally:
            self._remove_temporary_files([classifier_dump_file])

        return self

    def predict_pro(self, test_arff):

        classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)
        train_arff = tempfile.NamedTemporaryFile(delete=False)

        try:
            with open(classifier_dump_file.name, 'wb') as fp:
                fp.write(self.classifier_dump)

            with open(train_arff.name + '.arff', 'w') as fp:
                fp.write(self.train_data_)

            input_args = [
                '-I', '{}'.format(self.n_estimators),
                '-verbosity', "7",
                '-t', '"{}"'.format(train_arff.name + '.arff'),
                '-T', '"{}"'.format(test_arff),
                '-l', '"{}"'.format(classifier_dump_file.name),
            ]

            extra_args = [
                self.java_command,
                '-cp', '"{}*"'.format(self.meka_classpath),
                self.meka_classifier,
            ]

            input_args = extra_args + input_args

            weka_args = [
                '-W', self.meka_base_classifier,
                '--',
                '-W', self.weka_classifier,
            ]

            input_args += weka_args
            self._run_meka_command(input_args)
            self.parse_output()
        finally:
            self._remove_temporary_files(
                [train_arff, classifier_dump_file]
            )

        return self._results

    def predict(self, test_arff, threshold):

        pro_predictions = self.predict_pro(test_arff).toarray()
        binarized_predictions = self.binarize_results(pro_predictions, threshold)

        return sparse.lil_matrix(binarized_predictions)

    def binarize_results(self, class_probs, threshold):

        class_binarizes = []

        for class_prob in class_probs:
            class_binarize = np.where(class_prob >= threshold, 1, 0)
            class_binarizes.append(class_binarize)

        return np.array(class_binarizes, dtype=int)

    def parse_output(self):

        if self.output_ is None:
            self._results = None
            self._statistics = None
            return None

        predictions_split_head = '==== PREDICTIONS'
        predictions_split_foot = '|==========='

        if self._label_count is None:
            search_pattern = "(?<=Number of labels \(L\)           )[0-9]+"
            self._label_count = int(re.findall(search_pattern, self.output_)[0])

        if self._instance_count is None:
            search_pattern = "(?<=\=\=\=\= PREDICTIONS \(N=)[0-9]+"
            self._instance_count = int(re.findall(search_pattern, self.output_)[0])

        predictions = self.output_.split(predictions_split_head)[1].split(
            predictions_split_foot)[0].split('\n')[1:-1]
        predictions = [x.split("] [")[1].split("]")[0].strip().split(" ") for x in predictions]
        predictions = [[float(x) for x in y] for y in predictions]

        self._results = sparse.lil_matrix(np.array(predictions))

        statistics = [x for x in self.output_.split(
            '== Evaluation Info')[1].split('\n') if len(x) > 0 and '==' not in x]
        statistics = [y for y in [z.strip() for z in statistics] if '  ' in y]
        array_data = [z for z in statistics if '[' in z]
        non_array_data = [z for z in statistics if '[' not in z]

        self._statistics = {}
        for row in non_array_data:
            r = row.strip().split('  ')
            r = [z for z in r if len(z) > 0]
            r = [z.strip() for z in r]
            if len(r) < 2:
                continue
            try:
                test_value = float(r[1])
            except ValueError:
                test_value = r[1]

            r[1] = test_value
            self._statistics[r[0]] = r[1]

        for row in array_data:
            r = row.strip().split('[')
            r = [z.strip() for z in r]
            r[1] = r[1].replace(', ', ' ').replace(
                ',', '.').replace(']', '').split(' ')
            r[1] = [x for x in r[1] if len(x) > 0]
            self._statistics[r[0]] = r[1]

        return self._results


    def _run_meka_command(self, args):

        meka_command = " ".join(args)

        if sys.platform != 'win32':
            meka_command = shlex.split(meka_command)

        pipes = subprocess.Popen(meka_command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        self.output_, self._error = pipes.communicate()
        if type(self.output_) == bytes:
            self.output_ = self.output_.decode(sys.stdout.encoding)
        if type(self._error) == bytes:
            self._error = self._error.decode(sys.stdout.encoding)

        if pipes.returncode != 0:
            raise Exception(self.output_ + self._error)


if __name__ == '__main__':

    # meka_classpath = "/Users/kunyang/Documents/Eawag/envipath/meka-release-1.9.5/lib/"
    meka_classpath = download_meka()
    print(meka_classpath)
    meka_bagging_classifier = "meka.classifiers.multilabel.meta.BaggingML"
    meka_base_classifier = "meka.classifiers.multilabel.CC"
    weka_classifier = "weka.classifiers.trees.J48"
    java_command = '/usr/bin/java'

    meka = EccMeka(meka_classpath, meka_bagging_classifier, meka_base_classifier, weka_classifier, java_command, n_estimators=10)

    train_arff_file = ".../train_X.arff"
    test_arff_file = ".../test_X.arff"

    meka.fit(train_arff_file)
    # predictions1 = meka.predict_pro(test_arff_file)
    # result1 = meka.predict(test_arff_file, 0.51)

    result12 = meka.predict(test_arff_file, 0.4)
    # model_path = "experiment1/ecc10models"
    # meka.save(model_path)
    #
    # meka2 = EccMeka(meka_classpath, meka_bagging_classifier, meka_base_classifier, weka_classifier, java_command, n_estimators=10)
    # meka2.load(model_path)
    #
    # predictions2 = meka2.predict_pro(test_arff_file)
    print(result12)
