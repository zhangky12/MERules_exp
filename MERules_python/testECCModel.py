from skmultilearn.ext import download_meka
from ECCModel import *
from skmultilearn.dataset import load_dataset
from sklearn.metrics import hamming_loss
from scipy import sparse
from ModelInput import *
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split

meka_classpath = download_meka()
ecc = EccClassifier(meka_classpath=meka_classpath)
#
# X_train, y_train, _, _ = load_dataset('scene', 'train')
# X_test,  y_test, _, _ = load_dataset('scene', 'test')
#
# ecc.train(X_train, y_train)
# class_probs, class_binarizes = ecc.predict_prob(X_test, 0.4)
# print(class_probs)
# print(class_binarizes)

# path = "savedModels"
# ecc.save_ecc(path)
# ecc.load_ecc(path, 10)

# class_probs2, class_binarizes2 = ecc.predict_prob(X_test, 0.4)
# print(class_probs2)
# print(class_binarizes2)
#
# print(np.mean(class_probs == class_probs2) == 1)
# print(np.mean(class_binarizes == class_binarizes2) == 1)

# predictions = sparse.csr_matrix(class_binarizes2)
# print(hamming_loss(y_test, predictions))

ruleFile = "/Users/kunyang/Documents/Eawag/envipath/envipath-python/enviPath_python/MERules_exp/Decomposition_rules.txt"
input_handler = ModelInput(ruleFile, "experiment1")

compounds_file = "/Users/kunyang/Documents/Eawag/envipath/envipath-python/enviPath_python/MERules_exp/compounds.txt"
compounds = []
with open(compounds_file, 'r') as f:
    while (True):
        line = f.readline().replace("\n", "")
        if not line:
            break

        compounds.append(line)
    f.close()

# input_df = input_handler.multiCompounds(compounds)
# print(input_df.shape)

reactions_file = "/Users/kunyang/Documents/Eawag/envipath/envipath-python/enviPath_python/MERules_exp/bbd_reactions.txt"
reactions = []
reactions = set()
with open(reactions_file, 'r') as f:
    while (True):
        line = f.readline().replace("\t", "").replace("\n", "")
        if not line:
            break

        reactions.add(line)
    f.close()

# train_matrix = input_handler.get_trainMatrix(compounds, reactions)
# print(train_matrix.shape)

path_to_arff_file = "experiment1/train.arff"
label_count = 242
label_location = "start"
arff_file_is_sparse = False

X, y, feature_names, label_names = load_from_arff(
    path_to_arff_file,
    label_count=label_count,
    label_location=label_location,
    load_sparse=arff_file_is_sparse,
    return_attribute_definitions=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# ecc.train(X_train, y_train)
path = "savedModels"
# ecc.save_ecc(path)
ecc.load_ecc(path, 10)

class_probs, class_binarizes = ecc.predict(X_test, 0.4)
predictions = sparse.csr_matrix(class_binarizes)
print(hamming_loss(y_test, predictions))

