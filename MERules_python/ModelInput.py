from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from scipy import sparse
import pandas as pd
import os
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

class ModelInput:

    def __init__(self, ruleFile, experimentName):

        self.ruleFile = ruleFile
        self.rules, self.ruleNames = self.parseRuleFile()
        self.MACCSbits = 167
        self.RULEbits = len(self.rules)
        self.recordProducts = False
        self.ruleProducts = dict()
        self.experimentName = experimentName
        self.trainMatrix = None
        self.uncoveredReactions = None

        self.normalize_rules = []

        self.basic_rules = dict()
        self.basic_rules = {"ammoniumstandardization":"[H][#7+:1]([H])([H])[#6:2]>>[H][#7+0:1]([H])-[#6:2]",
                    "cyanate": "[H][#8:1][C:2]#[N:3]>>[#8-:1][C:2]#[N:3]",
                    "deprotonatecarboxyls": "[H][#8:1]-[#6:2]=[O:3]>>[#8-:1]-[#6:2]=[O:3]",
                    "forNOOH": "[H][#8:1]-[#7+:2](-[*:3])=[O:4]>>[#8-:1]-[#7+:2](-[*:3])=[O:4]",
                    "Hydroxylprotonation": "[#6;A:1][#6:2](-[#8-:3])=[#6;A:4]>>[#6:1]-[#6:2](-[#8-0:3][H])=[#6;A:4]",
                    "phosphatedeprotonation": "[H][#8:1]-[$([#15]);!$(P([O-])):2]>>[#8-:1]-[#15:2]",
                    "PicricAcid": "[H][#8:1]-[c:2]1[c:3][c:4][c:5]([c:6][c:7]1-[#7+:8](-[#8-:9])=[O:10])-[#7+:11](-[#8-:12])=[O:13]>>[#8-:1]-[c:2]1[c:3][c:4][c:5]([c:6][c:7]1-[#7+:8](-[#8-:9])=[O:10])-[#7+:11](-[#8-:12])=[O:13]",
                    "Sulfate1": "[H][#8:1][S:2]([#8:3][H])(=[O:4])=[O:5]>>[#8-:1][S:2]([#8-:3])(=[O:4])=[O:5]",
                    "Sulfate2": "[#6:1]-[#8:2][S:3]([#8:4][H])(=[O:5])=[O:6]>>[#6:1]-[#8:2][S:3]([#8-:4])(=[O:5])=[O:6]",
                    "Sulfate3": "[H][#8:3][S:2]([#6:1])(=[O:4])=[O:5]>>[#6:1][S:2]([#8-:3])(=[O:4])=[O:5]",
                    "Transform_c1353forSOOH": "[H][#8:1][S:2]([*:3])=[O:4]>>[#8-:1][S:2]([*:3])=[O:4]"}

        self.enhanced_rules = dict()
        self.enhanced_rules.update(self.basic_rules)
        self.enhanced_rules["fullPhosphatedeprotonation"] = "[H][#8:1]-[#15:2]>>[#8-:1]-[#15:2]"

        self.exotic_rules = dict()
        self.exotic_rules.update(self.enhanced_rules)
        self.exotic_rules["ThioPhosphate1"] = "[H][S:1]-[#15:2]=[$([#16]),$([#8]):3]>>[S-:1]-[#15:2]=[$([#16]),$([#8]):3]"

        self.cutCOA_rule = dict()
        self.cutCOA_rule["CutCoEnzymeAOff"] = "CC(C)(COP(O)(=O)OP(O)(=O)OCC1OC(C(O)C1OP(O)(O)=O)n1cnc2c(N)ncnc12)C(O)C(=O)NCCC(=O)NCCS[$(*):1]>>[O-][$(*):1]"

        self.enol2Ketone_rule = dict()
        self.enol2Ketone_rule["enol2Ketone"] = "[H][#8:2]-[#6:3]=[#6:1]>>[H][#6:1]-[#6:3]=[O:2]"

        self.normalize_rules.append(self.exotic_rules)
        self.normalize_rules.append(self.cutCOA_rule)
        self.normalize_rules.append(self.enol2Ketone_rule)


    def normalize_smiles(self, smiles):

        results = set()
        results.add(smiles)

        for rule_set in self.normalize_rules:

            pre_products = set()
            pre_products.add(smiles)
            try:
                while len(pre_products) != 0:

                    post_products = set()

                    for compound in pre_products:

                        mol = Chem.MolFromSmiles(compound)
                        mol = Chem.AddHs(mol)

                        for rule_name in rule_set:
                            rxn = AllChem.ReactionFromSmarts(rule_set[rule_name])
                            products = [results[0] for results in rxn.RunReactants((mol,))]
                            for product in products:
                                product_smiles = Chem.MolToSmiles(Chem.RemoveHs(product))
                                post_products.add(product_smiles)
                                results.add(product_smiles)

                    pre_products = post_products
            except:
                continue

        return results

    def parseRuleFile(self):

        rules = set()
        rules_list = []
        rule_names = []

        with open(self.ruleFile, 'r') as f:
            while (True):
                line = f.readline().replace("\t", "").replace("\n", "")
                if not line:
                    break
                rules.add(line)
            f.close()

        count = 1
        for rule in rules:
            rule_names.append("rule-" + str(count))
            rules_list.append(rule)
            count += 1

        return rules_list, rule_names

    def applyRules(self, smiles):

        rule_features = []
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)

        for rule in self.rules:
            rxn = AllChem.ReactionFromSmarts(rule)
            products_list = rxn.RunReactants((mol,))

            if len(products_list) > 0:
                rule_features.append(1)

                if self.recordProducts:

                    if smiles not in self.ruleProducts:
                        self.ruleProducts[smiles] = dict()

                    if rule not in self.ruleProducts[smiles]:
                        self.ruleProducts[smiles][rule] = set()

                    for result in products_list:
                        for product in result:
                            Chem.RemoveStereochemistry(product)
                            product_smiles = Chem.MolToSmiles(product)

                            if "." in product_smiles:
                                sub_products = product_smiles.split(".")
                                for sub_product in sub_products:
                                    self.ruleProducts[smiles][rule].add(sub_product)
                                    # self.ruleProducts[smiles][rule].union(self.normalize_smiles(sub_product))
                            else:
                                self.ruleProducts[smiles][rule].add(product_smiles)
                                # self.ruleProducts[smiles][rule].union(self.normalize_smiles(product_smiles))
            else:
                rule_features.append(0)

        return rule_features

    def singleCompound(self, smiles):

        rule_features = self.applyRules(smiles)
        maccs_fps = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)).ToList()
        all_features = rule_features + maccs_fps

        return sparse.csr_matrix(all_features.ToList())

    def multiCompounds(self, smiles_list):

        df = pd.DataFrame({"smiles": smiles_list})
        rule_features = df.apply(lambda x: self.applyRules(x["smiles"]), axis=1, result_type='expand')
        rule_features.columns = self.ruleNames
        rule_features["smiles"] = smiles_list
        maccs_fps = df.apply(lambda x: MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x["smiles"])).ToList(), axis=1,
                             result_type='expand')
        struct_names = ["Struct-" + str(i) for i in range(self.MACCSbits)]
        maccs_fps.columns = struct_names
        maccs_fps["smiles"] = smiles_list

        input_df = pd.merge(rule_features, maccs_fps, on="smiles", how="outer")
        input_df = input_df.drop(columns=["smiles"])

        total_length = 0
        for compound in self.ruleProducts:
            total_length += len(self.ruleProducts[compound])

        return sparse.csr_matrix(input_df)

    def get_rules(self):
        return self.rules

    def get_class(self, smiles, observed_records):

        classes = []

        if smiles not in self.ruleProducts:
            classes = ["?"] * len(self.rules)
            return classes

        if smiles not in observed_records:
            for rule in self.rules:
                if rule in self.ruleProducts[smiles]:
                    classes.append("0")
                else:
                    classes.append("?")
            return classes

        for rule in self.rules:
            if rule in self.ruleProducts[smiles] and rule in observed_records[smiles]:
                classes.append("1")
            elif rule in self.ruleProducts[smiles] and rule not in observed_records[smiles]:
                classes.append("0")
            else:
                classes.append("?")

        return classes

    def get_trainMatrix(self, smiles_list, reaction_list):

        self.recordProducts = True
        self.uncoveredReactions = None
        self.trainMatrix = None
        observed_records = dict()
        uncovered_reactions = []

        X_train = pd.DataFrame(self.multiCompounds(smiles_list).toarray())

        for reaction in reaction_list:
            educts, products = reaction.split(">>")
            found = False

            if "." in educts or educts not in self.ruleProducts:
                continue

            for product in products.split("."):
                product_mol = Chem.MolFromSmiles(product)
                Chem.RemoveStereochemistry(product_mol)
                product_smiles = Chem.MolToSmiles(product_mol)

                normalized_product_smiles = self.normalize_smiles(product_smiles)

                for rule in self.ruleProducts[educts].keys():
                    # if product_smiles in self.ruleProducts[educts][rule]:

                    rule_product_set = set()
                    for rule_product in self.ruleProducts[educts][rule]:
                        rule_product_set = rule_product_set.union(self.normalize_smiles(rule_product))

                    # if len(normalized_product_smiles.intersection(self.ruleProducts[educts][rule])) > 0:
                    if len(normalized_product_smiles.intersection(rule_product_set)) > 0:
                        if educts not in observed_records:
                            observed_records[educts] = set()
                        observed_records[educts].add(rule)
                        found = True
            if not found:
                uncovered_reactions.append(reaction)

        df = pd.DataFrame({"smiles": smiles_list})
        corr_classes = df.apply(lambda x: self.get_class(x["smiles"], observed_records), axis=1, result_type='expand')
        corr_classes.columns = self.ruleNames
        corr_classes["smiles"] = smiles_list

        X_train["smiles"] = smiles_list
        train_matrix = pd.merge(corr_classes, X_train, on="smiles", how="outer")
        train_matrix = train_matrix.drop(columns=["smiles"])

        self.ruleProducts = dict()
        self.recordProducts = False
        self.uncoveredReactions = uncovered_reactions
        self.trainMatrix = train_matrix

        # self.writeArff(train_matrix, uncovered_reactions)

        return train_matrix

    def writeUnobserved(self, file_dir, uncovered_reactions):

        with open(file_dir, "w") as f:
            for reaction in uncovered_reactions:
                f.write(reaction + "\n")

    def writeArff(self, arff_only=False):
        dir = (self.experimentName)

        if not os.path.isdir(dir):
            os.makedirs(dir)

        arff_file = self.experimentName + "/train.arff"

        if not arff_only:
            uncovered_reaction_file = self.experimentName + "/uncovered_reactions.txt"
            self.writeUnobserved(uncovered_reaction_file, self.uncoveredReactions)

        headLine = "@relation '" + self.experimentName + ": -C " + str(self.RULEbits) + "'\n"

        with open(arff_file, "w") as f:
            f.write(headLine)
            f.write("\n")

            # attributes line
            for rule_name in self.ruleNames:
                line = "@attribute CORR-" + rule_name + " {0,1}\n"
                f.write(line)
            for rule_name in self.ruleNames:
                line = "@attribute TRIG-" + rule_name + " {0,1}\n"
                f.write(line)
            for i in range(self.MACCSbits):
                line = "@attribute STRUCT-" + str(i) + " {0,1}\n"
                f.write(line)

            f.write("\n")
            f.write("@data\n")

            for row in range(len(self.trainMatrix)):
                line = ",".join([str(v) for v in self.trainMatrix.loc[row]]) + "\n"
                f.write(line)


if __name__ == '__main__':



    ruleFile = ".../Decomposition_rules.txt"
    input_handler = ModelInput(ruleFile, "experiment3")

    compounds_file = ".../compounds.txt"
    compounds = []
    with open(compounds_file, 'r') as f:
        while (True):
            line = f.readline().replace("\n", "")
            if not line:
                break

            compounds.append(line)
        f.close()

    reactions_file = ".../bbd_reactions.txt"
    reactions = set()
    with open(reactions_file, 'r') as f:
        while (True):
            line = f.readline().replace("\t", "").replace("\n", "")
            if not line:
                break

            reactions.add(line)
        f.close()


    train_matrix = input_handler.get_trainMatrix(compounds, reactions)
    input_handler.writeArff()

    # print(input_handler.normalize_smiles("CCCCC[N+]([H])([H])[H]"))



