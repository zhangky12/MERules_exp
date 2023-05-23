import getopt
import sys

from rdkit.Chem import AllChem
import rdkit.Chem as Chem


def reaction(rule, substrates):

    rxn = AllChem.ReactionFromSmarts(rule)
    mol = Chem.MolFromSmiles(substrates)
    products_list = rxn.RunReactants((mol,))

    return products_list


def main(argv):
    rule = ""
    substrates = ""
    opts, args = getopt.getopt(argv, "r:s:", ["rule=", "substrates="])

    for opt, arg in opts:
        if opt in ("-r", "--rule"):
            rule = arg
        elif opt in ("-s", "--substrates"):
            substrates = arg
        else:
            sys.exit()

    products_list = reaction(rule, substrates)
    for result in products_list:
        for product in result:
            print(Chem.MolToSmiles(product))
        print("-----")


if __name__ == '__main__':

    main(sys.argv[1:])
