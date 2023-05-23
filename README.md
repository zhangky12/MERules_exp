# MERules_exp

This python project is from enviRule manuscript and aims to evaluate the performance of models trained with rules reported from another study (DOI: https://doi.org/10.1016/j.ymben.2021.02.006). The original rules can be found in the folder "MERules_SM", and the subset of decomposition rules (one substrate -> multiple products) is stored in "Decomposition_rules.txt". 

Predictive models (ECC) of transformation products (TPs) are implemented with MEKA, which require training files in arff format (experiment3/train.arff). 

Evaluation results of models reported in Figure 6 of enviRule manuscript were achieved by running "SingleGenEval.py". 

To run "SingleGenEval.py", three files need to be provided:

1. Compound list of unique compounds in database (e.g., compounds.txt).
2. Reactions in database that cannot be covered by rules (e.g., experiment3/uncovered_reactions.txt).
3. Multi-label training files for MEKA (e.g., experiment3/train.arff).

```python
uncovered_reaction_file = "experiment3/uncovered_reactions.txt"
path_to_arff_file = "experiment3/train.arff"
results_file = "experiment3/singleGen_100folds_results.json"
```