import pandas as pd
import numpy as np
from decisiontree import dtree, entropy, gini
from algorithms import predict, metrics, collapseToBin
import warnings
warnings.filterwarnings("ignore")

# Intialize possible hyperparameters
criterions = [gini, entropy]
max_depths = [i for i in range(1, 23)]
target_impurities = np.arange(14, -1, -2) / 100
min_examples = [i for i in range(8, 2, -1)]

critMap = {gini: "gini", entropy: 'entropy'}


accuracies = []

df = pd.read_csv("../data/team_data_v4.csv")
df.drop(["Unnamed: 0"], inplace=True, axis=1)


# do mini
def validate(data, crit, max_dpth=None, min_inst=2, target_imp=0.0):
    folds = 10
    class_col = data.columns[-1]
    temp_accuracies = []

    for f in range(folds):
        train_fold = data[data.index % folds != f]
        model = dtree(train_fold, crit, class_col, max_dpth, min_inst, target_imp)
        # print("completed training -------")

        valid_fold = data[data.index % folds == f]
        predicts = predict(model, valid_fold)

        actual = np.array(valid_fold["track_genre"])
        scores = metrics(actual, predicts)
        # print(scores)
        accuracy = scores["accuracy"]
        temp_accuracies.append(accuracy)
    return round(sum(temp_accuracies) / 10, 4)


def testHyperparams(df):
    bestScore = 0
    bestDepth = max_depths[0]
    bestTarget = target_impurities[0]
    bestMin = min_examples[0]

    for max_depth in max_depths:
        for crit in criterions:
            print(critMap[crit], "| max depth;", max_depth, "| target impurity;", bestTarget, "| minimum instances;",
                  bestMin)
            score = validate(df, crit, max_depth, bestMin, bestTarget)
            if score >= bestScore:
                bestDepth = max_depth
                bestScore = score
            print(score)

    best_score = 0
    for target in target_impurities:
        for crit in criterions:
            print(critMap[crit], "| max depth;", bestDepth, "| target impurity;", target, "| minimum instances;",
                  bestMin)
            score = validate(df, crit, bestDepth, bestMin, target)
            if score >= bestScore:
                bestTarget = target
                bestScore = score
            print(score)

    best_score = 0
    for min_example in min_examples:
        for crit in criterions:
            print(critMap[crit], "| max depth;", bestDepth, "| target impurity;", bestTarget, "| minimum instances;",
                  min_example)
            score = validate(df, crit, bestDepth, min_example, bestTarget)
            if score >= bestScore:
                bestMin = min_example
                bestScore = score
            print(score)
    return bestScore, bestTarget, bestDepth, bestMin

selected = testHyperparams(df)
print(selected)
