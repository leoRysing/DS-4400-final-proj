"""
Matthew Keefer, Leroy Schaigorodsky, Alan Sourek, Ceara Zhang
DS 4400 / Hockey Game Analysis
Final Project
Date Created: 3/29/23
Last Updated: 4/5/2023
"""

from decisiontree import left, right, majorityClass, criterion, criteria
from collections import Counter


def predict_winner(model, row):
    """
    Predict the winner of a game
    :param row: row of dataframe, ie a given game
    :return:
    """
    if left(model) is not None and right(model) is not None:
        col, val = criterion(model), criteria(model)
        if row[col] >= val:
            return predict_winner(right(model), row)
        else:
            return predict_winner(left(model), row)
    else:
        return majorityClass(model)


# make a prediction based on some data
def predict(model, data):
    if model is not None:
        outputSeries = data.apply(lambda x: predict_winner(model, x), axis=1)
        return outputSeries
    else:
        return None
    # do something



def predict_bag(bag, data):
    if bag is not None:
        outputSeries = data.apply(lambda x: predict_bag_row(bag, x), axis=1)  # predict_bag_row(bag, x)
        outputSeries = outputSeries.apply(lambda x: most_common(x), axis=1)
        return outputSeries
    else:
        return None


def predict_bag_row(bag, row):
    output = bag.apply(lambda x: predict_winner(x, row))
    return output


def most_common(predictions):
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]


def collapseToBin(negative, x):
    if x == negative:
        return 0
    else:
        return 1


def metrics(y, ypred):
    """
    takes in a series of actual y labels predicted y labels, and returns
    the model accuracy, sensitivity, specificity, precision, f1-score
    :param y: (list) actual labels
    :param ypred: (list) predicted labels
    :return: 5 numbers; float
    """

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(len(y)):
        if y[i] == 1:
            # if actual y=1 and predicted y=1,
            # it is a true positive
            if ypred[i] == 1:
                true_pos += 1

            # if actual y=1 but predicted y=0,
            # it is a false negative
            else:
                false_neg += 1

        else:

            if y[i] == 0:
                # if actual y=0 and predicted y=0,
                # it is a true negative
                if ypred[i] == 0:
                    true_neg += 1

                # if actual y=0 but predicted y=1
                # it is a false positive
                else:
                    false_pos += 1

    # accuracy: proportion of correct predictions
    accuracy = (true_neg + true_pos) / len(y)

    # sensitivity: proportion of positive cases that were actually identified
    if true_pos + false_neg == 0:
        sensitivity = 0
    else:
        sensitivity = true_pos / (true_pos + false_neg)

    # specificity: proportion of negative cases that were actually identified
    if true_pos + false_neg == 0:
        specificity = 0
    else:
        specificity = true_neg / (true_neg + false_pos)

    # precision: proportion of positive cases that were actually positive
    if true_pos + false_neg == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)

    # f1 score: measures the balance between precision and sensitivity
    # 0-1, with 1 being a perfect balance between precision and sensitivity
    if true_pos + false_neg == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    scores = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1-score": f1_score
    }

    return scores
