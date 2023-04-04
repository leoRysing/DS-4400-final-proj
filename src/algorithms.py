def predict_winner(row):
    """
    Predict the winner of a game
    :param row: row of dataframe, ie a given game
    :return:
    """
    pass


# 2
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

    return accuracy, sensitivity, specificity, precision, f1_score