"""
Matthew Keefer, Leroy Schaigorodsky, Alan Sourek, Ceara Zhang
DS 4400 / Hockey Game Analysis
Final Project
Date Created: 3/29/23
Last Updated: 4/5/2023
"""

from decisiontree import left, right, majorityClass, criterion, criteria
from collections import Counter
import numpy as np
import pandas as pd
import random


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


def predictdot(X, w):
    """
    takes in a 2D array of attribute values and a weight vector to return
    a series of predictions per row of the table
    :param X: (dataframe) dataframe of attributes & bias term
    :param w: (list) weight vector
    :return: a series of predictions (1 or 0)
    """

    predictions = X.dot(w)
    return (predictions >= 0).astype(int)
def rand_w(min_val, max_val, num_ints):
    """
    returns a w vector of random integers
    :param min_val: (int) min value vector will have
    :param max_val: (int) max value vector will have
    :param num_ints: (int) number of integers vector will have
    :return: a list of random integers
    """
    w_vector = []
    for i in range(0,num_ints):
        n = random.randint(min_val, max_val)
        w_vector.append(n)
    return w_vector


import pandas as pd
import numpy as np

def perceptron(data, alpha=0.0001, epochs=1000):
    """
    linear perceptron function that finds the weight vector of a given dataset
    :param data: (dataframe) dataframe
    :param alpha: (float) the learning rate
    :param epochs: (int) number of epochs
    :return: a list of integers ie. weight vector
    """

    data = data.copy()
    # data = data[['home_team', 'home_score', 'away_team', 'away_score', 'game_state']]

    # List of categorical column names
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']

    # One-hot encoding for categorical columns
    for col in cat_cols:
        data[col] = data[col].astype('category').cat.codes

    # add a bias column to the data
    data.insert(15, 'bias', 1)

    # adding outcomes column team name
    # data['outcome'] = data.apply(lambda row:
    #                              1 if row['home_score'] > row['away_score'] else
    #                              -1 if row['home_score'] < row['away_score'] else
    #                              0, axis=1)

    # X is all attributes plus a bias column
    X = data.iloc[:, :-1].values

    # y is the actual values column
    y = data.iloc[:, -1].values

    # total number of attributes including bias column
    num_attrib = X.shape[1]

    # initial w
    w = np.random.rand(num_attrib)

    # lists to store MPE and model accuracy per epoch (for #6)
    mpe_list = []
    acc_list = []

    for i in range(epochs):

        # predict using x array and w array
        ypred = predictdot(X, w)

        # calculate accuracy for current epoch
        curr_accuracy = np.sum(ypred == y) / len(y)
        acc_list.append(curr_accuracy)

        # calculate MPE for current epoch
        curr_mpe = np.mean(np.abs(y - ypred))
        mpe_list.append(curr_mpe)

        for j in range(num_attrib):
            # update the weights if predictions mismatch
            # for each attribute in w
            w[j] = w[j] + alpha * np.sum((y - ypred) * X[:, j])

    print(data.head())
    return acc_list, mpe_list, w

blah = pd.read_csv('team_data_v4.csv')
blahh = pd.DataFrame(blah)
acc_list, mpe_list, w = perceptron(blahh)


from matplotlib import pyplot as plt
# x range
num_epochs = np.arange(0, 1000, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# # plot MPE
ax1.plot(num_epochs, mpe_list)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MPE')
ax1.set_title('Mean Perceptron Error')

# plot accuracy
ax2.plot(num_epochs, acc_list)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy')

# display the plot
plt.show()



