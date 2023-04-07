import pandas as pd
from collections import Counter
from math import log2
import numpy as np
import math
import random


# Helper functions to get the scores
# all of them require the use of the counter
def gini(cnt):
    tot = total(cnt)
    return 1 - sum([(v / tot) ** 2 for v in cnt.values()])


def entropy(cnt):
    tot = total(cnt)
    return sum([(-v / tot) * log2(v / tot) for v in cnt.values()])


def wavg(cnt1, cnt2, measure):
    tot1 = total(cnt1)
    tot2 = total(cnt2)
    tot = tot1 + tot2
    return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot


def total(cnt):
    return sum(cnt.values())


# Some helper functions
def criterion(T):
    return T[0] if T is not None else None


def criteria(T):
    return T[1] if T is not None else None


def majorityClass(T):
    return T[3] if T is not None else None


def left(T):
    return T[6] if T is not None else None


def right(T):
    return T[7] if T is not None else None


# What is the depth of the tree?
# The depth is the Maximum number of "steps" to any leaf
def depth(T):
    """ Return the maximum depth of the tree """

    return T[5]

# get the values of a Tree
def vals(T):
    """The values that reach that node """
    return T[2] if T is not None else None

# evaluate which split is best
# prune trees early - question
def evaluate_split(df, class_col, split_col, feature_val, measure):
    # returns; the score, left, and rigth
    #TODO - change to work with treshold
    df1 = df[df[split_col] < feature_val]
    df2 = df[df[split_col] >= feature_val]
    cnt1, cnt2 = Counter(df1[class_col]), Counter(df2[class_col])
    return wavg(cnt1, cnt2, measure), df1, df2


# get a smaller set of values for a given column
def reduce_column_complexity(column):
    column_2 = column.apply(safe_round)
    setColumn = set(column_2)
    # print(Counter(column_2))
    return setColumn


def safe_round(val):
    if type(val) == type(True):
        return val
    elif val > 10706:
        # reduce the durations to half minutes
        return int(val / 60000) * 60000 + int(((val % 60000) / 30000)) * 30000  # round(val, -3) #val / 60000
    elif type(val) == type(1.1) and val >= 50:
        return round(val, -1)
    elif type(val) == type(1.1) and val >= 0 and val <= 0.00001:
        return int((val * (10 ** 6))) / (10 ** 6)
    elif type(val) == type(1.1) and val >= 0 and val <= 0.0001:
        return int(val * 10 ** 6 / 5) * 5 / (10 ** 6)
    elif type(val) == type(1.1) and val >= 0:
        return int(round(val, 2) * 100 / 5) * 5 / 100
    elif type(val) == type(1.1) and val < 0:
        return round(val, 0) + round(math.ceil(-(val % 1) / 0.5) * 0.5, 1) + 0.5
    else:
        return val


# find the best split in a given column
def best_split_for_column(df, class_col, split_col, method):
    best_v = ''
    best_meas = float("inf")
    best_left, best_right = None, None

    # for the split; left (1) will be the values < v, and right (2) will be the values >= v
    # experiment; split on the unique values rounded to 1 or 2 digits
    # how many unique sets are there, how precise on rounding
    # round by div 100
    # reduce to some number of significant digits
    # if numbers are greater than zero - maybe convert to minutes from milliseconds
    # be clear with what is being done
    reduced_split = reduce_column_complexity(df[split_col])
    for v in reduced_split:

        meas, left, right = evaluate_split(df, class_col, split_col, v, method)
        if meas < best_meas and left.shape[0] != 0 and right.shape[0] != 0:
            best_v = v
            best_meas = meas
            best_left = left
            best_right = right
    return best_v, best_meas, best_left, best_right


# find one optimal measured split
# returns; best_column, best_value, best_measure (score), left split, and right split
def best_split(df, class_col, method):
    best_col = 0
    best_v = ''
    best_meas = float("inf")
    best_left = None
    best_right = None

    for split_col in df.columns:
        if split_col != class_col:
            v, meas, left, right = best_split_for_column(df, class_col, split_col, method)
            if meas < best_meas:
                best_v = v
                best_meas = meas
                best_col = split_col
                best_left = left
                best_right = right
    return best_col, best_v, best_meas, best_left, best_right


# the dtree method, meant for setting up the tree, as a tupple
# returns a tupple with the following;
# 0 - feature / column name (splitting criterion) [check]
# 1 - feature value threshold (splitting criteria) [check]
# 2 - examples_in_split (values)
# 3 - majority class
# 4 - impurity_score
# 5 - depth
# 6 - left_subtree (leading to examples where feature value <= test threshold)
# 7 - right_subtree (leading to examples where feature value > test threshold)

def dtree(train, criterion, class_column, max_depth=None, min_instances=2, target_impurity=0.0):
    """ Build a binary tree, but stop splitting if number of values
    falls falls below some threshhold (min_vals) """
    # plan; make optimal split for this one level

    if train is None or train.shape[0] == 0:
        return None
    else:
        return dtree_help(train, criterion, class_column, max_depth, 0, min_instances, target_impurity)


# helper method for the dtree; created to allow for my chosen implementation
# of the depth, by using an accumulator
def dtree_help(train, criterion, class_column, max_depth, curr_depth, min_inst, targt_imp):
    # class_column = "track_genre" # factor out as a param
    counts = Counter(train[class_column])
    maj_class = counts.most_common(1)[0][0]
    vals = train[class_column]

    num_classes = len(counts)

    if train.shape[
        0] < min_inst or num_classes < 2:  # stop for minimum instances -> at least n items of the two classes
        return (None, None, vals, maj_class, 0, curr_depth, None, None)
    elif curr_depth == max_depth:  # stop for maximum depth reached
        return (None, None, vals, maj_class, 0, max_depth, None, None)

    else:
        # Find this current optimal split, at this point
        split_criterion, split_criteria, score, left, right = best_split(train, class_column, criterion)
        # print(split_criterion, split_criteria)
        # print(maj_class)
        # print("---------")
        # print(left.shape[0], right.shape[0])
        # if (right.shape[0] == 4):
        # print(right)
        # print()

        if score <= targt_imp:
            return (split_criterion, split_criteria, vals, maj_class, score, -1, None, None)

        # left tree, right tree
        left_tree = dtree_help(left, criterion, class_column, max_depth, curr_depth + 1, min_inst, targt_imp)
        right_tree = dtree_help(right, criterion, class_column, max_depth, curr_depth + 1, min_inst, targt_imp)

        return (split_criterion, split_criteria, vals, maj_class, score, curr_depth, left_tree, right_tree)


# an attempt at bagging
def bag(train, criterion, class_column, max_depth=None, min_instances=2, target_impurity=0.0):
    models = []
    for _ in range(0, 100):
        random_split = train.sample(500, axis=0)
        random_columns = list(random_split[random_split.columns[:-1]].sample(6))
        random_columns.append(class_column)

        random_split_train = random_split[random_columns]
        rand_tree = dtree(random_split_train, criterion, class_column, max_depth, min_instances, target_impurity)
        models.append(rand_tree)
    return pd.Series(models)