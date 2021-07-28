"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of evaluation functions for scoring the population
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy.optimize import linear_sum_assignment
import re
import sys
import copy

def false_positive(individual, test_data, truth_data, name=None):
    """
    False positive is test_data == 1 when truth_data == 0
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false positive means predicting greater than 50k when the individual made less
    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
    Returns:
        objective score
    """
    # Put test data in the same form as truth data from buildClassifier
    test_data = np.array([elem[0] for elem in test_data])
    truth_data = np.array(truth_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    # FP = np.sum(test_data[truth_data==0] != 0)
    # FPR = FP / FP + TP
    TP = np.sum(test_data[truth_data == 1] == 1)
    FP = np.sum(test_data[truth_data == 0] != 0)
    TN = np.sum(test_data[truth_data == 0] == 0)
    FN = np.sum(test_data[truth_data == 1] != 1)
    return FP / (FP + TN)
def false_negative(individual, test_data, truth_data, name=None):
    """
    False negative is test_data == 0 when truth_data == 1
    For example adult dataset a 0 represents <= 50,000 while a 1 represents >50,000
    A false negative means predicting less than 50k when the individual made more
    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
    Returns:
        objective score
    """
    # Put test data in the same form as truth data from buildClassifier
    test_data = np.array([elem[0] for elem in test_data])
    truth_data = np.array(truth_data)
    if truth_data.shape != test_data.shape:
        return np.inf
    TP = np.sum(test_data[truth_data == 1] == 1)
    FP = np.sum(test_data[truth_data == 0] != 0)
    TN = np.sum(test_data[truth_data == 0] == 0)
    FN = np.sum(test_data[truth_data == 1] != 1)
    return FN / (FN + TP)

def roc_auc(individual, test_data, truth_data, name=None):
    """Returns area under receiver operating characteristic

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return roc_auc_score(truth_data, test_data)

def precision_auc(individual, test_data, truth_data, name=None):
    """Returns area under precision-recall curve

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return average_precision_score(truth_data, test_data)

def f1_score_min(individual, test_data, truth_data, name=None):
    """Returns F1-Score based on precision and recall

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return 1 - f1_score(truth_data, test_data)

def objective0EvalFunction(individual, test_data, truth_data, name=None):
    """RMS Error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return np.sqrt(np.mean((test_data-truth_data)**2))

def objective1EvalFunction(individual, test_data, truth_data, name=None):
    """Over Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    differences = truth_data - test_data
    overPrediction = differences < 0
    differences = np.array(differences)
    # If there were no under predictions, return 0 error
    if not any(overPrediction):
        return 0.0
    else:
        return np.mean(np.abs(differences[overPrediction]))

def objective2EvalFunction(individual, test_data, truth_data, name=None):
    """Under Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    differences = truth_data - test_data
    underPrediction = differences > 0
    differences = np.array(differences)
    # If there were no over predictions, return 0 error
    if not any(underPrediction):
        return 0.0
    else:
        return np.mean(differences[underPrediction])

def objective3EvalFunction(individual, test_data, truth_data, name=None):
    """Scores by height of individual tree

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    return individual.height

def objective4EvalFunction(individual, test_data, truth_data, name=None):
    """
    Probability of "Detection"
    Probability of prediciting within 1 decimeter

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()
    differences = truth_data - test_data
    under_one_decimeter = np.array(np.abs(differences) <= np.sqrt(0.5**2 + (0.013 * truth_data)**2) - 0.4)

    return 1.0-float(np.sum(under_one_decimeter))/len(under_one_decimeter)

def objective5EvalFunction(indivdual, test_data, truth_data, name=None):
    """Overall mean percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()

    differences = abs(truth_data - test_data)
    percents = differences/truth_data
    return abs(np.mean(percents))

def objective6EvalFunction(indivdual, test_data, truth_data, name=None):
    """Valid Overall mean percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()

    test_data[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
    # test_data[test_data == -1] = np.nan
    differences = abs(truth_data - test_data)
    percents = differences/truth_data
    return abs(np.nanmean(percents))

def objective7EvalFunction(individual, test_data, truth_data, name=None):
    """Valid RMS Error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()

    test_data = copy.deepcopy(test_data)
    test_data[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
    # testData[testData == -1] = np.nan
    return np.sqrt(np.nanmean((test_data-truth_data)**2))

def objective8EvalFunction(individual, test_data, truth_data, name=None):
    """Valid Over Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()

    differences = truth_data - test_data
    overPrediction = differences < 0
    differences = np.array(differences)
    # If there were no under predictions, return 0 error
    if not any(overPrediction):
        return 0.0
    else:
        differences[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
        # differences[testData == -1] = np.nan
        return np.nanmean(np.abs(differences[overPrediction]))

def objective9EvalFunction(individual, test_data, truth_data, name=None):
    """Valid Under Prediction

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()

    differences = truth_data - test_data
    underPrediction = differences > 0
    differences = np.array(differences)
    # If there were no over predictions, return 0 error
    if not any(underPrediction):
        return 0.0
    else:
        differences[np.logical_or(np.logical_and(test_data < 0, truth_data != -1), np.logical_and(test_data >= 0, truth_data == -1))] = np.nan
        # differences[testData < 0] = np.nan
        return np.nanmean(differences[underPrediction])

def objective10EvalFunction(individual, test_data, truth_data, name=None):
    """Number of individuals that could not be evaluated properly

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data).flatten()
    # negatives = testData == -1
    # false_array = np.array(testData)[negatives]
    # return np.sum(testData[testData == -1]) * -1
    return np.sum(test_data < 0)

def objective11EvalFunction(individual, test_data, truth_data, name=None):
    """
    False Positive Bottom

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # negatives = testData == -1
    # false_array = np.array(testData)[negatives]
    # return np.sum(testData[testData == -1]) * -1
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()
    return np.sum(np.logical_and(test_data >= 0, truth_data == -1))

def objective12EvalFunction(individual, test_data, truth_data, name=None):
    """
    False Negative Bottom

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # negatives = testData == -1
    # false_array = np.array(testData)[negatives]
    # return np.sum(testData[testData == -1]) * -1
    test_data = np.array(test_data).flatten()
    truth_data = np.array(truth_data).flatten()
    return np.sum(np.logical_and(test_data < 0, truth_data != -1))


def class0AccuracyEvalFunction(individual, test_data, truth_data, name=None):
    """
    Error in predicting class 0
    For EEG data, classes are 0 and 4

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        if truth_point == 0:
            if test_point < 0 or test_point > 2:
                num_wrong += 1
            total += 1
    if num_wrong == 0:
        # We don't want 'perfect' equate it with 100% error
        return 1.0
    else:
        return float(num_wrong)/float(total)

def class4AccuracyEvalFunction(individual, test_data, truth_data, name=None):
    """
    Error in predicting class 4
    For EEG data, classes are 0 and 4

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        if truth_point == 4:
            if test_point <= 2 or test_point > 4:
                num_wrong += 1
            total += 1
    if num_wrong == 0:
        # We don't want 'perfect' equate it with 100% error
        return 1.0
    else:
        return float(num_wrong)/float(total)

def drinking_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with drinking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Two represents a drinking event, Three is eating, One is Chewing
        if (truth_point == 2):
            if np.isnan(test_point) or test_point <= 1.5 or test_point > 2.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def drinking_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with drinking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Two represents a drinking event
        if truth_point != 2:
            if test_point > 1.5 and test_point <= 2.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def breadth_eval_function(individual, test_data, truth_data, name=None):
    """
    This function determines a metric for breadth by tracking how many times
    ARG0 appears in the individual.  This allows for competition with fitter solutions.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Compute the string representation of the individual
    string_rep = str(individual)
    # Generate array for appearances of ARG0
    data_appearances = [match.start() for match in re.finditer('ARG0', string_rep)]
    return -1.0*len(data_appearances)


def depth_breadth_eval_function(individual, test_data, truth_data, name=None):
    """
    A shallower tree should always beat a deeper tree.
    Given all else equal, a wider tree should beat a narrower tree.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Compute the string representation of the individual
    string_rep = str(individual)
    # Generate array for appearances of ARG0
    data_appearances = [match.start() for match in re.finditer('ARG0', string_rep)]
    # Computed by counting occurrences of GTMOEPDataPair per primitive
    # Let's make this dynamic in the future by querying all functions
    max_breadth = 3.0
    # Tradeoff formula
    return individual.height - len(data_appearances)/(max_breadth**individual.height + 1.0)

def num_elements_eval_function(individual, test_data, truth_data, name=None):
    """
    The fewer elements in the tree the better

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    # Let's take the sum of all elements in the tree where the name does not contain adf_
    num_elements = 0
    for tree in individual:
        num_elements += np.sum([1 if 'adf_' not in elem.name else 0 for elem in tree])

    # Let's now put the test cases on the individual to be used for fuzzy selection
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    test_case_vec = np.hstack((test_case_vec, num_elements*np.ones(len(test_data))))
    setattr(individual, name, test_case_vec)

    return num_elements

def num_elements_eval_function_capped(individual, test_data, truth_data, name=None):
    """The fewer elements in the tree the better

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    return max(len(individual), 1707)

def shaking_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with shaking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Nine represents a shaking event
        if (truth_point == 9):
            if np.isnan(test_point) or test_point <= 8.5 or test_point > 9.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def shaking_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with shaking

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Nine represents a shaking event
        if truth_point != 9:
            if test_point > 8.5 and test_point <= 9.5:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    return float(num_wrong)/float(total)

def scratching_error_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate "1 - PD"
    For dog behavioral data in predicting the class number
    associated with scratching

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    #import pdb; pdb.set_trace()
    test_data = np.array([elem[0] for elem in test_data])
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Eight represents a scratching event
        #if (truth_point == 8):
        if (truth_point == 1):
            if np.isnan(test_point) or test_point != 1:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    # The second == 1 is ignoring things that are not 0 nor 1
    test_case_vec = np.hstack((test_case_vec, np.logical_not(test_data[truth_data==1] == 1)))
    setattr(individual, name, test_case_vec)

    return float(num_wrong)/float(total)


def scratching_false_alarm_rate(individual, test_data, truth_data, name=None):
    """
    This function assesses the error rate of false alarms
    For dog behavioral data in predicting the class number
    associated with scratching

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array([elem[0] for elem in test_data])
    truth_data = np.array(truth_data)
    total = 0
    num_wrong = 0
    for test_point, truth_point in zip(test_data, truth_data):
        # Eight represents a scratching event
        #if truth_point != 8:
        if truth_point != 1:
            #if test_point > 7.5 and test_point <= 8.5:
            if test_point == 1:
                num_wrong += 1
            total += 1
    #if num_wrong == 0:
    #    # Perfection implies overtraining
    #    return 1.0
    #else:
    # First get what's there
    test_case_vec = getattr(individual, name)
    # Now stick on what's new
    # The == 1 is ignoring things that are not 0 nor 1
    test_case_vec = np.hstack((test_case_vec, test_data[truth_data == 0] == 1))
    setattr(individual, name, test_case_vec)

    return float(num_wrong)/float(total)


def get_over_predicted_inds(test_data, truth_data, name=None, tolerance=0):
    """Returns over predicted individuals

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return np.nonzero(test_data > truth_data + tolerance)[0]


def get_under_predicted_inds(test_data, truth_data, name=None, tolerance=0):
    """Returns under predicted individuals

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return np.nonzero(test_data < truth_data - tolerance)[0]


def count_over_predictions(individual, test_data, truth_data, name=None, tolerance=0):
    """Counts over predictions

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return len(get_over_predicted_inds(test_data, truth_data, tolerance)) / float(len(test_data))


def count_under_predictions(individual, test_data, truth_data, name=None, tolerance=0):
    """Counts under predictions

    Args:
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return len(get_under_predicted_inds(test_data, truth_data, tolerance)) / float(len(test_data))


def overall_standard_deviation(individual, test_data, truth_data, name=None):
    """Evaluates overall standard deviation

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return np.std(test_data - truth_data)


def standard_deviation_over(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates standard deviation of over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return overall_standard_deviation(individual, test_subset, truth_subset)


def standard_deviation_under(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates standard deviation of under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return overall_standard_deviation(individual, test_subset, truth_subset)


def average_percent_error(individual, test_data, truth_data, name=None):
    """Evaluates average percent error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    return np.mean(np.abs(test_data - truth_data) / truth_data)


def average_precent_error_over(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates average percent error for over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return average_percent_error(individual, test_subset, truth_subset)


def average_precent_error_under(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates average percent error for under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return average_percent_error(individual, test_subset, truth_subset)


def max_over_prediction_error(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates max for over predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    over_predicted_inds = get_over_predicted_inds(test_data,
                                                  truth_data,
                                                  tolerance)
    if len(over_predicted_inds) == 0:
        return np.nan
    test_subset = test_data[over_predicted_inds]
    truth_subset = truth_data[over_predicted_inds]
    return np.max(test_subset - truth_subset)


def max_under_prediction_error(individual, test_data, truth_data, name=None, tolerance=0):
    """Evaluates max for under predicted individuals

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter
        tolerance: tolerance threshold

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    under_predicted_inds = get_under_predicted_inds(test_data,
                                                    truth_data,
                                                    tolerance)
    if len(under_predicted_inds) == 0:
        return np.nan
    test_subset = test_data[under_predicted_inds]
    truth_subset = truth_data[under_predicted_inds]
    return np.max(truth_subset - test_subset)


def continuous_mse(individual, test_data, truth_data, name=None):
    """Evaluates continuous mean squared error

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    error = 0

    for test_item, truth_item in zip(test_data, truth_data):
        error += np.sqrt(np.mean(np.square(truth_item - test_item)))

    return error

def continuous_bias(individual, test_data, truth_data, name=None):
    """Evaluates continuous bias

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    error = 0
    for test_item, truth_item in zip(test_data, truth_data):
        error += np.mean(np.abs(test_data - truth_data))
    return error

def continuous_var(individual, test_data, truth_data, name=None):
    """Evaluates continuous variance

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    test_data = np.array(test_data)
    truth_data = np.array(truth_data)
    error = 0
    for test_item, truth_item in zip(test_data, truth_data):
        error += np.var(test_data - truth_data)
    return error

def _partition_distance(p1, p2, n):
    """
    Calculates the partition distance D(p1, p2), defined as the minimum number
    of elements that need to be removed from both partitions to make them
    identical. p1 and p2 should both be a list of lists that form a partition
    of the numbers {0,...,n - 1}. Uses a O(n^3) reduction to a weighted
    assignment problem by Konovalov, Litow and Bajema (2005).
    (https://academic.oup.com/bioinformatics/article/21/10/2463/208566)

    Args:
        p1: list of partitions
        p2: list of partitions
        n: total number of distinct elements

    Returns:
        partition distance between p1 and p2
    """
    m = max(len(p1), len(p2))
    p1_str = np.zeros((m, n), dtype=int)
    for i, x in enumerate(p1):
        for v in x:
            p1_str[i][v] = 1
    p2_str = np.ones((m, n), dtype=int)
    for i, x in enumerate(p2):
        for v in x:
            p2_str[i][v] = 0
    cost = np.array([[np.sum(a * b) for b in p2_str] for a in p1_str])
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()

def _fast_partition_distance(p1_list, p2_list, n):
    """
    Calculates the partition distance D(p1, p2), defined as the minimum number
    of elements that need to be removed from both partitions to make them
    identical. p1 and p2 should both be a list of lists that form a partition
    of the numbers {0,...,n - 1}. This method will fail and return 'None' if
    D(p1,p2) >= n / 5. Uses an O(n) algorithm given by Porumbel, Hao and Kuntz (2009).
    (https://www.sciencedirect.com/science/article/pii/S0166218X10003069)


    Args:
        p1: list of partitions
        p2: list of partitions
        n: total number of distinct elements

    Returns:
        partition distance between p1 and p2, or None if the distance is >= n / 5
    """
    p1 = {}
    for i, box in enumerate(p1_list):
        for x in box:
            p1[x] = i
    p2 = {}
    for i, box in enumerate(p2_list):
        for x in box:
            p2[x] = i

    k = max(len(p1_list), len(p2_list))
    similarity = 0
    t = np.empty((k, k), dtype=int)
    m = np.zeros(k, dtype=int)
    sigma = np.zeros(k, dtype=int)
    size_p1 = np.zeros(k, dtype=int)
    size_p2 = np.zeros(k, dtype=int)
    for x in range(n):
        t[p1[x], p2[x]] = 0
    for x in range(n):
        i = p1[x]
        j = p2[x]
        t[i, j] += 1
        size_p1[i] += 1
        size_p2[j] += 1
        if t[i, j] > m[i]:
            m[i] = t[i, j]
            sigma[i] = j
    for i in range(k):
        if m[i] != 0:
            if 3*m[i] <= size_p1[i] +  size_p2[sigma[i]]:
                return None
            similarity = similarity + t[i,sigma[i]]
    return n - similarity

def cluster_partition_distance(individual, test_data, truth_data, name=None):
    """Returns normalized partition distance for labeled cluster data

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    distance_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of target data
        test_clusters = test_clusters[-1].flatten()

        p1_dict = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_dict:
                p1_dict[x] = []
            p1_dict[x].append(i)

        p2_dict = {}
        for i, x in enumerate(truth_clusters):
            if x not in p2_dict:
                p2_dict[x] = []
            p2_dict[x].append(i)

        p1 = list(p1_dict.values())
        p2 = list(p2_dict.values())
        d = _fast_partition_distance(p1, p2, len(test_clusters))
        if d is None:
            d = _partition_distance(p1, p2, len(test_clusters))
        distance_sum += d
        max_sum += len(test_clusters) - 1
    return distance_sum / max_sum

def cluster_error1(individual, test_data, truth_data, name=None):
    """
    Returns normalized type 1 error for labeled cluster data.
    Normalized number of points that should be in the same cluster that are in different clusters.
    Has a minimum value of 0, and maximum value of 1.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    error_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of target data
        test_clusters = test_clusters[-1].flatten()

        p1_clusters = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_clusters:
                p1_clusters[x] = []
            p1_clusters[x].append(i)

        p2_map = {}
        for i, x in enumerate(truth_clusters):
            p2_map[i] = x

        num_error = 0
        max_error = 0
        for a in list(p1_clusters.values()):
            for i in range(len(a)):
                x = a[i]
                for j in range(len(a)):
                    if i != j:
                        y = a[j]
                        max_error += 1
                        if p2_map[x] != p2_map[y]:
                            num_error += 1
        error_sum += num_error
        max_sum += max_error
    return 0 if max_sum == 0 else error_sum / max_sum

def cluster_error2(individual, test_data, truth_data, name=None):
    """
    Returns normalized type 2 error for labeled cluster data.
    Normalized number of points that should be in different clusters that are in the same cluster.
    Has a minimum value of 0, and maximum value of 1.

    Args:
        individual: individual to score
        test_data: test data
        truth_data: labeled data
        name: optional name parameter

    Returns:
        objective score
    """
    error_sum = 0
    max_sum = 0
    for test_clusters, truth_clusters in zip(test_data, truth_data):
        # Get last column of test data
        test_clusters = test_clusters[-1].flatten()

        p1_clusters = {}
        for i, x in enumerate(test_clusters):
            if x not in p1_clusters:
                p1_clusters[x] = []
            p1_clusters[x].append(i)

        p2_map = {}
        for i, x in enumerate(truth_clusters):
            p2_map[i] = x

        num_error = 0
        max_error = 0
        p1 = list(p1_clusters.values())
        for i in range(len(p1)):
            a = p1[i]
            for j in range(len(p1)):
                if i != j:
                    b = p1[j]
                    for x in a:
                        for y in b:
                            max_error += 1
                            if p2_map[x] == p2_map[y]:
                                num_error += 1
        error_sum += num_error
        max_sum += max_error
    return 0 if max_sum == 0 else error_sum / max_sum
