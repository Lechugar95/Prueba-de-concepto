__title__ = ''
__author__ = 'Claudio Mori'
__copyright__ = 'Copyright 2020, Thesis Project'

import numpy  as np
import pandas as pd
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
import seaborn as sn
import matplotlib.pyplot as plt


def Randomized_table_plot(rs_clf, param_name, num_results=15, negative=True, graph=True, display_all_params=True):
    """Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          """
    from matplotlib import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = rs_clf.best_estimator_
    clf_params = rs_clf.best_params_
    if negative:
        clf_score = -rs_clf.best_score_
    else:
        clf_score = rs_clf.best_score_
    clf_stdev = rs_clf.cv_results_['std_test_score'][rs_clf.best_index_]
    cv_results = rs_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results).sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()


# Evaluamos la grilla búsqueda aleatoria (Randomized Search con CV) y la grilla de busqueda (Grid Search con CV)
def evaluate(model_name, model, testing_features, test_labels):
    np.seterr(divide='ignore', invalid='ignore')
    predictions = model.predict(testing_features)
    accuracy = acc(test_labels, predictions)
    error_rate = 1 - accuracy
    precision_score = precision(test_labels, predictions)
    recall_score = recall(test_labels, predictions)
    f1_score = f1(test_labels, predictions)
    print('Model Performance del' + model_name)
    print('Error Rate: {:0.6f} degrees.'.format(error_rate))
    print('Accuracy = {:0.6f}%.'.format(accuracy))
    print('Precision = {:0.6f}%.'.format(precision_score))
    print('Recall = {:0.6f}%.'.format(recall_score))
    print('F1 Score = {:0.6f}%.'.format(f1_score))

    return predictions, error_rate, accuracy, precision_score, recall_score, f1_score
