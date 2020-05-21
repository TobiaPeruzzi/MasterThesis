import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV  # used for cross-validation
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score  # confusion matrix, accuracy
from sklearn.manifold import TSNE

seed = 42

def SVM_clf(X_train, y_train, X_validation, y_validation, clf_C, clf_gamma='auto', clf_kernel='linear',
            clf_degree=3, weights='balanced'):
    """
    :param X_train: training set
    :param y_train: labels for training set
    :param X_validation: validation set
    :param y_validation: labels for validation set
    :param clf_C: regularization parameter for SVM
    :param clf_gamma: gamma parameter for SVM (ignored for linear kernel). Default:'auto'
    :param clf_kernel: kernel for SVM. Options are: 'linear', 'rbf', 'poly'. Default: 'linear'
    :param clf_degree: degree for polynomial kernel. Ignored for rbf and linear kernels. Default: 3
    :param weights: wheter or not to assign weights to the classes. Default: 'balanced'
    :return: returns confusion matrix, confusion matrix normalized for SVM and SVM classifier
    """

    print("Results for " + clf_kernel + " kernel: \n \n")

    test_SVM = svm.SVC(kernel=clf_kernel, C=clf_C, gamma=clf_gamma, class_weight=weights, probability=True,
                       degree=clf_degree)
    test_SVM.fit(X_train, y_train)

    # calculating training and validation errors
    training_error = 1. - test_SVM.score(X_train, y_train)
    validation_error = 1. - test_SVM.score(X_validation, y_validation)

    print("Training error: %f" % training_error)
    print("Validation error: %f" % validation_error, "\n \n")

    # SVM prediction and confusion matrix for validation

    prediction = test_SVM.predict(X_validation)

    con_mat = confusion_matrix(y_validation, prediction)

    np.set_printoptions(precision=2, suppress=True)

    u, counts = np.unique(y_validation, return_counts=True)

    norm_con_mat = con_mat / counts[:, None]

    accuracy_balanced = balanced_accuracy_score(y_validation, prediction)
    accuracy = accuracy_score(y_validation, prediction)

    print("Labels and frequencies in test set: ", counts, '\n')

    print('Accuracy score on validation: ', accuracy, '\n')
    print('Balanced accuracy on validation: ', accuracy_balanced, '\n')

    print("Confusion matrix SVM  \n \n", con_mat, '\n')
    print("Confusion matrix SVM (normalized)   \n \n", norm_con_mat)

    return con_mat, norm_con_mat, test_SVM, prediction

def heatmap_cl(input_confusion_matrix, xlabels=['Type 1', 'Type 2', 'Int'], ylabels=['Type 1', 'Type 2', 'Int'],
              savefigure=False, filename=''):
    """
    :param input_confusion_matrix: confusion matrix to plot. Can be normalized or not
    :param xlabels: labels for x-axis (predicted labels)
    :param ylabels: labels for y-axis (true labels)
    :param savefigure: if you want to save the figure set to True. Default: False
    :param filename: name of the figure to save. Recommended format: .pdf
    :return:
    """

    sns_plot = sn.heatmap(input_confusion_matrix, annot=True, xticklabels=xlabels, yticklabels=ylabels)
    sns_heatmap = sns_plot.get_figure()
    plt.tight_layout()
    plt.title('')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if savefigure==True:
        sns_heatmap.savefig(filename)
    else:
        pass

def subset_creation(input_data, input_labels):
    """
    Divides the dataset into train, validation and test sets

    :param input_data: input dataset to split
    :param input_labels: input labels
    :return: training, validation and test sets with corresponding labels
    """

    # preparing training, validation and test sets
    m_test = (len(input_data) * 20) / 100
    m_validation = int(((len(input_data) - m_test) * 20) / 100)
    m_train = int(len(input_data) - m_test - m_validation)

    # random permutation
    permutation = np.random.permutation(input_data.shape[0])

    input_data = input_data[permutation]
    input_labels = input_labels[permutation]

    # features
    X_train = input_data[:m_train]
    X_validation = input_data[m_train:(m_validation + m_train)]
    X_test = input_data[(m_validation + m_train):]

    # labels
    y_train = input_labels[:m_train]
    y_validation = input_labels[m_train:(m_validation + m_train)]
    y_test = input_labels[(m_validation + m_train):]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def TSNE_dim_reduction(input_data, perp=5, components=2):
    """
    Performs t-SNE dimensionality reduction and plots embedded space
    :param data: dataset to give as input to t-SNE
    :param perp: value of the perplexity for t-SNE. Default: 5
    :param components: number of components to keep. Default: 2
    :return: returns X_embedded matrix that contains the embedded coordinates
    """
    # calculating embedded space
    X_tsne = input_data
    tsne = TSNE(n_components=components, perplexity=perp)
    X_embedded = tsne.fit_transform(X_tsne)

    return X_embedded

def randomsearch_cv(parameters, classifier, input_data, data_labels, num_iter=10, cv_index=3):
    """
    Performs random hyperparameters optimization using cross validation.
    :param parameters: dict of parameters on which to perform random search
    :param classifier: choosen classifier
    :param input_data: data to perform random search
    :param data_labels: labels of input_data
    :param num_iter: n_iter for RandomizedSearch
    :param cv_index: cv for RandomizedSearch
    :return: returns set of best parameters found and results of crossvalidation. The latter can be saved in a pandas dataframe
    """

    input_parameters = parameters

    hyperparam_clf = classifier

    random_search = RandomizedSearchCV(hyperparam_clf, input_parameters, n_iter=num_iter, cv=cv_index)

    search = random_search.fit(input_data, data_labels)

    best_param_set = search.best_params_
    print("Best parameters set found: ", best_param_set)

    crossv_results = search.cv_results_

    return best_param_set, crossv_results

def saliency_map(clf, input_data, i_th, num_classes, perturbation=0.01):
    """
    Calculates saliency map for given sample.
    :param clf: pretrained classifier
    :param input_data: data over which calculate saliency map. For example the validation subset can be used
    :param i_th: i-th sample of the input_data
    :param num_classes: number of classes in the dataset. Must be 2 or 3
    :param perturbation: magnitude of the perturbation to introduce. Default: 0.01
    :return: returns saliency map
    """
    if num_classes == 2:
        # array of prediction probabilities
        class_prob = clf.predict_proba(input_data)

        test1 = np.empty_like(input_data[i_th])
        test2 = np.empty_like(input_data[i_th])

        # epsilon is the perturbation parameter
        epsilon = perturbation
        saliency_map = np.vstack((test1, test2))
        initial_prob = class_prob[i_th]
        for i in range(len(input_data[i_th])):
            test_spectrum = np.copy(input_data[i_th])
            test_spectrum[i] += epsilon * test_spectrum[i]
            final_prob = clf.predict_proba(test_spectrum.reshape(1, -1))
            saliency_map[0, i] = (final_prob[0, 0] - initial_prob[0]) / (test_spectrum[i] - input_data[i_th, i])
            saliency_map[1, i] = (final_prob[0, 1] - initial_prob[1]) / (test_spectrum[i] - input_data[i_th, i])
    elif num_classes == 3:
        # array of prediction probabilities
        class_prob = clf.predict_proba(input_data)

        test1 = np.empty_like(input_data[i_th])
        test2 = np.empty_like(input_data[i_th])
        test3 = np.empty_like(input_data[i_th])

        # epsilon is the perturbation parameter
        epsilon = perturbation
        saliency_map = np.vstack((test1, np.vstack((test2, test3))))
        initial_prob = class_prob[i_th]
        for i in range(len(input_data[i_th])):
            test_spectrum = np.copy(input_data[i_th])
            test_spectrum[i] += epsilon * test_spectrum[i]
            final_prob = clf.predict_proba(test_spectrum.reshape(1, -1))
            saliency_map[0, i] = (final_prob[0, 0] - initial_prob[0]) / (test_spectrum[i] - input_data[i_th, i])
            saliency_map[1, i] = (final_prob[0, 1] - initial_prob[1]) / (test_spectrum[i] - input_data[i_th, i])
            saliency_map[2, i] = (final_prob[0, 2] - initial_prob[2]) / (test_spectrum[i] - input_data[i_th, i])
    else:
        raise AttributeError('num_classes must be 2 or 3')
    return saliency_map, class_prob

def color_coding(input_saliency_map, num_classes):
    """
    Calculates color coding saliency map
    :param input_saliency_map: saliency map from which perform color coding
    :param num_classes: number of classes involved
    :return: returns color coded saliency map
    """
    
    color_codes = np.empty_like(input_saliency_map)

    for j in range(0, num_classes):
        for i in range(color_codes.shape[1]):
            if input_saliency_map[j, i] == 0:
                color_codes[j, i] = 0
            elif input_saliency_map[j, i] > 0:
                color_codes[j, i] = 1
            elif input_saliency_map[j, i] < 0:
                color_codes[j, i] = 2
            else:
                print('Error occurred')

    return color_codes