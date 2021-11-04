import os
import time
import numpy as np
from nltk import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif


def read_data(main_dir):
    neg_dir = main_dir + "/negative_polarity"
    neg_dec_dir = neg_dir + "/deceptive_from_MTurk"
    neg_truth_dir = neg_dir + "/truthful_from_Web"
    train_neg_dec, test_neg_dec = read_dir(neg_dec_dir)
    train_neg_truth, test_neg_truth = read_dir(neg_truth_dir)
    train_neg = np.append(train_neg_dec, train_neg_truth)
    test_neg = np.append(test_neg_dec, test_neg_truth)
    labels_train = np.append(np.zeros((320,), dtype=int), np.ones((320,), dtype=int))
    labels_test = np.append(np.zeros((80,), dtype=int), np.ones((80,), dtype=int))
    return train_neg, labels_train, test_neg, labels_test


def read_dir(main_dir):
    train_reviews = []
    test_reviews = []
    for fold in os.listdir(main_dir):
        # ignore hidden files
        if not fold.startswith('.'):
            if "5" in fold:
                l = test_reviews
            else:
                l = train_reviews
            fold = os.path.join(main_dir, fold)
            for review_path in os.listdir(fold):
                review_path = os.path.join(fold, review_path)
                f = open(review_path, "r")
                processed_review = preprocessing(f.read()[:-1])
                l.append(' '.join(processed_review))
                f.close()
    return np.array(train_reviews), np.array(test_reviews)


def preprocessing(review):
    stemmer = SnowballStemmer("english")
    tokenized_sentence = word_tokenize(review)
    return [stemmer.stem(word) for word in tokenized_sentence if word.isalpha()]


def build_vectorizer(ngram, min_df, max_df):
    return CountVectorizer(ngram_range=(1, ngram), min_df=min_df, max_df=max_df)


def ngrams_train(corpus, vectorizer):
    return vectorizer.fit_transform(corpus)


def ngrams_test(corpus, vectorizer):
    return vectorizer.transform(corpus)

def logistic_regression():
    # perform hyperparameter tuning by k-fold cross validation; values completely random yet!!!
    param_grid = [
        {
            'penalty': ['l2'],
            'C': [0.15, 0.2, 0.25, 0.3, 0.35],
            'solver': ['saga'],
            'max_iter': [1000, 1500]
        }
    ]
    return LogisticRegression(solver='saga', C=0.3, penalty='l2', max_iter=1000)
    # return GridSearchCV(estimator=LogisticRegression(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def classification_tree():
    # perform hyperparameter tuning by k-fold cross validation; values completely random yet!!!
    param_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': np.arange(0, 40, 2).tolist(),
            'min_samples_split': np.arange(0, 80, 2).tolist(),
            'ccp_alpha': np.arange(0, 1, 0.1).tolist()
        }
    ]
    return tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 10, ccp_alpha = 0)
    # return GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def random_forest():
    # perform hyperparameter tuning by k-fold cross validation; values completely random yet!!!
    param_grid = [
        {
            'bootstrap' : [True],
            # nmin
            'min_samples_split' : np.arange(10,51,10).tolist(),
            # minleaf
            'min_samples_leaf' : np.arange(1, 26, 4).tolist(),
            # nfeat
            'max_features' : [50, 187, 750, 3000, 12000, 48000],
            # m
            'n_estimators' : [100]
        }
    ]
    return RandomForestClassifier(bootstrap=True, max_features=750, min_samples_leaf=5, min_samples_split=40, n_estimators=750)
    # return GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def multinomial_NB():
    return MultinomialNB()

def adapt_dataset_to_top_k(x_train, y_train, x_test, k):
    # do top k approach with mutual information
    mutual_information_array = mutual_info_classif(x_train, y_train)
    top_k_feature_indices = np.argpartition(mutual_information_array, -k)[-k:]
    return x_train.todense()[:, top_k_feature_indices], x_test.todense()[:, top_k_feature_indices]


def print_scores(clf_name, y_test, y_pred, start_time, clf_best_params=""):
    print(f"{clf_name} scores\n"
          f" - accuracy = {accuracy_score(y_test, y_pred)}\n"
          f" - precision = {precision_score(y_test, y_pred)}\n"
          f" - recall = {recall_score(y_test, y_pred)} \n"
          f" - F1-score = {f1_score(y_test, y_pred)}\n"
          f"The best parameters for {clf_name} are: {clf_best_params}\n"
          f"--- {clf_name} time {time.time() - start_time} seconds ---\n")


def get_best_features(coefficients, vocab, comparator):
    best_features_score = [0, 0, 0, 0, 0]
    best_features_indexes = [-1, -1, -1, -1, -1]
    len_coef = len(coefficients)
    len_best = len(best_features_score)
    for i in range(0, len_coef):
        c = coefficients[i]
        if comparator(c, best_features_score[0]):
            index = 5
            for j in range(1, len_best):
                if not comparator(c, best_features_score[j]):
                    index = j
                    break
            best_features_score.insert(index, c)
            best_features_score.pop(0)
            best_features_indexes.insert(index, i)
            best_features_indexes.pop(0)

    print(best_features_score)
    reverse_dict = {v: k for k, v in vocab.items()}
    print([reverse_dict[i] for i in best_features_indexes])


def classify(clf_function, x_train, y_train, x_test, y_test):
    start_time = time.time()
    clf = clf_function()
    clf.fit(x_train, y_train)
    labels_pred = clf.predict(x_test)
    correctness = list(map(lambda xy: 1 if xy[0]==xy[1] else 0, zip(y_test, labels_pred)))
    print_scores(clf_function.__name__, y_test, labels_pred, start_time)
    # print_scores(clf_function.__name__, y_test, labels_pred, start_time, clf.best_params_)
    return correctness, clf


def confusion_matrix(a, b):
    ab, TN, a_, b_ = 0, 0, 0, 0
    for i in range(len(a)):
        if a[i] == 1:
            # Positive
            if b[i] == 1:
                ab += 1
            else:
                a_ += 1
        else:
            # Negative
            if b[i] == 1:
                b_ += 1
            else:
                TN += 1
    # accuracy = (ab + TN) / N
    # print(f'accuracy = {accuracy}')
    # precision = ab / (ab + a_)
    # print(f'precision = {precision}')
    # recall = ab / (ab + b_)
    # print(f'recall = {recall}')
    print(f'Confusion matrix')
    print(f'ab: {ab}     a_: {a_}')
    print(f'b_: {b_}     TN: {TN}')


def main():
    start_time = time.time()
    x_train, y_train, x_test, y_test = read_data("./op_spam_v1.4")
    vectorizer = build_vectorizer(2, 0, 0.9)
    x_train = ngrams_train(x_train, vectorizer)
    x_test = ngrams_test(x_test, vectorizer)
    print(f"--- pre-processing time {time.time() - start_time} seconds ---")
    # Multinomial Naive Bayes
    x_train_nb, x_test_nb = adapt_dataset_to_top_k(x_train, y_train, x_test, 12000)
    correctness_naive_bayes, _ = classify(multinomial_NB, x_train_nb, y_train, x_test_nb, y_test)
    # Logistic Regression
    correctness_logistic, clf_log = classify(logistic_regression, x_train, y_train, x_test, y_test)
    # Classification tree
    correctness_tree, _ = classify(classification_tree, x_train, y_train, x_test, y_test)
    # Random forest
    correctness_forest, _ = classify(random_forest, x_train, y_train, x_test, y_test)

    print("---  naive bayes vs logistic  ---")
    confusion_matrix(correctness_naive_bayes, correctness_logistic)
    print("---  forest vs logistic  ---")
    confusion_matrix(correctness_forest, correctness_logistic)
    print("---  forest vs naive bayes ---")
    confusion_matrix(correctness_forest, correctness_naive_bayes)
    print("---  best features ---")
    get_best_features(clf_log.coef_[0], vectorizer.vocabulary_, float.__lt__)
    get_best_features(clf_log.coef_[0], vectorizer.vocabulary_, float.__gt__)

if __name__ == "__main__":
    main()
