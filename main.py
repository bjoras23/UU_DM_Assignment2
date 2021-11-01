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
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.05, 0.1],
            'solver': ['saga'],
            'max_iter': [100, 200]
        }
    ]
    return LogisticRegression(solver='saga', C=0.2, penalty='l2', max_iter=1000)
    # return GridSearchCV(estimator=LogisticRegression(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def classification_tree():
    # perform hyperparameter tuning by k-fold cross validation; values completely random yet!!!
    param_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(1, 101, 20).tolist(),
            'min_samples_split': np.arange(5, 31, 5).tolist(),
            'min_samples_leaf': np.arange(5, 56, 10).tolist()
        }
    ]
    return tree.DecisionTreeClassifier()
    # return GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def random_forest():
    # perform hyperparameter tuning by k-fold cross validation; values completely random yet!!!
    param_grid = [
        {
            'bootstrap': [True],
            'max_depth': [20, 100],
            'min_samples_split': np.arange(5, 31, 5).tolist(),
            'min_samples_leaf': np.arange(5, 56, 10).tolist(),
            'max_features': [100, 1000, 10000],
            'n_estimators': [10, 50, 100]
        }
    ]
    return RandomForestClassifier()
    # return GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3)


def multinomial_NB():
    return MultinomialNB()


def print_scores(clf_name, y_test, y_pred, start_time, clf_best_params=""):
    print(f"{clf_name} scores\n"
          f" - accuracy = {accuracy_score(y_test, y_pred)}\n"
          f" - precision = {precision_score(y_test, y_pred)}\n"
          f" - recall = {recall_score(y_test, y_pred)} \n"
          f" - F1-score = {f1_score(y_test, y_pred)}\n"
          f"The best parameters for logistic regression are: {clf_best_params}"
          f"--- {clf_name} time {time.time() - start_time} seconds ---")


def classify(clf_function, x_train, y_train, x_test, y_test):
    start_time = time.time()
    clf = clf_function()
    clf.fit(x_train, y_train)
    labels_pred = clf.predict(x_test)
    print_scores(clf_function.__name__, y_test, labels_pred, start_time)
    # print_scores(clf_function.__name__, y_test, labels_pred, start_time, clf.best_params_)


def main():
    start_time = time.time()
    x_train, y_train, x_test, y_test = read_data("./op_spam_v1.4")
    vectorizer = build_vectorizer(2, 0, 0.9)
    x_train = ngrams_train(x_train, vectorizer)
    x_test = ngrams_test(x_test, vectorizer)
    print(f"--- pre-processing time {time.time() - start_time} seconds ---")
    # Logistic Regression
    classify(logistic_regression, x_train, y_train, x_test, y_test)
    # Classification tree
    classify(classification_tree, x_train, y_train, x_test, y_test)
    # Random forest
    classify(random_forest, x_train, y_train, x_test, y_test)
    # Multinomial Naive Bayes
    classify(multinomial_NB, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
