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
import matplotlib.pyplot as plt


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
    return tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 19, ccp_alpha = 0)
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

def classify_and_return_scores(clf_function, x_train, y_train, x_test, y_test):
    clf = clf_function()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)



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
    x_train_bigram = ngrams_train(x_train, vectorizer)
    x_test_bigram = ngrams_test(x_test, vectorizer)
    print(f"--- pre-processing time {time.time() - start_time} seconds ---")
    # Multinomial Naive Bayes
    x_train_nb, x_test_nb = adapt_dataset_to_top_k(x_train_bigram, y_train, x_test_bigram, 12000)
    accuracy_nb_bigram, precision_nb_bigram, recall_nb_bigram, f1_score_nb_bigram = classify_and_return_scores(multinomial_NB, x_train_nb, y_train, x_test_nb, y_test)
    # Logistic Regression
    accuracy_lg_bigram, precision_lg_bigram, recall_lg_bigram, f1_score_lg_bigram = classify_and_return_scores(logistic_regression, x_train_bigram, y_train, x_test_bigram, y_test)
    # Classification tree
    accuracy_ct_bigram, precision_ct_bigram, recall_ct_bigram, f1_score_ct_bigram = classify_and_return_scores(classification_tree, x_train_bigram, y_train, x_test_bigram, y_test)
    # Random forest
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    for _ in range(10):
        accuracy_rf, precision_rf, recall_rf, f1_score_rf = classify_and_return_scores(random_forest, x_train_bigram, y_train, x_test_bigram, y_test)
        accuracies.append(accuracy_rf)
        precisions.append(precision_rf)
        recalls.append(recall_rf)
        f1_scores.append(f1_score_rf)
    accuracy_rf_bigram = sum(accuracies) / len(accuracies)
    precision_rf_bigram = sum(precisions) / len(precisions)
    recall_rf_bigram = sum(recalls) / len(recalls)
    f1_score_rf_bigram = sum(f1_scores) / len(f1_scores)
    # visualize all scores on bigrams
    visualize_scores_all_classifiers_bigram(
        [accuracy_nb_bigram, precision_nb_bigram, recall_nb_bigram, f1_score_nb_bigram],
        [accuracy_lg_bigram, precision_lg_bigram, recall_lg_bigram, f1_score_lg_bigram],
        [accuracy_ct_bigram, precision_ct_bigram, recall_ct_bigram, f1_score_ct_bigram],
        [accuracy_rf_bigram, precision_rf_bigram, recall_rf_bigram, f1_score_rf_bigram]
        )
    # visualize unigram vs bigram
    vectorizer = build_vectorizer(1, 0, 0.9)
    x_train_unigram = ngrams_train(x_train, vectorizer)
    x_test_unigram = ngrams_test(x_test, vectorizer)
    print(f"--- pre-processing time {time.time() - start_time} seconds ---")
    # Multinomial Naive Bayes unigram
    x_train_nb, x_test_nb = adapt_dataset_to_top_k(x_train_unigram, y_train, x_test_unigram, 12000)
    accuracy_nb_unigram, precision_nb_unigram, recall_nb_unigram, f1_score_nb_unigram = classify_and_return_scores(multinomial_NB, x_train_nb, y_train, x_test_nb, y_test)
    # Logistic Regression unigram
    accuracy_lg_unigram, precision_lg_unigram, recall_lg_unigram, f1_score_lg_unigram = classify_and_return_scores(logistic_regression, x_train_unigram, y_train, x_test_unigram, y_test)
    # Classification tree unigram
    accuracy_ct_unigram, precision_ct_unigram, recall_ct_unigram, f1_score_ct_unigram = classify_and_return_scores(classification_tree, x_train_unigram, y_train, x_test_unigram, y_test)
    # Random forest unigram
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    for _ in range(10):
        accuracy_rf, precision_rf, recall_rf, f1_score_rf = classify_and_return_scores(random_forest, x_train_unigram, y_train, x_test_unigram, y_test)
        accuracies.append(accuracy_rf)
        precisions.append(precision_rf)
        recalls.append(recall_rf)
        f1_scores.append(f1_score_rf)
    accuracy_rf_unigram = sum(accuracies) / len(accuracies)
    precision_rf_unigram = sum(precisions) / len(precisions)
    recall_rf_unigram = sum(recalls) / len(recalls)
    f1_score_rf_unigram = sum(f1_scores) / len(f1_scores)
    # visualize uni vs bigram
    uni_vs_bi(
        accuracy_nb_unigram,
        accuracy_nb_bigram,
        precision_nb_unigram,
        precision_nb_bigram,
        recall_nb_unigram,
        recall_nb_bigram,
        f1_score_nb_unigram,
        f1_score_nb_bigram)
    uni_vs_bi(
        accuracy_lg_unigram,
        accuracy_lg_bigram,
        precision_lg_unigram,
        precision_lg_bigram,
        recall_lg_unigram,
        recall_lg_bigram,
        f1_score_lg_unigram,
        f1_score_lg_bigram   
    )
    uni_vs_bi(
        accuracy_ct_unigram,
        accuracy_ct_bigram,
        precision_ct_unigram,
        precision_ct_bigram,
        recall_ct_unigram,
        recall_ct_bigram,
        f1_score_ct_unigram,
        f1_score_ct_bigram
    )
    uni_vs_bi(
        accuracy_rf_unigram,
        accuracy_rf_bigram,
        precision_rf_unigram,
        precision_rf_bigram,
        recall_rf_unigram,
        recall_rf_bigram,
        f1_score_rf_unigram,
        f1_score_rf_bigram
    )
    


    # # Multinomial Naive Bayes
    # x_train_nb, x_test_nb = adapt_dataset_to_top_k(x_train, y_train, x_test, 12000)
    # correctness_naive_bayes, _ = classify(multinomial_NB, x_train_nb, y_train, x_test_nb, y_test)
    # # Logistic Regression
    # correctness_logistic, clf_log = classify(logistic_regression, x_train, y_train, x_test, y_test)
    # # Classification tree
    # correctness_tree, _ = classify(classification_tree, x_train, y_train, x_test, y_test)
    # # Random forest
    # correctness_forest, _ = classify(random_forest, x_train, y_train, x_test, y_test)

    # print("---  naive bayes vs logistic  ---")
    # confusion_matrix(correctness_naive_bayes, correctness_logistic)
    # print("---  forest vs logistic  ---")
    # confusion_matrix(correctness_forest, correctness_logistic)
    # print("---  forest vs naive bayes ---")
    # confusion_matrix(correctness_forest, correctness_naive_bayes)
    # print("---  best features ---")
    # get_best_features(clf_log.coef_[0], vectorizer.vocabulary_, float.__lt__)
    # get_best_features(clf_log.coef_[0], vectorizer.vocabulary_, float.__gt__)

def visualize_scores_all_classifiers_bigram(nb_scores, lg_scores, ct_scores, rf_scores):
    # visualize accuracy, precision, recall and F1 score on bigrams
    print(f"nb scores are: {nb_scores}")
    print(f"lg_scores are: {lg_scores}")
    print(f"ct_scores are: {ct_scores}")
    print(f"rf_scores are: {rf_scores}")
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    w = 0.2
    bar1 = np.arange(len(labels))
    bar2 = [i+w for i in bar1]
    bar3 = [i+w for i in bar2]
    bar4 = [i+w for i in bar3]

    nb_bars = plt.bar(bar1, nb_scores, w, label="Naive Bayes")
    for bar in nb_bars:
        bar.set_color("#308089")
    lg_bars = plt.bar(bar2, lg_scores, w, label="Logistic Regression")
    for bar in lg_bars:
        bar.set_color("#D2D534")
    ct_bars = plt.bar(bar3, ct_scores, w, label="Classification Tree")
    for bar in ct_bars:
        bar.set_color("#A82BD7")
    rf_bars = plt.bar(bar4, rf_scores, w, label="Random Forest")
    for bar in rf_bars:
        bar.set_color("#E09521")
    
    plt.xlabel("Measure")
    plt.ylabel("Score")
    plt.xticks(bar1 + 1.5 * w, labels)
    plt.ylim(0.7, 0.95)
    plt.legend()
    plt.title = "Accuracy, Precision, Recall and F1 Score on bigrams"
    plt.show()

def uni_vs_bi(accuracy_uni, accuracy_bi, precision_uni, precision_bi, recall_uni, recall_bi, f1_score_uni, f1_score_bi):
    print(f"uni scores are: {accuracy_uni, precision_uni, recall_uni, f1_score_uni}")
    print(f"bi scores are: {accuracy_bi, precision_bi, recall_bi, f1_score_bi}")
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    w = 0.2
    bar1 = np.arange(len(labels))
    bar2 = [i+w for i in bar1]

    uni_bars = plt.bar(bar1, [accuracy_uni, precision_uni, recall_uni, f1_score_uni], w, label="Unigram")
    for bar in uni_bars:
        bar.set_color("#308089")
    bi_bars = plt.bar(bar2, [accuracy_bi, precision_bi, recall_bi, f1_score_bi], w, label="Bigram")
    for bar in bi_bars:
        bar.set_color("#D2D534")

    
    plt.xlabel("Measure")
    plt.ylabel("Score")
    plt.xticks(bar1 + w, labels)
    plt.ylim(0.5, 0.95)
    plt.legend()
    plt.title = "Unigram vs Bigram"
    plt.show()




if __name__ == "__main__":
    main()
