import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE203", "TYPE"]
df = pd.read_csv("data/glass.data", names=names)

X = df.drop("TYPE", axis=1)  # Features
y = df["TYPE"]  # Target variable

for test_size in range(1, 9):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/10, random_state=42)

    preprocessings = {
        "Normalizer": Normalizer(),
        "StandardScaler": StandardScaler()
    }

    classifiers = {
        "GaussianNB": GaussianNB(),
        "DecisionTreeClassifier(max_depth=3)": DecisionTreeClassifier(max_depth=3),
        "DecisionTreeClassifier(max_depth=5)": DecisionTreeClassifier(max_depth=5),
        "DecisionTreeClassifier(max_depth=7)": DecisionTreeClassifier(max_depth=7),
    }

    pca = {
        "PCA(n_components=3)": PCA(n_components=3),
        "PCA(n_components=3)": PCA(n_components=4),
        "PCA(n_components=3)": PCA(n_components=5)
    }

    for classifier_name, classifier in classifiers.items():
        print("\n", classifier_name, "train size:", test_size*10, "%")
        for preprocessing_name, preprocessing in preprocessings.items():
            X_train_preprocessed = preprocessing.fit_transform(X_train)
            X_test_preprocessed = preprocessing.transform(X_test)

            classifier.fit(X_train_preprocessed, y_train)
            y_pred = classifier.predict(X_test_preprocessed)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            confusion_mat = confusion_matrix(y_test, y_pred)

            print(preprocessing_name)
            print("Accuracy: {:.2f}%".format(accuracy * 100))
            print("Precision: {:.2f}%".format(precision * 100))
            print("Recall: {:.2f}%".format(recall * 100))
            #print("Confusion Matrix:")
            #print(confusion_mat)