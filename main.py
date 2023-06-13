import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler

names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE203", "TYPE"]
df = pd.read_csv(
    "data/glass.data",
    names=names
)

print(df)

metrics = ("accuracy", "f1_weighted", "recall_weighted")

scaler = StandardScaler()
normalizer = Normalizer()
pca1 = PCA(n_components=1)
pca2 = PCA(n_components=2)

scaled_data = pd.DataFrame(scaler.fit_transform(df),columns=names)
normalized_data = pd.DataFrame(normalizer.transform(df),columns=names)
pca1_data = pd.DataFrame(pca1.fit_transform(df),columns=['PCA_1'])
pca2_data = pd.DataFrame(pca2.fit_transform(df),columns=['PCA_1','PCA_2'])


data_sets = {
    "df": df,
    "pca1_data":pca1_data,
    "pca2_data":pca2_data
}

classifiers = {
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier(max_depth=3)": DecisionTreeClassifier(max_depth=3),
    "DecisionTreeClassifier(max_depth=5)": DecisionTreeClassifier(max_depth=5),
    "DecisionTreeClassifier(max_depth=7)": DecisionTreeClassifier(max_depth=7),
}

for classifier_name, classifier in classifiers.items():
    print("\n", classifier_name)
    for metric in metrics:
        for data_name, data in data_sets.items():
            X = data.iloc[:, :-1].to_numpy()
            y = data.iloc[:, -1].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # scores = cross_val_score(classifier, X, y, cv=3, scoring=metric)
            print(data_name, classifier_name, metric, np.mean(scores))


