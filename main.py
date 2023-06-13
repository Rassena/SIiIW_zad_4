import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler
df = pd.read_csv(
    "data/glass.data",
    names=["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE203", "TYPE"]
)

print(df)

metrics = ("accuracy", "f1_weighted", "recall_weighted")

scaler = StandardScaler()
normalizer = Normalizer()

scaled_data = scaler.fit_transform(df)
normalized_data = normalizer.transform(df)

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()




classifiers = {
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier(max_depth=3)": DecisionTreeClassifier(max_depth=3),
    "DecisionTreeClassifier(max_depth=5)": DecisionTreeClassifier(max_depth=5),
    "DecisionTreeClassifier(max_depth=7)": DecisionTreeClassifier(max_depth=7),
}

for classifier_name, classifier in classifiers.items():
    for metric in metrics:
        scores = cross_val_score(classifier, X, y, cv=3, scoring=metric)
        print(classifier_name, metric, np.mean(scores))


