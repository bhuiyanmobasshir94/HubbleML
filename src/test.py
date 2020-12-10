from src.hubbleml import HubbleMLClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from pprint import pprint

d = load_breast_cancer()
y = d["target"]
X = pd.DataFrame(d["data"], columns=d["feature_names"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = HubbleMLClassifier()
model.fit(X_train, y_train)

pprint(balanced_accuracy_score(y_test, model.predict(X_test)))

pprint(model.best_pipeline, indent=4)
