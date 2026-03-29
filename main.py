import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

data = pd.read_csv("focus_dataset.csv")
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
rf = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)

print("Logistic Regression accuracy:", accuracy_score(y_test, pred_lr))
print("Random Forest accuracy:", accuracy_score(y_test, pred_rf))
print("\nRandom Forest confusion matrix:")
print(confusion_matrix(y_test, pred_rf))
print("\nRandom Forest report:")
print(classification_report(y_test, pred_rf, zero_division=0))

print("\n5-fold CV accuracy (LR):", cross_val_score(lr, X, y, cv=5, scoring="accuracy").mean())
print("5-fold CV accuracy (RF):", cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean())

pca_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=3)),
    ("model", LogisticRegression(max_iter=1000))
])
print("5-fold CV accuracy (LR + PCA):", cross_val_score(pca_lr, X, y, cv=5, scoring="accuracy").mean())
