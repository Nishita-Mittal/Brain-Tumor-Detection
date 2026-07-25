import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

print("Loading features...")

X_train = np.load("features/X_train_feats.npy")
y_train = np.load("features/y_train.npy")

print("Loaded X_train shape:", X_train.shape)
print("Loaded y_train shape:", y_train.shape)


print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


print("\nTraining SVM classifier with cross-validation...")
param_grid = {
    "C": [5, 10, 25, 50],
    "gamma": ["scale", 0.01, 0.001]
}
base = SVC(kernel="rbf", probability=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(base, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

clf = grid.best_estimator_
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

print("\nTraining complete!")


output_path = "models/classifier.joblib"

joblib.dump({
    "scaler": scaler,
    "classifier": clf
}, output_path)

print(f"\nSaved classifier to {output_path}")

