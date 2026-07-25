import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def main(clf_path='models/classifier.joblib'):
    data = joblib.load(clf_path)
    scaler = data['scaler']
    clf = data['classifier']

    X_test = np.load('features/X_test_feats.npy')
    y_test = np.load('features/y_test.npy')

    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"Test features shape: {X_test.shape}")
    print(f"Test labels shape:   {y_test.shape}")

    X_test_scaled = scaler.transform(X_test)
    preds = clf.predict(X_test_scaled)

    print(f'\nTest samples: {len(y_test)}')
    print(f'True distribution:      {dict(Counter(y_test.astype(int)))}')
    print(f'Predicted distribution: {dict(Counter(preds.astype(int)))}')
    print(f'\nAccuracy: {accuracy_score(y_test, preds):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, preds, target_names=CLASSES))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))


if __name__ == '__main__':
    main()

