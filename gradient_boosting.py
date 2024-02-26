import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load your dataset
df = pd.read_csv('loan_default_prediction_150k.csv')

# Assuming 'Default' is the target variable
X = df.drop('Default', axis=1)
y = df['Default']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Optional: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define grid of hyperparameters
param_grid = {
    'learning_rate': [0.01],
    'max_depth': [5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [2],
    'subsample': [1.0]
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Use the best estimator for predictions
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
