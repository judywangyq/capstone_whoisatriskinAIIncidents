import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import capstone_data_processing as dp  # Import your data processing module

# ==============================================================================================================
# K-Nearest Neighbors (KNN) Modeling

# Instantiate the KNN model
knn_model = KNeighborsClassifier(n_neighbors=8)

# Train the model using the resampled training data
knn_model.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

# Evaluate the model
y_pred = knn_model.predict(dp.X_test_tfidf)

# Get the actual labels used in the predictions
unique_labels = np.unique(dp.y_test)

# Print classification report
report_knn = classification_report(dp.y_test, y_pred, target_names=[dp.targets_one_hot.columns[i] for i in unique_labels])
print(report_knn)

# ==============================================================================================================

# KNN Confusion Matrix

# Plot confusion matrix
cm = confusion_matrix(dp.y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dp.targets_one_hot.columns)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.show()


'''
# Adjusted KNN

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Instantiate the KNN model
knn = KNeighborsClassifier()

# Perform grid search
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf_resampled, y_train_resampled)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Use the best estimator
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_tfidf)

# Print the classification report
unique_labels = np.unique(y_test)
print(classification_report(y_test, y_pred, target_names=[targets_one_hot.columns[i] for i in unique_labels]))
'''

'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Build and train a Random Forest model
rf_model = RandomForestClassifier(
    min_samples_split=2, #added later
    max_features='sqrt', #added later
    random_state=42,
    n_estimators=100
    )  # n_estimators is the number of trees in the forest
rf_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test_tfidf)

# Get the actual labels used in the predictions
unique_labels_rf = np.unique(y_test)

# Print classification report
report_rf = classification_report(y_test, y_pred_rf, target_names=[targets_one_hot.columns[i] for i in unique_labels_rf])
print(report_rf)
'''