import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import capstone_data_processing as dp  # Import your data processing module

# ==============================================================================================================
# XGBoost Model Training

# Instantiate the XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(dp.targets_one_hot.columns), eval_metric='mlogloss')

# Train the XGBoost model
xgb_model.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

# ==============================================================================================================
# Evaluate the XGBoost Model

# Make predictions on the test data
y_pred = xgb_model.predict(dp.X_test_tfidf)

# Get the actual labels used in the predictions
unique_labels = np.unique(dp.y_test)

# Print classification report
report_xgb = classification_report(dp.y_test, y_pred, target_names=[dp.targets_one_hot.columns[i] for i in unique_labels])
print(report_xgb)

# ==============================================================================================================
# XGBoost ROC Curve

# Predict probabilities for the test set
y_pred_probs_xgb = xgb_model.predict_proba(dp.X_test_tfidf)

# Binarize the output labels for multi-class ROC
y_test_binarized = label_binarize(dp.y_test, classes=np.unique(dp.y_test))
n_classes = y_test_binarized.shape[1]

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs_xgb[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - XGBoost')
plt.legend(loc="lower right")
plt.show()

# ==============================================================================================================
# Confusion Matrix
cm = ConfusionMatrixDisplay.from_predictions(dp.y_test, y_pred, display_labels=dp.targets_one_hot.columns, cmap='Blues')
cm.plot()
plt.title('Confusion Matrix - XGBoost')
plt.show()

# ==============================================================================================================
# XGBoost Precision-Recall Curve

# Compute Precision-Recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_probs_xgb[:, i])
    average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_probs_xgb[:, i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Class {i} (area = {average_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve - XGBoost')
plt.legend(loc="best")
plt.grid()
plt.show()
