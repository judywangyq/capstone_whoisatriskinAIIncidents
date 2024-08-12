import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer 
import capstone_data_processing as dp 
import pickle 

# ==============================================================================================================
# Logistic Regression Modeling
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

# Evaluate the model
y_pred = logistic_model.predict(dp.X_test_tfidf)

# Get the actual labels used in the predictions
unique_labels = np.unique(dp.y_test)

# Print classification report
report_lr = classification_report(dp.y_test, y_pred, target_names=[dp.targets_one_hot.columns[i] for i in unique_labels])
print(report_lr)

# ==============================================================================================================

# ROC of Logistic Regression
y_pred_probs_lr = logistic_model.predict_proba(dp.X_test_tfidf)

# Binarize the output labels for multi-class ROC
y_test_binarized = label_binarize(dp.y_test, classes=np.unique(dp.y_test))

# Save the LR results for comparison in DNN script
# with open('lr_results.pkl', 'wb') as file:
#     pickle.dump((y_pred_probs_lr, y_test_binarized), file)

n_classes = y_test_binarized.shape[1]

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs_lr[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference (AUC=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# ==============================================================================================================

# PCA for Visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(dp.X_train_tfidf_resampled)
X_test_2d = pca.transform(dp.X_test_tfidf)

# Fit Logistic Regression model on the reduced 2D data
logistic_model_2d = LogisticRegression(max_iter=1000)
logistic_model_2d.fit(X_train_2d, dp.y_train_resampled)

# Create a mesh grid to plot the decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on the mesh grid
Z = logistic_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plot the training points
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=dp.y_train_resampled, edgecolors='k', cmap=plt.cm.RdYlBu)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary with Logistic Regression')

# Set the limits for x and y axes
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.6)

# Create a legend
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

plt.show()

# ==============================================================================================================
# Feature Importance Plot

# tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# tfidf.fit(dp.X_train) 
# feature_names = tfidf.get_feature_names_out()
feature_names = dp.vectorizer.get_feature_names_out()
num_classes = logistic_model.coef_.shape[0]

# Create a subplot for each class
fig, axes = plt.subplots(num_classes, 1, figsize=(12, num_classes * 6))

if num_classes == 1:
    axes = [axes]  # Ensure axes is iterable for a single class case

# For each class, find the top 3 features that contribute most to that class
for class_index in range(num_classes):
    class_coefficients = np.abs(logistic_model.coef_[class_index])
    top_indices = np.argsort(class_coefficients)[::-1][:3]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = class_coefficients[top_indices]

    axes[class_index].barh(range(len(top_indices)), top_importances, align="center")
    axes[class_index].set_yticks(range(len(top_indices)))
    axes[class_index].set_yticklabels(top_features)
    axes[class_index].invert_yaxis()
    axes[class_index].set_xlabel("Feature Importance")
    axes[class_index].set_title(f"Top 3 Feature Importances (Class: {dp.targets_one_hot.columns[class_index]})")

plt.tight_layout()
plt.show()

# ==============================================================================================================
# Confusion Matrix
cm = confusion_matrix(dp.y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dp.targets_one_hot.columns)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# ==============================================================================================================
# Precision-Recall Curve
y_test_bin = label_binarize(dp.y_test, classes=np.unique(dp.y_test))
y_pred_bin = label_binarize(y_pred, classes=np.unique(dp.y_test))

precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_bin[:, i])

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Class {0} (area = {1:0.2f})'.format(dp.targets_one_hot.columns[i], average_precision[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="best")
plt.grid()
plt.show()
