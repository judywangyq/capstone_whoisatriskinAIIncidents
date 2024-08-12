import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import capstone_data_processing as dp  # Import your data processing module
import pickle  # Add this import to load LR results

# import os
# if os.path.exists('lr_results.pkl'):
#     print("File exists.")
# else:
#     print("File does not exist.")


# Load the LR results to compare the ROC
# with open('lr_results.pkl', 'rb') as file:
#     y_pred_probs_lr, y_test_binarized = pickle.load(file)


# ==============================================================================================================
# Build and Train the DNN Model

# Define the DNN architecture
model = Sequential([
    Dense(512, activation='relu', input_shape=(dp.X_train_tfidf_resampled.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(dp.targets_one_hot.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled, epochs=20, validation_data=(dp.X_test_tfidf, dp.y_test), verbose=2, callbacks=[early_stopping])

# ==============================================================================================================
# Evaluate the DNN Model

# Predict probabilities for the test set
y_pred_probs = model.predict(dp.X_test_tfidf)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get the actual labels used in the predictions
unique_labels = np.unique(dp.y_test)

# Print classification report
report_dnn = classification_report(dp.y_test, y_pred, target_names=[dp.targets_one_hot.columns[i] for i in unique_labels])
print(report_dnn)

# ==============================================================================================================
# DNN ROC Curve

# Binarize the output labels for multiclass ROC
y_test_binarized = label_binarize(dp.y_test, classes=np.unique(dp.y_test))
n_classes = y_test_binarized.shape[1]

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - DNN')
plt.legend(loc="lower right")
plt.show()

# ==============================================================================================================
# Average ROC Curve Comparison (DNN vs. LR)

def average_roc_curve(y_test_binarized, y_pred_probs):
    """Compute average ROC curve."""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(y_test_binarized.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute the average FPR and TPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_test_binarized.shape[1])]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(y_test_binarized.shape[1]):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= y_test_binarized.shape[1]

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

# Calculate average ROC for DNN
avg_fpr_dnn, avg_tpr_dnn, avg_auc_dnn = average_roc_curve(y_test_binarized, y_pred_probs)

# Assuming you have y_pred_probs_lr from Logistic Regression predictions (if you want to compare)
# avg_fpr_lr, avg_tpr_lr, avg_auc_lr = average_roc_curve(y_test_binarized, y_pred_probs_lr)

# Plotting the average ROC curves
plt.figure()
# Uncomment the following line if you want to compare with Logistic Regression
# plt.plot(avg_fpr_lr, avg_tpr_lr, color='blue', label=f'Logistic Regression (AUC = {avg_auc_lr:.2f})')
plt.plot(avg_fpr_dnn, avg_tpr_dnn, color='red', label=f'Deep Neural Network (AUC = {avg_auc_dnn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference (AUC=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()
