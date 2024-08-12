import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import capstone_data_processing as dp  # Import your data processing module
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, class_model_map):
        self.models = models
        self.class_model_map = class_model_map

    def fit(self, X, y):
        for model_name, model in self.models.items():
            if model_name == 'DNN':
                y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(self.class_model_map))
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                model.fit(X, y_one_hot, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=2)
            else:
                skf = StratifiedKFold(n_splits=5)
                for train_idx, val_idx in skf.split(X, y):
                    model.fit(X[train_idx], y[train_idx])
        return self

    def predict(self, X):
        final_preds = np.zeros(X.shape[0], dtype=int)
        for class_index, model_name in self.class_model_map.items():
            model = self.models[model_name]
            if model_name == 'DNN':
                class_probs = model.predict(X)
                class_preds = np.argmax(class_probs, axis=1)
            else:
                class_preds = model.predict(X)
            final_preds[class_preds == class_index] = class_index
        return final_preds

# ==============================================================================================================
# Instantiate and Compile the Models

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=8)

# Deep Neural Network
dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(dp.X_train_tfidf_resampled.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(dp.targets_one_hot.columns), activation='softmax')
])

# Compile the DNN model
dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dictionary of models
models = {
    'LR': log_reg,
    'KNN': knn,
    'DNN': dnn_model
}

# Fit the non-DNN models on the resampled training data
log_reg.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)
knn.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

# ==============================================================================================================
# Define Mapping and Train the Custom Classifier

# Mapping of class index to model name
class_model_map = {
    0: 'LR',   # Business
    1: 'LR',  # Consumers
    2: 'LR',  # General Public
    3: 'KNN',  # Government
    4: 'LR',   # Minorities
    5: 'LR'    # Workers
}

# Instantiate and train the custom classifier
custom_clf = CustomClassifier(models=models, class_model_map=class_model_map)
custom_clf.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

# ==============================================================================================================
# Evaluate the Custom Classifier

# Make predictions on the test data
y_pred = custom_clf.predict(dp.X_test_tfidf)

# Identify the unique classes in the test set
unique_labels_test = np.unique(dp.y_test)
unique_labels_train = np.unique(dp.y_train)
unique_labels_pred = np.unique(y_pred)

# Ensure target names cover all possible classes
all_classes = np.unique(np.concatenate([unique_labels_train, unique_labels_test, unique_labels_pred]))
target_names = [dp.targets_one_hot.columns[i] for i in all_classes]

# Print unique labels and target names for debugging
print("Unique labels in the test set:", unique_labels_test)
print("Unique labels in the train set:", unique_labels_train)
print("Unique labels in the predictions:", unique_labels_pred)
print("Target names corresponding to unique labels:", target_names)

# Print classification report with corrected labels
print(classification_report(dp.y_test, y_pred, labels=all_classes, target_names=target_names))
