import pandas as pd
import numpy as np
import string
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
def load_dataset(filepath):
    return pd.read_excel(filepath)

# Drop unnecessary columns
def drop_columns(dataframe, columns):
    return dataframe.drop(columns, axis=1, errors='ignore')

# Combine text columns into a single feature
def combine_text(row):
    text = f"{row['Title']} {row['Summary']} {row['AI Principles']} {row['Industries']} {row['Harm Types']}"
    text = text.replace(',', ' ').replace('/', ' ')
    return text

def create_combined_text_column(dataframe):
    dataframe['Text'] = dataframe.apply(combine_text, axis=1)
    return dataframe

# Filter out rows with specific criteria
def filter_rows(x):
    return not ('Unknown' in x or x.strip() == '' or x.strip().lower() == 'n/a')

def apply_filter(dataframe):
    dataframe = dataframe[dataframe['Affected Stakeholders'].apply(filter_rows)]
    return dataframe

# Convert Affected Stakeholders to list
def convert_to_list(dataframe):
    dataframe['Affected Stakeholders'] = dataframe['Affected Stakeholders'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    return dataframe

# Combine certain stakeholder categories
def combine_minority(row):
    row = [stakeholder if stakeholder != 'Women' else 'Minorities' for stakeholder in row]
    row = [stakeholder if stakeholder != 'Children' else 'General public' for stakeholder in row]
    return row

def apply_combine_minority(dataframe):
    dataframe['Affected Stakeholders'] = dataframe['Affected Stakeholders'].apply(combine_minority)
    return dataframe

# Duplicate rows based on 'Affected Stakeholders'
def duplicate_rows(dataframe):
    return dataframe.explode('Affected Stakeholders').reset_index(drop=True)

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply text cleaning
def apply_text_cleaning(dataframe):
    dataframe['Cleaned_Text'] = dataframe['Text'].apply(clean_text)
    return dataframe

# Remove stop words from features
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    meaningful_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(meaningful_words)

def apply_remove_stop_words(features):
    return features.apply(remove_stop_words)

# Main function to process data
def process_data(filepath):
    df = load_dataset(filepath)
    df = drop_columns(df, ['Severity'])
    df = create_combined_text_column(df)
    df['Affected Stakeholders'] = df['Affected Stakeholders'].fillna('')
    df = apply_filter(df)
    df = convert_to_list(df)
    df = apply_combine_minority(df)
    df = duplicate_rows(df)
    df = apply_text_cleaning(df)
    features = apply_remove_stop_words(df['Cleaned_Text'])
    targets = df['Affected Stakeholders']
    return features, targets

# One-hot encode the target labels
def one_hot_encode_targets(targets):
    return pd.get_dummies(targets)

# Prepare training and testing sets
def prepare_train_test_data(features, targets_one_hot):
    y_labels = np.argmax(targets_one_hot.values, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    return X_train, X_test, y_train, y_test

# Vectorize the text using TF-IDF
def vectorize_text(X_train, X_test, max_features=5000, ngram_range=(1, 2)):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    return X_train_tfidf, X_test_tfidf, tfidf

# Apply SMOTE to balance the classes
def apply_smote(X_train_tfidf, y_train):
    smote = SMOTE(random_state=42, k_neighbors=1)
    # return smote.fit_resample(X_train_tfidf, y_train)

    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

    # Check the distribution after resampling
    print("Class distribution after SMOTE:")
    print(Counter(y_resampled))

    return X_resampled, y_resampled

# Compute class weights
def compute_weights(y_train_resampled):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
    return {i: class_weights[i] for i in range(len(class_weights))}

# Main execution
# filepath = 'oecd_incidents_summary_total.xlsx'
# features, targets = process_data(filepath)
# targets_one_hot = one_hot_encode_targets(targets)

# # Check class distribution before resampling
# print("Class distribution before resampling:")
# print(targets.value_counts())

# X_train, X_test, y_train, y_test = prepare_train_test_data(features, targets_one_hot)
# X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)
# X_train_tfidf_resampled, y_train_resampled = apply_smote(X_train_tfidf, y_train)
# class_weights_dict = compute_weights(y_train_resampled)

# if __name__ == "__main__":
#     filepath = 'oecd_incidents_summary_total.xlsx'
#     features, targets = process_data(filepath)
#     targets_one_hot = one_hot_encode_targets(targets)

#     X_train, X_test, y_train, y_test = prepare_train_test_data(features, targets_one_hot)
#     X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
#     X_train_tfidf_resampled, y_train_resampled = apply_smote(X_train_tfidf, y_train)

#     # class_weights_dict = compute_weights(y_train_resampled)

#     # Check class distribution before resampling
#     print("Class distribution before resampling:")
#     print(targets.value_counts())

if __name__ == "__main__":
    filepath = 'oecd_incidents_summary_total.xlsx'
    features, targets = process_data(filepath)
    targets_one_hot = one_hot_encode_targets(targets)

    X_train, X_test, y_train, y_test = prepare_train_test_data(features, targets_one_hot)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    X_train_tfidf_resampled, y_train_resampled = apply_smote(X_train_tfidf, y_train)

    # Print or plot if needed
    print("Class distribution before resampling:")
    print(targets.value_counts())
else:
    filepath = 'oecd_incidents_summary_total.xlsx'
    features, targets = process_data(filepath)
    targets_one_hot = one_hot_encode_targets(targets)

    X_train, X_test, y_train, y_test = prepare_train_test_data(features, targets_one_hot)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    X_train_tfidf_resampled, y_train_resampled = apply_smote(X_train_tfidf, y_train)
    
    # Assign these variables to the module level
    globals().update({
        "X_train_tfidf_resampled": X_train_tfidf_resampled,
        "y_train_resampled": y_train_resampled,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "targets_one_hot": targets_one_hot,
        "vectorizer": vectorizer,
    })
