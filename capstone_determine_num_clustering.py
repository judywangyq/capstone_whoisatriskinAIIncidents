import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import capstone_data_processing as dp 

# Elbow Method to find the optimal number of clusters
def plot_elbow_method(X, max_clusters=10):
    """Plot the Elbow Method to determine the optimal number of clusters."""
    wcss = []  # within-cluster sum of squares
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

# Silhouette Score to find the optimal number of clusters
def plot_silhouette_score(X, max_clusters=10):
    """Plot the Silhouette Score to determine the optimal number of clusters."""
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Load and process data using the filepath from the data processing module
# features, _ = dp.process_data(dp.filepath)

# Vectorize the text using TF-IDF
# tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# X_tfidf = tfidf.fit_transform(features).toarray()

# Plot Elbow Method
# plot_elbow_method(X_tfidf, max_clusters=10)

# Plot Silhouette Score
# plot_silhouette_score(X_tfidf, max_clusters=10)

if __name__ == "__main__":
    plot_elbow_method(dp.X_train_tfidf_resampled, max_clusters=10)
    plot_silhouette_score(dp.X_train_tfidf_resampled, max_clusters=10)