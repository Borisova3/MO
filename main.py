import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Шаг 1: Чтение данных
print("Шаг 1: Чтение данных")
file_path = "Laptop-Price.csv"
df = pd.read_csv(file_path)
print(f"Размер данных: {df.shape}")

# Шаг 2: Обработка пропущенных данных
print("\nШаг 2: Обработка пропущенных данных")
missing_data = df.isnull().sum()
print("Пропущенные данные:")
print(missing_data)

# Удаляем только ненужный столбец, а не все строки
df = df.drop(columns=["Unnamed: 16"], errors='ignore')
print(f"После удаления ненужного столбца: {df.shape}")

# Шаг 3: Выбор числовых признаков
print("\nШаг 3: Выбор числовых признаков")
numeric_df = df.select_dtypes(include=[np.number])

# Шаг 4: Масштабирование данных и PCA
print("\nШаг 4: Масштабирование данных и PCA")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df.drop(columns=["Price_euros"]))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Размер данных после PCA: {X_pca.shape}")

# Шаг 5: Кластеризация KMeans
print("\nШаг 5: Кластеризация KMeans")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score (KMeans): {silhouette_kmeans:.6f}")

# Шаг 6: Кластеризация DBSCAN
print("\nШаг 6: Кластеризация DBSCAN")
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Clusters found (DBSCAN): {n_clusters_dbscan}")

# Шаг 7: Agglomerative Clustering
print("\nШаг 7: Agglomerative Clustering")
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X_scaled)
silhouette_agglo = silhouette_score(X_scaled, agglo_labels)
print(f"Silhouette Score (Agglomerative): {silhouette_agglo:.6f}")

# Визуализация кластеров
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Cluster")
    plt.show()

plot_clusters(X_pca, kmeans_labels, "KMeans")
plot_clusters(X_pca, dbscan_labels, "DBSCAN")
plot_clusters(X_pca, agglo_labels, "Agglomerative Clustering")
