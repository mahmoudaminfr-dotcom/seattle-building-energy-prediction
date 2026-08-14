import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score

# 1. Chargement des données
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Entraînement du KMeans
model = KMeans(n_clusters=3, n_init="auto", random_state=42)
model.fit(X)

# --- 3. RÉALIGNEMENT CORRECT DES LABELS ---
labels = np.zeros_like(model.labels_)
labels[model.labels_ == 0] = 2
labels[model.labels_ == 1] = 0
labels[model.labels_ == 2] = 1

# --- 4. IMPRESSION DES RÉSULTATS MÉTRIQUES ---
print("score (Inertie négative) :", model.score(X))
print("silhouette_score         :", silhouette_score(X, model.labels_))
print("accuracy_score corrigé  :", accuracy_score(y, labels))
print("confusion_matrix :\n", confusion_matrix(y, labels))

# --- 5. VISUALISATION GRAPHIQUE (MATPLOTLIB) ---
plt.figure(figsize=(10, 6))

# Affichage des points (Longueur vs Largeur du Pétale)
scatter = plt.scatter(
    X[:, 2], X[:, 3], c=model.labels_, cmap="viridis", s=60, edgecolors="k"
)

# Affichage des 3 centres de clusters (sur les dimensions Pétale)
plt.scatter(
    model.cluster_centers_[:, 2],
    model.cluster_centers_[:, 3],
    c="red",
    marker="X",
    s=200,
    linewidths=2,
    label="Centres de clusters",
)

plt.title("Clustering KMeans - Dataset Iris (Pétales)", fontsize=14)
plt.xlabel("Longueur du pétale (cm)", fontsize=12)
plt.ylabel("Largeur du pétale (cm)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Affiche la fenêtre pop-up graphique
plt.show()