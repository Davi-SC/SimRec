import os
import pandas as pd
import numpy as np
import torch
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import pairwise_distances
from kmedoids import fasterpam, fastermsc, dynmsc


def load_embeddings(embeddings_path):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Arquivo de embeddings não encontrado em {embeddings_path}")
    embeddings_tensor = torch.load(embeddings_path)

    if not torch.is_tensor(embeddings_tensor):
        embeddings_tensor = torch.tensor(embeddings_tensor)
    return embeddings_tensor
    
def save_clusters(cluster_ids, output_path):
    df = pd.DataFrame({'item_id': np.arange(len(cluster_ids)), 'cluster': cluster_ids})
    df.to_csv(output_path, index=False)
    print(f"Clusters salvos em {output_path}")
    
def run_spectral(embeddings, n_clusters, n_neighbors):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', n_neighbors=n_neighbors)
    cluster_ids = spectral.fit_predict(embeddings)
    return cluster_ids, spectral

def run_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids, kmeans

def run_agnes(embeddings, n_clusters, linkage='ward'):
    agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_ids = agnes.fit_predict(embeddings)
    return cluster_ids, agnes

def run_fasterpam(embeddings, n_clusters):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    diss_matrix = pairwise_distances(embeddings, metric='cosine')
    fasterPam = fasterpam(diss_matrix, n_clusters)

    return fasterPam.labels, fasterPam.medoids

def run_fastermsc(embeddings, n_clusters):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    diss_matrix = pairwise_distances(embeddings, metric='cosine')
    fasterMsc = fastermsc(diss_matrix, n_clusters)

    return fasterMsc.labels, fasterMsc.medoids

def run_dynmsc(embeddings, n_clusters):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    diss_matrix = pairwise_distances(embeddings, metric='cosine')
    # dynMsc = dynmsc(diss_matrix, n_clusters)
    dynMsc = dynmsc(diss_matrix, medoids=n_clusters ,minimum_k=n_clusters)
    
    return dynMsc.labels, dynMsc.medoids

def run_dbscan(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_ids = dbscan.fit_predict(embeddings)
    return cluster_ids, dbscan

def run_hdbscan(embeddings, min_cluster_size=5, min_samples=None):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_ids = hdbscan_model.fit_predict(embeddings)
    return cluster_ids, hdbscan_model

# # #################### Metricas trocadas
# def run_fasterpam(embeddings, n_clusters):
#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#     diss_matrix = pairwise_distances(embeddings, metric='euclidean')
#     fasterPam = fasterpam(diss_matrix, n_clusters)

#     return fasterPam.labels, fasterPam.medoids

# def run_fastermsc(embeddings, n_clusters):
#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#     diss_matrix = pairwise_distances(embeddings, metric='euclidean')
#     fasterMsc = fastermsc(diss_matrix, n_clusters)

#     return fasterMsc.labels, fasterMsc.medoids

# def run_dynmsc(embeddings, n_clusters):
#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#     diss_matrix = pairwise_distances(embeddings, metric='euclidean')
#     dynMsc = dynmsc(diss_matrix, n_clusters)
    
#     return dynMsc.labels, dynMsc.medoids

# def run_dbscan(embeddings, eps=0.5, min_samples=5):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
#     cluster_ids = dbscan.fit_predict(embeddings)
#     return cluster_ids, dbscan

# def run_hdbscan(embeddings, min_cluster_size=5, min_samples=None):
#     hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='cosine')
#     cluster_ids = hdbscan_model.fit_predict(embeddings)
#     return cluster_ids, hdbscan_model
# #########################################

