
import os
import pandas as pd
import numpy as np
import torch
from torch_kmeans import KMeans
from pyclustering.cluster.agglomerative import agglomerative as agnes
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.agglomerative import agglomerative_visualizer
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import distance_metric,type_metric



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
    
def run_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, device=embeddings.device)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids.cpu().numpy, kmeans

def run_agnes(embeddings, n_clusters):
    data = embeddings.cpu().tolist()
    initial_centers = kmeans_plusplus_initializer(data, n_clusters).initialize()
    model = agnes(data, initial_centers)
    model.process()
    clusters = model.get_clusters()

    cluster_ids = np.zeros(len(data), dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            cluster_ids[idx] = cluster_id
    
    return cluster_ids, model

def run_clarans(embeddings, n_clusters, numlocal=2, maxneighbor=5):
    data = embeddings.cpu().tolist()
    
    model = clarans(data, n_clusters, numlocal=numlocal, maxneighbor=maxneighbor)
    model.process()
    
    clusters = model.get_clusters()
    
    cluster_ids = np.zeros(len(data), dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            cluster_ids[idx] = cluster_id

    return cluster_ids, model

def run_pam(embeddings, n_clusters):
    data = embeddings.cpu().tolist()
    
    initial_medoids = kmeans_plusplus_initializer(data, n_clusters).initialize()
    
    metric = distance_metric(type_metric.EUCLIDEAN)
    
    model = kmedoids(data, initial_medoids, data_type='points', metric=metric)
    model.process()
    
    clusters = model.get_clusters()

    cluster_ids = np.zeros(len(data), dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            cluster_ids[idx] = cluster_id

    return cluster_ids, model
