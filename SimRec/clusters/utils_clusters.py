from sklearn.metrics import silhouette_score
from torchmetrics.clustering import DunnIndex
import pandas as pd
import torch

def load_item_clusters(cluster_file):
    df = pd.read_csv(cluster_file)
    item_clusters = dict(zip(df['item_id'], df['cluster']))
    return item_clusters


def evaluate_silhouette(embeddings, labels):
    return silhouette_score(embeddings, labels)   


def evaluate_dunn(embeddings, labels):
    dunn = DunnIndex()
    labels_tensor = torch.tensor(labels, dtype=torch.int, device=embeddings.device)
    return dunn(embeddings, labels_tensor).item()

# def evaluate_connectivity(embeddings, labels):

