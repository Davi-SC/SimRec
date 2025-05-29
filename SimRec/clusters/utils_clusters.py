from sklearn.metrics import silhouette_score
from torchmetrics.clustering import DunnIndex
import pandas as pd
import torch
from sklearn.decomposition import PCA
import umap
import os
from clusters.clusters import *


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

# redução da dimensionalidade dos embeddings
def dimensionality_reduction(embeddings, method, n_components, random_state = 42):
    if method == 'pca':
        print(f'Aplicando PCA com n_components = {n_components}...')
        reducer = PCA(n_components = n_components)
    elif method == 'umap':
        print(f'Aplicando UMAP com n_components = {n_components}...')
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError('Metodo invalido !!!')
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Dimensionalidade reduzida de {embeddings.shape[1]} para {reduced_embeddings.shape[1]}")
    return reduced_embeddings

# executa os metodos de clusterização que recebem apenas k como parametro : run_kmeans, run_fasterpam, run_fastermsc, run_dynmsc, run_agnes
def run_methods_with_k(embeddings, methods, k_values, output_path, dataset_name):
    os.makedirs(output_path, exist_ok=True)
    
    for method in methods:
        print(f"Executando {method.__name__}")
        results = []
        for k in k_values:
            print(f" - k = {k}")
            try:
                cluster_ids, model = method(embeddings, k)

                method_path = os.path.join(output_path, method.__name__)
                os.makedirs(method_path, exist_ok=True)

                save_path = os.path.join(method_path, f"{dataset_name}_cluster_{method.__name__}_k{k}.csv")
                save_clusters(cluster_ids, save_path)

                silhouette = evaluate_silhouette(embeddings, cluster_ids)
                dunn = evaluate_dunn(embeddings, cluster_ids)

                results.append({
                    "Method": method.__name__,
                    "k": k,
                    "Silhouette": silhouette,
                    "Dunn": dunn
                })
            except Exception as e:
                print(f"Erro ao executar {method.__name__} com k = {k}: {e}")
        
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_path, f"{dataset_name}_{method.__name__}_results.csv")
        results_df.to_csv(results_csv_path, index=False)    
    print("Clusterização concluída e resultados salvos")

def run_dbscan_combinations(embeddings, eps_values, min_samples_values, output_path, dataset_name):
    os.makedirs(output_path, exist_ok=True)
    method_name = "run_dbscan"
    print(f"Executando {method_name}...")
    result_rows =[]
    for eps in eps_values:
        for min_s in min_samples_values:
            print(f"DBSCAN - eps = {eps}, min_samples = {min_s}")
            try:
                cluster_ids, model = run_dbscan(embeddings, eps, min_s)
                
                unique_clusters = set(cluster_ids)
                if -1 in unique_clusters:
                    num_clusters_actual = len(unique_clusters) - 1
                else:
                    num_clusters_actual = len(unique_clusters)
                if num_clusters_actual <= 1:
                    print(f"DBSCAN (eps={eps}, min_samples={min_s}) gerou 1 cluster ou só ruído, ignorado.")
                    continue

                method_path = os.path.join(output_path, method_name)
                os.makedirs(method_path, exist_ok=True)

                save_path = f"{method_path}/{dataset_name}_clusters_{method_name}_eps{eps}_min{min_s}.csv"
                save_clusters(cluster_ids, save_path)

                silhouette = evaluate_silhouette(embeddings, cluster_ids)
                dunn = evaluate_dunn(embeddings, cluster_ids)

                result_rows.append({
                    "Method": method_name,
                    "eps": eps,
                    "min_samples": min_s,
                    'N_Clusters_Actual': num_clusters_actual,
                    "Silhouette": silhouette,
                    "Dunn": dunn
                })

            except Exception as e:
                print(f"Erro no DBSCAN (eps={eps}, min_samples={min_s}): {e}")
    
    results_df = pd.DataFrame(result_rows)
    results_csv_path = os.path.join(output_path, f"{dataset_name}_{method_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)    
    print("Clusterização concluída e resultados salvos")

def run_hdbscan_combinations(embeddings, min_cluster_sizes,min_samples, output_path, dataset_name):
    os.makedirs(output_path, exist_ok=True)
    method_name = "run_hdbscan"
    print(f"Executando {method_name}...")
    result_rows =[]
    for size in min_cluster_sizes:
        for ms in min_samples:
            print(f"HDBSCAN - min_cluster_size = {size}, min_samples = {ms}")
            try:
                cluster_ids, model = run_hdbscan(embeddings,size,ms)
                
                unique_clusters = set(cluster_ids)
                if -1 in unique_clusters:
                    num_clusters_actual = len(unique_clusters) - 1
                else:
                    num_clusters_actual = len(unique_clusters)

                if num_clusters_actual <= 1:
                    print(f"HDBSCAN (min_cluster_size={size}) gerou apenas um cluster ou ruído. Ignorado.")
                    continue

                method_path = os.path.join(output_path, method_name)
                os.makedirs(method_path, exist_ok=True)

                save_path = f"{method_path}/{dataset_name}_clusters_{method_name}_minsize{size}_minsample{ms}.csv"
                save_clusters(cluster_ids, save_path)

                silhouette = evaluate_silhouette(embeddings, cluster_ids)
                dunn = evaluate_dunn(embeddings, cluster_ids)

                result_rows.append({
                    "Method": method_name,
                    "min_samples": ms,
                    "min_cluster_size": size,
                    'N_Clusters_Actual': num_clusters_actual,
                    "Silhouette": silhouette,
                    "Dunn": dunn
                })

            except Exception as e:
                print(f"Erro no HDBSCAN (min_cluster_size={size} e min_samples ={ms}): {e}")
    
    results_df = pd.DataFrame(result_rows)
    results_csv_path = os.path.join(output_path, f"{dataset_name}_{method_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)    
    print("Clusterização concluída e resultados salvos")

def run_spectral_combinations(embeddings, k_values, neighbor_values, output_path, dataset_name):
    os.makedirs(output_path, exist_ok=True)
    method_name = 'run_spectral'
    print(f"Executando {method_name}...")
    result_rows =[]
    for k in k_values:
        for n in neighbor_values:
            print(f"Executando SPECTRAL (k_values = {k}, neighbor_values = {n})...")
            try:
                cluster_ids, model = run_spectral(embeddings, k, n)
                if len(set(cluster_ids)) <= 1:
                    print(f"SPECTRAL (k_values = {k}, neighbor_values = {n}) gerou apenas um cluster ou ruído. Ignorado.")
                    continue

                method_path = os.path.join(output_path, method_name)
                os.makedirs(method_path, exist_ok=True)

                save_path = f"{method_path}/{dataset_name}_clusters_{method_name}_k{k}_n{n}.csv"
                save_clusters(cluster_ids, save_path)

                silhouette = evaluate_silhouette(embeddings, cluster_ids)
                dunn = evaluate_dunn(embeddings, cluster_ids)

                result_rows.append({
                    "Method": method_name,
                    "k": k,
                    "n_neighbors": n,
                    "Silhouette": silhouette,
                    "Dunn": dunn
                })


            except Exception as e:
                print(f"Erro no Spectral (k={k}, n={n}): {e}")

    results_df = pd.DataFrame(result_rows)
    results_csv_path = os.path.join(output_path, f"{dataset_name}_{method_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)    
    print("Clusterização concluída e resultados salvos")

