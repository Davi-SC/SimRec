#!/bin/zsh

# --- Configurações Comuns ---

EMBEDDING_MODEL="thenlper_gte-large"
DATASET_PARTIAL_PATH="../data_preprocessing/Beauty/Beauty" # Ou ML-1M/ML-1M
DATASET="${DATASET_PARTIAL_PATH}.txt"
ITEM_FREQ="${DATASET_PARTIAL_PATH}-train_item_freq.txt"
SIMILARITY_INDICES="${DATASET_PARTIAL_PATH}-similarity-indices-${EMBEDDING_MODEL}.pt"
SIMILARITY_VALUES="${DATASET_PARTIAL_PATH}-similarity-values-${EMBEDDING_MODEL}.pt"

SIMILARITY_THREHOLD=0.9
TEMPERATURE=0.5
LAMBDA=0.3
LAMBDA_SCHEDULING="LINEAR"
LAMBDA_WARMPUP=1000
LAMBDA_STEPS=81000
MAX_LEN=50
BATCH_SIZE=128
LR=0.0001
DROPOUT=0.5
NUM_BLOCKS=3
EPOCHS=210
DEVICE="cuda:0"
HIDDEN_DIM=100


declare -A METHODS_TO_TEST
METHODS_TO_TEST[kmeans]="50 750 3000"
METHODS_TO_TEST[fasterpam]="50 500 3000"
METHODS_TO_TEST[agnes]="50 1000 3000"

# --- Loop para Rodar Experimentos ---
for METHOD_NAME in "${!METHODS_TO_TEST[@]}"; do 
    K_VALUES_STR="${METHODS_TO_TEST[$METHOD_NAME]}" 

    for K_VAL in ${K_VALUES_STR}; do 
        NOW_RUN=`date +'%Y%m%d_%H%M%S'` # Data/hora para o nome do diretório de treino
        
        CLUSTERING_METHOD="${METHOD_NAME}_k${K_VAL}" # Ex: kmeans_k2, agnes_k50
        CLUSTERS_CSV_PATH="clusters/Beauty/best_clusters/${CLUSTERING_METHOD}.csv" # Caminho do CSV de clusters

        TRAIN_DIR_CURRENT="results/beauty/${METHOD_NAME}_k${K_VAL}_${NOW_RUN}"

        echo "--- Starting experiment for: Dataset=${DATASET}, Method=${CLUSTERING_METHOD}, K=${K_VAL} ---"
        echo "Train directory: ${TRAIN_DIR_CURRENT}"
        echo "Clusters path: ${CLUSTERS_CSV_PATH}"

        python main.py \
            --dataset "${DATASET}" \
            --item_frequency "${ITEM_FREQ}" \
            --similarity_indices "${SIMILARITY_INDICES}" \
            --similarity_values "${SIMILARITY_VALUES}" \
            --similarity_threshold ${SIMILARITY_THREHOLD} \
            --temperature ${TEMPERATURE} \
            --lambd ${LAMBDA} \
            --lambd_scheduling "${LAMBDA_SCHEDULING}" \
            --lambd_warmup_steps ${LAMBDA_WARMPUP} \
            --lambd_steps ${LAMBDA_STEPS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --maxlen ${MAX_LEN} \
            --dropout_rate ${DROPOUT} \
            --num_blocks ${NUM_BLOCKS} \
            --num_epochs ${EPOCHS} \
            --hidden_units ${HIDDEN_DIM} \
            --train_dir "${TRAIN_DIR_CURRENT}" \
            --device "${DEVICE}" \
            --clustering_method "${CLUSTERING_METHOD}" \
            --clusters_path "${CLUSTERS_CSV_PATH}"

        echo "--- Finished experiment for: Method=${CLUSTERING_METHOD}, K=${K_VAL} ---"
        echo "" s
    done
done

echo "All experiments completed."