#!/bin/zsh
NOW=`date +'%I_%M_%d_%m'`

EMBEDDING_MODEL=thenlper_gte-large
DATASET_PARTIAL_PATH="../data_preprocessing/ML-1M/ml-1m"
DATASET="${DATASET_PARTIAL_PATH}.txt"
ITEM_FREQ="${DATASET_PARTIAL_PATH}-train_item_freq.txt"
SIMILARITY_INDICES="${DATASET_PARTIAL_PATH}-similarity-indices-${EMBEDDING_MODEL}.pt"
SIMILARITY_VALUES="${DATASET_PARTIAL_PATH}-similarity-values-${EMBEDDING_MODEL}.pt"

#clusters
CLUSTERING_METHOD="kmeans100"  
CLUSTERS_PATH="clusters/ML-1M/best_${CLUSTERING_METHOD}_umap.csv"

SIMILARITY_THREHOLD=0.5
TEMPERATURE=1
LAMBDA=0.6
LAMBDA_SCHEDULING=LINEAR
LAMBDA_WARMPUP=1000
LAMBDA_STEPS=7000
MAX_LEN=200
#BATCH_SIZE=128
BATCH_SIZE=64
LR=0.001
DROPOUT=0.5
NUM_BLOCKS=3
EPOCHS=200
DEVICE="cuda:0"
HIDDEN_DIM=100
TRAIN_DIR="results/ML-1M/${NOW}"

python main.py --dataset ${DATASET}\
               --item_frequency ${ITEM_FREQ}\
               --similarity_indices ${SIMILARITY_INDICES}\
               --similarity_values ${SIMILARITY_VALUES}\
               --similarity_threshold ${SIMILARITY_THREHOLD}\
               --temperature ${TEMPERATURE}\
               --lambd ${LAMBDA}\
               --lambd_scheduling "${LAMBDA_SCHEDULING}"\
               --lambd_warmup_steps ${LAMBDA_WARMPUP}\
               --lambd_steps ${LAMBDA_STEPS}\
               --batch_size ${BATCH_SIZE}\
               --lr ${LR}\
               --maxlen ${MAX_LEN}\
               --dropout_rate ${DROPOUT}\
               --num_blocks ${NUM_BLOCKS}\
               --num_epochs ${EPOCHS}\
               --hidden_units ${HIDDEN_DIM}\
               --train_dir ${TRAIN_DIR}\
               --device ${DEVICE}\
               --clustering_method ${CLUSTERING_METHOD}\
               --clusters_path ${CLUSTERS_PATH}