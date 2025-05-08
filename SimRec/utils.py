import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from enum import auto, Enum
from itertools import chain

class LinearScheduleWithWarmup:
    def __init__(self, lambd, warmup_steps, lamb_steps):
        self.lambd = 0
        self.warmup_steps = warmup_steps
        self.lamb_steps = lamb_steps
        self.warmup_alpha = lambd / warmup_steps
        self.alpha = lambd / (warmup_steps - lamb_steps)
        self.bias = lambd * (1 - (warmup_steps / (warmup_steps - lamb_steps)))
        self.current_step = -1
        self.step()

    def get_lambd(self):
        return max(self.lambd, 0)

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.lambd = self.warmup_alpha * self.current_step
        else:
            self.lambd = self.alpha * self.current_step + self.bias

class NoneSchedule:
    def __init__(self, lambd):
        self.lambd = lambd

    def get_lambd(self):
        return self.lambd

    def step(self):
        pass

PAD_IDX = 0

def create_similarity_distirbution(similarity_indices, similarity_values, temperature, positive_indices):
    num_items = similarity_indices.shape[0]
    num_positives = positive_indices.shape[0]
    # (num_positives, top_k_similar)
    pos_similarity_indices =  torch.index_select(similarity_indices, index=positive_indices, dim=0)
    pos_similarity_values = torch.index_select(similarity_values, index=positive_indices, dim=0)
    
    # (num_positives, num_items)
    similarities = torch.full((num_positives, num_items), fill_value=-float('inf'), device=similarity_indices.device)
    similarities.scatter_(dim=1, index=pos_similarity_indices, src=pos_similarity_values)

    similarities /= temperature

    distribution = torch.nn.functional.softmax(similarities, dim=-1)
    return distribution

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return user, seq, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, augmentations_fname=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    if augmentations_fname is not None:
        with open(f'data/{augmentations_fname}.txt', 'r') as f:
            for line in f:
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# evaluate on test set
def evaluate_test(model, dataset, args, item_freq):

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    sum_MAP = 0.0   # MAP@10
    # sum_MAE = 0.0   # MAE
    valid_users_map = 0
    valid_user = 0.0
    id_hr = defaultdict(list)   #Os valores sãos listas de 1 ou 0(hit ou não hit) para cada usuario que interagiu com o item
    id_ndcg = defaultdict(list) #Os valores são listas do valor de NDCG@10 para cada usuário que interagiu com o item.
    recommendations = {} #armazenar recomendações {user: [topN items]}
    
    # #definir o numero de candidatos(para melhoria das metricas de avaliação) - ajustar conforme necessario
    # candidate_size = 1000

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)


        item_idx = [test[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        predictions_original = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions_original = predictions_original[0].cpu().numpy() # - for 1st argsort DESC
        
        predictions = -predictions_original # para HR e NDCG
        predictions_tomap = predictions_original # para MAP

        rank = predictions.argsort().argsort()[0].item()
        # Atualiza HR@10 e NDCG@10
        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2) # Fórmula do NDCG
            NDCG += ndcg
            HT += 1            
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        
        #Calculo de MAP@10
    
        ranked_list = np.argsort(predictions_tomap)[::-1]   #ordenando os indics das previsões  de ordem decrescente
        ranked_items = [item_idx[i] for i in ranked_list]
        relevant_items = [test[u][0]]
        ap = compute_MAP(relevant_items,ranked_items,k=10)
        if ap is not None:
            sum_MAP += ap
            valid_users_map += 1
        
        # coletando topN itens recomendados
        topN_items = [item_idx[i] for i in ranked_list[:10]]  # topN = 10
        recommendations[u] = topN_items  # Armazena recomendações
    
    map_score = sum_MAP / valid_users_map if valid_users_map > 0 else 0.0
    # Calcular diversidade e popularidade
    diversity = calculate_coverage_diversity(recommendations, itemnum, len(users), 10)
    popularity = calculate_popularity(recommendations, item_freq, len(users), 10)  # item_freq é o dicionário de frequências
    return (NDCG / valid_user, HT / valid_user , map_score, diversity, popularity ), id_hr, id_ndcg
# evaluate on val set
def evaluate_valid(model, dataset, args, item_freq):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    sum_MAP = 0.0   # MAP@10
    valid_users_map = 0 #contador de usuarios validos para o MAP
    valid_user = 0.0
    id_hr = defaultdict(list)
    id_ndcg = defaultdict(list)
    recommendations = {}
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions_original = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions_original = predictions_original[0].cpu().numpy() # - for 1st argsort DESC
        
        predictions = -predictions_original # para HR e NDCG
        predictions_tomap = predictions_original # para MAP

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2)
            NDCG += ndcg
            HT += 1
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        
        #Calculo de MAP@10
        ranked_list = np.argsort(predictions_tomap)[::-1]   #ordenando os indics das previsões  de ordem decrescente
        ranked_items = [item_idx[i] for i in ranked_list]
        relevant_items = [valid[u][0]]
        ap = compute_MAP(relevant_items,ranked_items,k=10)
        if ap is not None:
            sum_MAP += ap
            valid_users_map += 1

        # coletando topN itens recomendados
        topN_items = [item_idx[i] for i in ranked_list[:10]]  # topN = 10
        recommendations[u] = topN_items  # Armazena recomendações
    
    map_score = sum_MAP / valid_users_map if valid_users_map > 0 else 0.0
    
    # Calcular diversidade e popularidade
    diversity = calculate_coverage_diversity(recommendations, itemnum, len(users), 10)
    popularity = calculate_popularity(recommendations, item_freq, len(users), 10)  # item_freq é o dicionário de frequências
    
    return (NDCG / valid_user, HT / valid_user , map_score, diversity, popularity ), id_hr, id_ndcg

# evaluate on train set
def evaluate_train(model, dataset, args, item_freq):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    sum_MAP = 0.0   # MAP@10
    valid_users_map = 0 #contador de usuarios validos para o MAP
    valid_user = 0.0
    id_hr = defaultdict(list)
    id_ndcg = defaultdict(list)
    recommendations = {}
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u][:-1])
        rated.add(0)
        item_idx = [train[u][-1]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions_original = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions_original = predictions_original[0].cpu().numpy() # - for 1st argsort DESC
        
        predictions = -predictions_original # para HR e NDCG
        predictions_tomap = predictions_original # para MAP

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2)
            NDCG += ndcg
            HT += 1
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        
        #Calculo de MAP@10
        ranked_list = np.argsort(predictions_tomap)[::-1]   #ordenando os indics das previsões  de ordem decrescente
        ranked_items = [item_idx[i] for i in ranked_list]
        relevant_items = [train[u][-1]]
        ap = compute_MAP(relevant_items,ranked_items,k=10)
        if ap is not None:
            sum_MAP += ap
            valid_users_map += 1
        
        # coletando topN itens recomendados
        topN_items = [item_idx[i] for i in ranked_list[:10]]  # topN = 10
        recommendations[u] = topN_items  # Armazena recomendações
    
    map_score = sum_MAP / valid_users_map if valid_users_map > 0 else 0.0
    
    # Calcular diversidade e popularidade
    diversity = calculate_coverage_diversity(recommendations, itemnum, len(users), 10)
    popularity = calculate_popularity(recommendations, item_freq, len(users), 10)  # item_freq é o dicionário de frequências
    
    return (NDCG / valid_user, HT / valid_user , map_score, diversity, popularity ), id_hr, id_ndcg

def compute_MAP(relevant_items,ranked_list,k):
    if not relevant_items:
        return None   #Se um usuario não tem itens relevantes, ele não contribui para o MAP
    
    hits = 0
    sum_precision = 0.0

    for i,item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            hits += 1
            precision_at_i = hits/(i+1)  #precisão acumulada ate o item i
            sum_precision += precision_at_i

    return sum_precision / len(relevant_items)  # Normalizar pelo número de itens relevantes
    # return sum_precision  # AP para 1 item relevante = precisão na posição do item

def calculate_coverage_diversity(recommendations, total_items, num_users, topN):
    unique_items = set()
    for items in recommendations.values():
        unique_items.update(items)
    return len(unique_items) / (num_users * topN)
    #return len(unique_items) / (total_items)  # Versão alternativa: diversidade absoluta

def calculate_popularity(recommendations, item_freq, num_users, topN):
    Ru_total = 0
    for items in recommendations.values():
        Ru_total += sum(item_freq.get(item, 0) for item in items)
    total_interactions = sum(item_freq.values())
    return Ru_total / (num_users * topN) / total_interactions  # Popularidade, versão normalizada