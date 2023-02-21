import json
import random
import logging
from collections import defaultdict


def train_generate(dataset, batch_size, few, symbol2id, ent2id, e1relt_e2):
    logging.info('Loading Train Data')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('Loading Candidates')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)

    rel_idx = 0
    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)  # shuffle task order
        task_choice = task_pool[rel_idx % num_tasks]  # choose a rel task
        rel_idx += 1
        candidates = rel2candidates[task_choice]
        if len(candidates) <= 20:
            continue
        task_triples = train_tasks[task_choice]
        random.shuffle(task_triples)
        # select support set, len = few
        support_triples = task_triples[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]
        support_time = [int(triple[3]) for triple in support_triples]

        # select query set, len = batch_size
        other_triples = task_triples[few:]
        if len(other_triples) == 0:
            continue
        if len(other_triples) < batch_size:
            query_triples = [random.choice(other_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(other_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]
        query_time = [int(triple[3]) for triple in query_triples]

        false_pairs = []
        false_left = []
        false_right = []
        false_time = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            t = triple[3]
            while True:
                noise = random.choice(candidates)  # select noise from candidates
                if (noise not in e1relt_e2[e_h + rel + t]) and noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])
            false_time.append(int(t))

        task_choice = [symbol2id[task_choice]]
        yield task_choice, support_time, query_time, false_time, support_left, support_right, query_left, query_right, false_left, false_right
        # return task_choice, support_pairs, query_pairs, false_pairs, support_time, query_time, false_time, support_left, support_right, query_left, query_right, false_left, false_right


def load_symbol2id(datapath):
    symbol_id = {}
    rel2id = json.load(open(datapath + '/relation2ids'))
    ent2id = json.load(open(datapath + '/ent2ids'))
    i = 0
    for key in ent2id.keys():
        if key not in ['', 'OOV']:
            symbol_id[key] = i
            i += 1

    for key in rel2id.keys():
        if key not in ['', 'OOV']:
            symbol_id[key] = i
            i += 1

    symbol_id['PAD'] = i
    symbol2id = symbol_id
    return symbol2id
