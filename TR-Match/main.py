from collections import defaultdict
from torch import optim
from collections import deque
from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
import logging
import json
import random
import psutil


class Trainer(object):
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        self.load_symbol2id()
        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        print(self.pad_id)

        self.Matcher = Matcher(few=self.few, max_nb=self.max_neighbor, embed_dim=self.embed_dim,
                               num_symbols=self.num_symbols, num_attention_heads=self.num_attention_heads,
                               dropout=self.dropout, process_step=self.process_step, device=self.device
                               )


        self.Matcher.to(self.device)
        self.batch_nums = 0

        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.Matcher.parameters())

        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1relt_e2 = defaultdict(list)
        self.e1relt_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))
        self.meta = True

    def load_symbol2id(self):
        # gen symbol2id, without embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        # rel and ent combine together
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 3)) * self.pad_id).astype(int)
        self.e1_rele2t = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2, t = line.rstrip('n').split('\t')
                self.e1_rele2t[e1].append((self.symbol2id[rel], self.symbol2id[e2], int(t)))  # 1-n
                self.e1_rele2t[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1], int(t)))  # n-1

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2t[ent]
            if len(neighbors) == 0:
                self.e1_degrees[id_] = 0
                neighbors.append((0, self.pad_id, 0))
            elif len(neighbors) > max_:
                self.e1_degrees[id_] = 1
                neighbors = neighbors[:max_]
            else:
                self.e1_degrees[id_] = 1
            degrees[ent] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # rel
                self.connections[id_, idx, 1] = _[1]  # tail
                self.connections[id_, idx, 2] = _[2]  # time
        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.Matcher.state_dict(), path)

    def load(self, path=None):
        if path:
            self.Matcher.load_state_dict(torch.load(path))
        else:
            self.Matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(self.device)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(self.device)
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        left = Variable(torch.LongTensor(left)).to(self.device)
        right = Variable(torch.LongTensor(right)).to(self.device)
        return (left, left_connections, left_degrees, right, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')
        best_mrr = 0.0
        # best_hits1 = 0.0
        # best_hits5 = 0.0
        # best_hits10 = 0.0

        best_mrr_batches = 0
        # best_hits1_batches = 0
        # best_hits5_batches = 0
        # best_hits10_batches = 0

        losses = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.few, self.symbol2id, self.ent2id,
                                   self.e1relt_e2):
            task_choice, support_time, query_time, false_time, support_left, support_right, query_left, query_right, false_left, false_right = data

            self.batch_nums += 1

            task_choice = Variable(torch.LongTensor(task_choice)).to(self.device)
            support_time = Variable(torch.LongTensor(support_time)).to(self.device)
            query_time = Variable(torch.LongTensor(query_time)).to(self.device)
            false_time = Variable(torch.LongTensor(false_time)).to(self.device)



            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)


            self.Matcher.train()
            positive_score = self.Matcher(task_choice,
                                          support_time,
                                          query_time,
                                          support_meta,
                                          query_meta
                                          )

            negative_score = self.Matcher(task_choice,
                                          support_time,
                                          false_time,
                                          support_meta,
                                          false_meta)

            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            lr = adjust_learning_rate(optimizer=self.optim, epoch=self.batch_nums, lr=self.lr,
                                      warm_up_step=self.warm_up_step,
                                      max_update_step=self.max_batches)
            losses.append(loss.item())

            self.optim.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                logging.info(
                    'Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}, '.format(
                        self.batch_nums,
                        np.mean(losses),
                        lr))
                self.writer.add_scalar('Avg_batch_loss_every_log', np.mean(losses), self.batch_nums)

            if self.batch_nums % self.eval_every == 0:
                for name, parms in self.Matcher.named_parameters():
                    print('-->name:', name)
                    print('-->para:', type(parms))
                    print('-->grad_requirs:', parms.requires_grad)
                    print('-->grad_value:', parms.grad)
                    print("===")
                logging.info('Batch_nums is %d' % self.batch_nums)
                hits10, hits5, hits1, mrr = self.eval(mode='dev')
                self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                self.writer.add_scalar('HITS1', hits1, self.batch_nums)
                self.writer.add_scalar('MRR', mrr, self.batch_nums)
                self.save()


                if mrr > best_mrr:
                    self.save(self.save_path + '_best_mrr')
                    best_mrr = mrr
                    best_mrr_batches = self.batch_nums
                logging.info('Best_mrr is {:.6f}, when batch num is {:d}'.format(best_mrr, best_mrr_batches))

                # if hits1 > best_hits1:
                #     self.save(self.save_path + '_best_hits1')
                #     best_hits1 = hits1
                #     best_hits1_batches = self.batch_nums
                # logging.info('Best_hits1 is {:.6f}, when batch num is {:d}'.format(best_hits1, best_hits1_batches))
                #
                # if hits5 > best_hits5:
                #     self.save(self.save_path + '_best_hits5')
                #     best_hits5 = hits5
                #     best_hits5_batches = self.batch_nums
                # logging.info('Best_hits5 is {:.6f}, when batch num is {:d}'.format(best_hits5, best_hits5_batches))
                #
                # if hits10 > best_hits10:
                #     self.save(self.save_path + '_best_hits10')
                #     best_hits10 = hits10
                #     best_hits10_batches = self.batch_nums
                # logging.info('Best_hits10 is {:.6f}, when batch num is {:d}'.format(best_hits10, best_hits10_batches))



            if self.batch_nums == self.max_batches:
                self.save()
                break

    def eval(self, mode='dev'):
        self.Matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []
        for query_ in test_tasks.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]


            support_left = [self.ent2id[triple[0]] for triple in support_triples]
            support_right = [self.ent2id[triple[2]] for triple in support_triples]
            support_time = [int(triple[3]) for triple in support_triples]
            support_meta = self.get_meta(support_left, support_right)

            task_choice = Variable(torch.LongTensor([symbol2id[query_]])).to(self.device)
            support_time = Variable(torch.LongTensor(support_time)).to(self.device)

            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_time = []
                time = int(triple[3])
                query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])
                query_left = []
                query_right = []
                query_left.append(self.ent2id[triple[0]])
                query_right.append(self.ent2id[triple[2]])
                query_time.append(time)
                for ent in candidates:
                    if (ent not in self.e1relt_e2[triple[0] + triple[1] + triple[3]]) and ent != true:
                        query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                        query_left.append(self.ent2id[triple[0]])
                        query_right.append(self.ent2id[ent])
                        query_time.append(time)

                query_time = Variable(torch.LongTensor(query_time)).to(self.device)

                query_meta = self.get_meta(query_left, query_right)
                scores = self.Matcher(task_choice,
                                      support_time,
                                      query_time,
                                      support_meta,
                                      query_meta
                                     )
                scores.detach()
                scores = scores.data

                scores = scores.cpu().numpy()

                sort = list(np.argsort(scores, kind='stable'))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f}, MRR:{:.3f}'.format(query_,
                                                                                               np.mean(
                                                                                                   hits10_),
                                                                                               np.mean(hits5_),
                                                                                               np.mean(hits1_),
                                                                                               np.mean(mrr_),
                                                                                               ))
            logging.info('Number of candidates: {}, number of test examples {}'.format(len(candidates), len(hits10_)))
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for test')
        self.eval(mode='test')

    def eval_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for dev')
        self.eval(mode='dev')


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    '''
    epoch: 训练的task数量
    '''
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = read_options()
    if not os.path.exists('./logs_'):
        os.mkdir('./logs_')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    seed_everything(args.seed)

    logging.info('*' * 100)
    logging.info('*** hyper-parameters ***')
    for k, v in vars(args).items():
        logging.info(k + ': ' + str(v))
    logging.info('*' * 100)

    trainer = Trainer(args)

    if args.test:
        trainer.test_(args.save_path + '_best_mrr')

    else:
        trainer.train()
        print('last checkpoint!')
        trainer.eval_()
        trainer.test_()
        print('best checkpoint!')
        trainer.test_(args.save_path + '_best_mrr')

