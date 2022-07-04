import argparse
import datetime
import hashlib
import json
import pathlib
import time
import sys
from torch import optim
import networkx as nx
import numpy as np
from collections import defaultdict
import os
from seal_attr import *
from utils import *
from mwmatching import maxWeightMatching


def load_dataset(root, name):
    root = pathlib.Path(root)   
    prefix = f'{name}-1.90'     
    total_nodes = []
    with open(root / f'{prefix}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        for x in edges:
            for i in x.split():
                total_nodes.append(int(i))
        edges = np.array([[int(i) for i in x.split()] for x in edges])  
    with open(root / f'{prefix}.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]  
    if (root / f'{prefix}.nodefeat.txt').exists():
        with open(root / f'{prefix}.nodefeat.txt') as fh:
            nodefeats = [x.split() for x in fh.read().strip().split('\n')]
            nodefeats = {int(k): [int(i) for i in v] for k, *v in nodefeats}
    else:
        nodefeats = None
    graph = Graph(edges); total_nodes = list(set(total_nodes))
    return graph, comms, nodefeats, prefix, total_nodes

def split_comms(graph, comms, train_size): 
    train_comms, test_comms = comms[:train_size], comms[train_size:] 
    n_valid = max(1, int(len(train_comms) * 0.3))  
    train_comms, valid_comms = train_comms[:-n_valid], train_comms[-n_valid:]
    train_comms = [list(x) for nodes in train_comms for x in graph.connected_components(nodes) if len(x) == len(nodes)] 
    valid_comms = [list(x) for nodes in valid_comms for x in graph.connected_components(nodes) if len(x) == len(nodes)]
    max_size = max(len(x) for x in train_comms + valid_comms)  
    return train_comms, valid_comms, test_comms, max_size

def constructGraph(filename, test_nodes, k_shell):
    G = nx.Graph(); k_shell_list = []; k_shell_sum = [] 
    with open(filename) as file:
        for line in file:
            line = line.strip('\n')
            head, tail = [int(x) for x in line.split(' ')]
            G.add_edges_from([(head, tail)])
    G.remove_edges_from(nx.selfloop_edges(G))

    degreeList = []; relations = defaultdict(list)
    list_k_shell_1 = sorted(nx.core_number(G).items(), key=lambda x: x[1]); switch = 'ON'
    for key in list_k_shell_1:
        if switch == 'ON':
            k_shell_value = key[1]; switch = 'OFF'
            relations[k_shell_value].append(int(key[0]))
            continue
        if key[1] == k_shell_value:
            relations[k_shell_value].append(int(key[0]))
        else:
            k_shell_value = key[1]
            relations[k_shell_value].append(int(key[0]))

    for num in relations.keys():
        if int(num) >= int(k_shell):
            k_shell_sum.extend(relations.get(num))  
    for node in test_nodes:
        if node in k_shell_sum:
            k_shell_list.append(node)

    return k_shell_list

def matching_score(set1, set2):
    """Calculates the matching score between two sets (e.g., a cluster and a complex)
    using the approach of Bader et al, 2001"""
    return len(set1.intersection(set2))**2 / (float(len(set1)) * len(set2))

def pretrain_g(g: Generator, train_comms, bs, n, writer, use_set=True):   
    for i in range(n):
        np.random.shuffle(train_comms)
        batch_loss = 0.
        for j in range(len(train_comms) // bs + 1): 
            batch = train_comms[j*bs:(j+1)*bs]   
            if len(batch) == 0:
                continue
            batch = [g.graph.sample_expansion_from_community(x) for x in batch]
            if use_set:                                 
                policy_loss = g.train_from_sets(batch)
            else:
                policy_loss = g.train_from_lists(batch)
            batch_loss += policy_loss
        batch_loss /= j + 1  
        if use_set:
            s = 'Set '
        else:
            s = 'List'
        if writer is not None:
            writer.add_scalar(f'Pretrain/GLoss{s.strip()}', batch_loss, i)
        print(f'[Pretrain-{s} {i+1:3d}] Loss = {batch_loss:2.4f}')


def pretrain_d(d: Discriminator, train_comms, fn, bs, n, writer):  
    for i in range(n):
        np.random.shuffle(train_comms)
        batch_loss = 0.
        batch_acc = 0.
        for j in range(len(train_comms) // bs + 1):
            true_comms = train_comms[j*bs:(j+1)*bs]
            if len(true_comms) == 0:
                continue
            seeds = np.random.choice(d.graph.n_nodes, size=bs, replace=False)
            fake_comms = fn(seeds)
            fake_comms = [x[:-1] if x[-1] == 'EOS' else x for x in fake_comms]
            loss, info = d.train_step(true_comms, fake_comms)
            acc = info['acc']
            batch_loss += loss
            batch_acc += acc
        batch_loss /= j + 1
        batch_acc /= j + 1
        if writer is not None:
            writer.add_scalar('Pretrain/DLoss', batch_loss, i)
            writer.add_scalar('Pretrain/DAcc', batch_acc, i)
        print(f'[Pretrain-D {i+1:3d}] Loss = {batch_loss: .2f} Acc = {batch_acc:.2f}')


def save_communities(comms, fname):
    with open(fname, 'w') as fh:
        s = '\n'.join([' '.join([str(i) for i in x]) for x in comms])
        fh.write(s)


class DummyWriter:

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Runner:

    def __init__(self, args):
        self.args = args
        # Data
        self.graph, *comms, self.eval_seeds, nodefeats, self.max_size, self.ds_name, self.total_nodes, self.k_shell_list = self.load_data()  
        self.train_comms, self.valid_comms, self.test_comms = comms
        # Save Dir and Pretrained Dir
        self.savedir, self.pretrain_dir, self.writer = self.init_dir()
        # Model
        self.device = torch.device('cuda:0')
        self.g = self.init_g(nodefeats)  
        self.d = self.init_d(self.g.nodefeats)

    def close(self):
        self.writer.close()

    def load_data(self):
        args = self.args
        graph, comms, nodefeats, ds_name, total_nodes = load_dataset(args.root, args.dataset)
        train_comms, valid_comms, test_comms, max_size = split_comms(graph, comms, args.train_size)
        args.ds_name = ds_name       
        args.max_size = max_size  
        eval_seeds = [min(x) for x in valid_comms] + [max(x) for x in valid_comms]
        test_nodes = {int(i) for x in test_comms for i in x}
        k_shell_list = constructGraph(pathlib.Path(args.root) / f'{ds_name}.ungraph.txt', test_nodes, args.k_shell)
        k_shell_list = np.array(list(set(k_shell_list)))
        total_nodes = np.array(list(set(total_nodes)))
        print(f'[{ds_name}] # Nodes: {graph.n_nodes} k_shell_list: {len(k_shell_list)}', flush=True)
        print(f'[# comms] Train: {len(train_comms)} Valid: {len(valid_comms)} Test: {len(test_comms)}', flush=True)
        return graph, train_comms, valid_comms, test_comms, eval_seeds, nodefeats, max_size, ds_name, total_nodes, k_shell_list

    def init_dir(self):
        args = self.args
        savedir = pathlib.Path(args.savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        writer = DummyWriter()
        with open(savedir / 'settings.json', 'w') as fh:
            arg_dict = vars(args)
            json.dump(arg_dict, fh, sort_keys=True, indent=4)
        pretrain_dir = pathlib.Path('pretrained')
        pretrain_dir.mkdir(exist_ok=True)
        return savedir, pretrain_dir, writer

    def init_g(self, nodefeats):
        args = self.args
        device = self.device
        g_model = Agent(args.hidden_size, args.with_attr).to(device)
        g_optimizer = optim.Adam(g_model.parameters(), lr=args.g_lr)
        g = Generator(self.graph, g_model, g_optimizer, device,
                      entropy_coef=args.entropy_coef,
                      n_rollouts=args.n_rollouts,
                      max_size=args.max_size,
                      max_reward=5.)
        if args.with_attr:
            attr_filename = self.pretrain_dir / f'{self.ds_name}.{g.conv}.{args.hidden_size}.npy'
            if attr_filename.exists():
                processed_attrs = np.load(str(attr_filename))
                g.load_nodefeats(processed_attrs)
                print(f'Load the processed node features. Shape={processed_attrs.shape}')
            else:
                print(f'Process the raw node features.')
                processed_attrs = g.preprocess_nodefeats(nodefeats)
                g.load_nodefeats(processed_attrs)
                np.save(str(attr_filename), processed_attrs)
                print(f'Save the feature. Shape={processed_attrs.shape}')
        return g

    def init_d(self, nodefeats):
        args = self.args
        d_model = GINClassifier(args.hidden_size, 3, dropout=args.d_dropout, feat_dropout=args.feat_dropout,
                                norm_type='batch_norm', agg_type='sum',
                                with_attr=args.with_attr and args.d_use_attr).to(self.device)
        d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr)
        d = Discriminator(self.graph, d_model, d_optimizer,
                          device=self.device,
                          log_reward=True,
                          max_boundary_size=args.max_boundary,
                          nodefeats=nodefeats)
        return d

    def evaluate_and_print(self, prefix=''):
        pred_comms = self.g.generate(self.eval_seeds)
        pred_comms = [x[:-1] if x[-1] == 'EOS' else x for x in pred_comms] 
        pred_comms = self.merge(pred_comms)
        recall, precision, f1, MRR = self.bm(self.valid_comms, pred_comms)
        print(f'[EVAL-{prefix}] F1={f1:.2f} Precision={precision:.2f} Recall={recall:.2f} MRR={MRR:.2f}')
        return precision, recall, f1, MRR

    def score_fn(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
        v = self.d.score_comms(cs)
        if args.locator_coef > 0:  
            v += args.locator_coef * self.l.score_comms(cs)
        if args.radius_penalty > 0:
            v -= args.radius_penalty * np.array([self.graph.subgraph_depth(x) for x in cs])
        return v

    def save(self, fname):
        data = {'g': self.g.model.state_dict(),
                'd': self.d.model.state_dict()}
        torch.save(data, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.g.model.load_state_dict(data['g'])
        self.d.model.load_state_dict(data['d'])

    def _pretrain(self):
        args = self.args
        pretrain_g(self.g, self.train_comms, args.g_batch_size, args.pretrain_list, writer=None, use_set=False)
        pretrain_g(self.g, self.train_comms, args.g_batch_size, args.pretrain_set, writer=None, use_set=True)
        pretrain_d(self.d, self.train_comms, self.g.generate, args.batch_size, args.pretrain_d, writer=None)

    def pretrain(self):
        args = self.args
        arg_dict = vars(args)
        pretrain_related_args = ['pretrain_list', 'pretrain_set', 'pretrain_d', 'hidden_size',
                                 'dataset', 'train_size', 'seed', 'max_size', 'max_boundary',
                                 'g_lr', 'd_lr', 'g_batch_size', 'with_attr', 'd_use_attr', 'ds_name']
        code = ' '.join([str(arg_dict[k]) for k in pretrain_related_args])
        code = hashlib.md5(code.encode('utf-8')).hexdigest().upper()
        print(f'CODE: {code}')
        pth_fname = self.pretrain_dir / f'{code}.pth'
        if pth_fname.exists():
            print('Load the pre-trained model!')
            self.load(pth_fname)
        else:
            self._pretrain()
            print('Save the pre-trained model!')
            self.save(pth_fname)

    def train_g_step(self, g_it):
        seeds = np.random.choice(self.graph.n_nodes, size=args.g_batch_size, replace=False)
        # Reinforcement Learning
        _, r, policy_loss, value_loss, entropy, length = self.g.train_from_rewards(seeds, self.score_fn)
        # Teacher Forcing
        if not args.without_tf:
            true_comms = random.choices(self.train_comms, k=args.g_batch_size)
            true_comms = [self.graph.sample_expansion_from_community(x) for x in true_comms]
            tf_loss = self.g.train_from_sets(true_comms)
        else:
            tf_loss = 0.
        self.writer.add_scalar('G/Reward', r, g_it)
        self.writer.add_scalar('G/PolicyLoss', policy_loss, g_it)
        # self.writer.add_scalar('G/ValueLoss', value_loss, g_it)
        self.writer.add_scalar('G/Entropy', entropy, g_it)
        self.writer.add_scalar('G/Length', length, g_it)
        self.writer.add_scalar('G/TFLoss', tf_loss, g_it)
        print(f'    Reward={r:.2f} PLoss={policy_loss: 2.2f} VLoss={value_loss:2.2f} '
              f'Entropy={entropy:1.2f} Length={length:2.1f} '
              f'TFLoss={tf_loss: 2.2f}')

    def examine_rewards_detail(self, it):
        seeds = np.random.choice(self.graph.n_nodes, size=self.args.batch_size, replace=False)
        generated_comms = [x[:-1] if x[-1] == 'EOS' else x for x in self.g.generate(seeds)]
        r_from_d = self.d.score_comms(generated_comms).mean()
        r_from_l = r_penalty = 0.
        self.writer.add_scalar('Reward/D', r_from_d, it)
        if args.locator_coef > 0:
            r_from_l = self.l.score_comms(generated_comms).mean()
            self.writer.add_scalar('Reward/L', r_from_l, it)
        if args.radius_penalty > 0:
            r_penalty = np.array([self.graph.subgraph_depth(x) for x in generated_comms]).mean()
            self.writer.add_scalar('Reward/R', r_penalty, it)
        total_r = r_from_d + args.locator_coef * r_from_l + args.radius_penalty * r_penalty
        self.writer.add_scalar('Reward/All', total_r, it)

    def train_d_step(self, d_it):
        bs = self.args.batch_size
        seeds = np.random.choice(self.graph.n_nodes, size=bs, replace=False)
        fake_comms = self.g.generate(seeds)
        fake_comms = [x[:-1] if x[-1] == 'EOS' else x for x in fake_comms]
        true_comms = random.choices(self.train_comms, k=bs)
        d_loss, info = self.d.train_step(true_comms, fake_comms)
        d_acc = info['acc']
        self.writer.add_scalar('D/Loss', d_loss, d_it)
        self.writer.add_scalar('D/Acc', d_acc, d_it)
        print(f'    Loss={d_loss: .2f} Acc={d_acc:.2f}')

    def merge(self, predcomms):
        redundancy = []; predicts = []
        for key1 in range(len(predcomms)):
            for k in range(key1+1, len(predcomms)):
                inter = list(set(predcomms[key1]) & set(predcomms[k]))
                if len(inter) == len(predcomms[key1]) and len(inter) == len(predcomms[k]):
                    redundancy.append(int(k))
        redundancy = list(set(redundancy))
        
        for key in range(len(predcomms)):
            switch = 0
            for k in redundancy:
                if int(key) == int(k):
                    switch = 1
            if switch == 0 and len(predcomms[key]) >= 2:
                predicts.append(predcomms[key])
        return predicts        

    def bm(self, complexes, predicts, _w = 0.2, _min_size = 2):
        comps = list(map(set, filter(lambda x:len(x)>=_min_size, complexes)))
        preds = list(map(set, filter(lambda x:len(x)>=_min_size, predicts)))
        confusion = np.zeros(len(comps)*len(preds)).reshape(len(comps), len(preds))
        for i in range(len(comps)):
            for j in range(len(preds)):
                confusion[i, j] = len(set.intersection(comps[i], preds[j]))
        
        w = np.zeros(len(comps)*len(preds)).reshape(len(comps), len(preds))
        for i in range(len(comps)):
            for j in range(len(preds)):
                w[i,j] = confusion[i,j]**2/(len(comps[i])*len(preds[j]))
            
        n_r = sum([1 if any(w[i,:]>=_w) else 0 for i in range(len(comps))])
        n_p = sum([1 if any(w[:,j]>=_w) else 0 for j in range(len(preds))])
        r = n_r/len(comps) if len(comps)>0 else 0
        p = n_p/len(preds) if len(preds)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)!=0 else 0
        ###MMR
        scores = {}
        count = 0
        n = len(comps)
        for id1, c1 in enumerate(comps):
            for id2, c2 in enumerate(preds):
                score = matching_score(c1, c2)
                if score == 1:
                    count = count + 1
                if score <= _w:
                    continue
                scores[id1, id2+n] = score
        inpt = [(v1, v2, w) for (v1, v2), w in scores.items()]
        mates = maxWeightMatching(inpt)
        score = sum(scores[i, mate] for i, mate in enumerate(mates) if i < mate)
        MMR = score / n

        return [r, p, f1, MMR]

    def run(self):
        # Eval before training: 
        self.evaluate_and_print('Init') 
        # Pretrain
        self.pretrain()  
        self.evaluate_and_print('Pretrained') 
        # Train
        d_it = g_it = -1
        for i_epoch in range(args.n_epochs):  
            print('=' * 20)
            print(f'[Epoch {i_epoch + 1:4d}]')
            tic = time.time()
            print('Update D')
            for _ in range(args.n_d_updates):  
                d_it += 1
                self.train_d_step(d_it)           
            print('Update G')
            for _ in range(args.n_g_updates):
                g_it += 1
                self.train_g_step(g_it)
            toc = time.time()
            print(f'Elapsed Time: {toc - tic:.1f}s')
            # Eval
            if (i_epoch + 1) % args.eval_every == 0:
                precision, recall, f1, MRR, = self.evaluate_and_print(f'Epoch {i_epoch+1:4d}')
                metrics_string = '_'.join([f'{x * 100:0>2.0f}' for x in [f1, precision, recall, MRR]])
                self.examine_rewards_detail(i_epoch)
                self.save(self.savedir / f'{i_epoch + 1:0>5d}_{metrics_string}.pth')
                self.writer.add_scalar('Eval/Precision', precision, i_epoch)
                self.writer.add_scalar('Eval/Recall', recall, i_epoch)
                self.writer.add_scalar('Eval/F1', f1, i_epoch)
                self.writer.add_scalar('Eval/MRR', MRR, i_epoch)
            if (i_epoch + 1) % args.test_every == 0:
                print('=' * 50)
                print('[Test]')
                pred_comms = self.g.generate(self.k_shell_list)
                pred_comms = [x[:-1] if x[-1] == 'EOS' else x for x in pred_comms]
                newPred_comms = self.merge(pred_comms)
                recall, precision, f1, MRR = self.bm(self.test_comms, newPred_comms)
                metrics_string = '_'.join([f'{x * 100:0>2.0f}' for x in [f1, precision, recall, MRR]])
                print(f'[EVAL-Test] F1={f1:.2f} Precision={precision:.2f} Recall={recall:.2f} MRR={MRR:.2f}')
                save_communities(newPred_comms, self.savedir / f'cmty.{i_epoch+1:0>4d}_{metrics_string}.txt')
                print("Save community")
  
def main(args):
    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yeast')
    #parser.add_argument('--dataset', type=str, default='human')
    parser.add_argument('--root', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=300)
    #parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--k_shell', type=int, default=1)
    parser.add_argument('--with_attr', action='store_true', default=False)
    parser.add_argument('--d_use_attr', action='store_true', default=False)
    # Model
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--g_lr', type=float, default=1e-2)
    parser.add_argument('--n_rollouts', type=int, default=5)
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--locator_coef', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=1e-2)
    parser.add_argument('--d_dropout', type=float, default=0.0)
    parser.add_argument('--feat_dropout', type=float, default=0.8)
    parser.add_argument('--s_dropout', type=float, default=0.5)
    parser.add_argument('--max_boundary', type=int, default=1000)
    parser.add_argument('--radius_penalty', type=float, default=0.)
    # Train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--g_batch_size', type=int, default=16)
    parser.add_argument('--pretrain_list', type=int, default=10)
    parser.add_argument('--pretrain_set', type=int, default=25)
    parser.add_argument('--pretrain_d', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_g_updates', type=int, default=5)
    parser.add_argument('--n_d_updates', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--without_tf', action='store_true', default=False)

    args = parser.parse_args()
    seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    args.savedir = f'ckpts/{args.dataset}/{now.strftime("%Y%m%d%H%M%S")}/'
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    main(args)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)