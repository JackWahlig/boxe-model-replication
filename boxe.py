'''
Reimplementation of BoxE
Code compiled from Google Colab notebook
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
import logging
import argparse
import logging
import argparse
import torch.nn.functional as F
import torch.nn as nn
from numpy import mean
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on {}'.format(device))

class BoxELoss():
  def __init__(self, options):
    if options.loss_type in ['uniform', 'u']:
      self.loss_fn = uniform_loss
      self.fn_kwargs = {'gamma': options.margin, 'w': 1/options.loss_k}
    elif options.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
      self.loss_fn = adversarial_loss
      self.fn_kwargs = {'gamma': options.margin, 'alpha': options.adversarial_temp}

  def __call__(self, entity_final_emb, box_low, box_high, neg_emb, neg_box_low, neg_box_high):
    return self.loss_fn(entity_final_emb, box_low, box_high, neg_emb, neg_box_low, neg_box_high, **self.fn_kwargs)


def dist(entity_emb, boxes):
  lb = boxes[0,:,:]  # lower boundries
  ub = boxes[1,:,:]  # upper boundries
  c = (lb + ub)/2  # centres
  w = ub - lb + 1  # widths
  k = 0.5*(w - 1) * (w - 1/w)
  return torch.where(torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)),
                     torch.abs(entity_emb - c) / w,
                     torch.abs(entity_emb - c) * w - k)

def score(r_headbox, r_tailbox, e_head, e_tail, order=2):
  # once the representaion of r is known this should probably just take r and the entities
  return torch.norm(dist(e_head, r_headbox), p=order) + torch.norm(dist(e_tail, r_tailbox), p=order, dim=1)

'''
Same as the score function, but adapted to the embedding representation created by the model forward pass
'''
def score_(point_embs, box_low, box_high, order=2):
  e_head, e_tail = point_embs[:,0,:], point_embs[:,1,:]
  headboxes, tailboxes = torch.stack((box_low[:,0,:], box_high[:,0,:]), dim=0), torch.stack((box_low[:,1,:], box_high[:,1,:]), dim=0)
  return score(headboxes, tailboxes, e_head, e_tail, order)


def uniform_loss(point_embs, box_low, box_high, negative_point_embs, negative_box_low, negative_box_high, gamma, w):
  e_head, e_tail = point_embs[:,0,:], point_embs[:,1,:]
  headboxes, tailboxes = torch.stack((box_low[:,0,:], box_high[:,0,:]), dim=0), torch.stack((box_low[:,1,:], box_high[:,1,:]), dim=0)
  s1 = - torch.log(torch.sigmoid(gamma - score(headboxes, tailboxes, e_head, e_tail)))
  s2_terms = []
  if not torch.is_tensor(w):
    w = torch.tensor([w]).repeat(len(negative_point_embs))
  for i in range(len(negative_point_embs)):
    ne_head, ne_tail = negative_point_embs[i,:,0,:], negative_point_embs[i,:,1,:]
    nheadboxes, ntailboxes = torch.stack((negative_box_low[i,:,0,:], negative_box_high[i,:,0,:]), dim=0), torch.stack((negative_box_low[i,:,1,:], negative_box_high[i,:,1,:]), dim=0)
    s2_terms.append(w[i] * torch.log(torch.sigmoid(score(nheadboxes, ntailboxes, ne_head, ne_tail) - gamma)))
  s2 = torch.sum(torch.stack(s2_terms), dim=0)
  return torch.sum(s1 - s2)

def triple_probs(point_embs, box_low, box_high, alpha):
  scores = []
  for i in range(len(point_embs)):
    e_head, e_tail = point_embs[i,:,0,:], point_embs[i,:,1,:]
    headboxes, tailboxes = torch.stack((box_low[i,:,0,:], box_high[i,:,0,:]), dim=0), torch.stack((box_low[i,:,1,:], box_high[i,:,1,:]), dim=0)
    scores.append(torch.exp(alpha * score(headboxes, tailboxes, e_head, e_tail)))
  scores = torch.stack(scores)
  div = torch.repeat_interleave(torch.sum(scores, dim=1).unsqueeze(1), repeats=scores.shape[1], dim=1)
  return torch.div(scores, div)


def adversarial_loss(point_embs, box_low, box_high, negative_point_embs, negative_box_low, negative_box_high, gamma, alpha):
  triple_weights = triple_probs(negative_point_embs, negative_box_low, negative_box_high, alpha)
  return uniform_loss(point_embs, box_low, box_high, negative_point_embs, negative_box_low, negative_box_high, gamma, triple_weights)

def get_loss_function(loss_type='uniform'):
  if loss_type in ['uniform', 'u']:
    return uniform_loss
  elif loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
    return adversarial_loss

'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/',
                        help='Path to datasets')
parser.add_argument('--data_name', default='FB15k',
                        help='Name of knowledge graph')
parser.add_argument('--margin', default=0.2, type=float,
                        help='Loss margin.')
parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training batch size.')
parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of the embedding.')
parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Learning rate.')
parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
parser.add_argument('--normed_bumps', action='store_true',
                        help='Do not normalize the image embeddings.')
parser.add_argument('--truncate_datasets', default=-1, type=int,
                        help='Truncate datasets to a subset of entries')
parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss')
parser.add_argument('--loss_k', default=1, type=float,
                        help='k parameter for uniform loss')
parser.add_argument('--loss_type', default='u', type=str,
                        help="Toggle between uniform ('u') and self-adversarial ('a') loss")
parser.add_argument('--num_negative_samples', default=10, type=int,
                        help="Number of negative samples per positive (true) triple")
opt = parser.parse_args(['--data_path', '', '--data_name', '', '--batch_size', '50', '--embedding_dim', '5',
                         '--learning_rate', '0.01', '--num_epochs', '50', '--truncate', '200', '--loss_type', 'a'])
print(opt)
kg = Data(opt, truncate=opt.truncate_datasets)
train_loader, val_loader, test_loader = kg.get_loaders()
opt.num_entity, opt.num_relation = kg.num_entity, kg.num_relation

model = BoxEModel(opt)
optimizer = Adam(model.parameters(), lr=opt.learning_rate)
loss_fn = BoxELoss(opt)

for epoch in range(opt.num_epochs):
  for i, data in enumerate(train_loader):
    # switch to train mode
    #model.train()
    negatives = kg.sample_negatives(data, 4)
    optimizer.zero_grad()
    entity_final_emb, box_low, box_high = model.forward(data)
    neg_emb, neg_box_low, neg_box_high = model.forward_negatives(negatives)
    loss = loss_fn(entity_final_emb, box_low, box_high, neg_emb, neg_box_low, neg_box_high)
    if i % 100 == 0:
      print('LOSS: {}'.format(loss))
    loss.backward()
    optimizer.step()
'''

#----------------------
# train.py
#----------------------

def main(): # Jiaqi
    settings = ['--data_path', '', '--data_name', '', '--batch_size', '1024',
                '--embedding_dim', '100', '--learning_rate', '0.001', '--num_epochs',
                '500', '--loss_type', 'u', '--num_negative_samples', '100',
                '--margin', '18', '--val_step', '100']
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='Path to datasets')
    parser.add_argument('--data_name', default='FB15k',
                        help='Name of knowledge graph')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training batch size.')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of the embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Learning rate.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--normed_bumps', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--truncate_datasets', default=-1, type=int,
                        help='Truncate datasets to a subset of entries')
    parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss')
    parser.add_argument('--loss_k', default=1, type=float,
                        help='k parameter for uniform loss')
    parser.add_argument('--loss_type', default='u', type=str,
                        help="Toggle between uniform ('u') and self-adversarial ('a') loss")
    parser.add_argument('--num_negative_samples', default=10, type=int,
                        help="Number of negative samples per positive (true) triple")
    opt = parser.parse_args(settings)

    print(opt)

    kg = Data(opt, truncate=opt.truncate_datasets)
    # Load data loaders
    train_loader, val_loader, test_loader = kg.get_loaders()

    opt.num_entity, opt.num_relation = kg.num_entity, kg.num_relation

    # Construct the model
    model = BoxEModel(opt).to(device)
    optim = Adam(model.parameters(), lr=opt.learning_rate)
    loss_fn = BoxELoss(opt)

    # Train the Model
    best_val_mrr, best_params = train(opt, model, train_loader, val_loader, optim, kg, loss_fn)

    # Test on test-data
    model.load_parameters(best_params)
    test_results = test(test_loader, opt, model, kg, loss_fn)
    return best_val_mrr, best_params, test_results

def test(dataloader, opt, model, knowledge_graph, loss_fn):
  print('Testing started')
  MR = []
  MRR = []
  hits1 = []
  hits3 = []
  accuracy = []
  outputs = {}
  for i, data in enumerate(dataloader):
      data = data.to(device)
      output = evaluate(opt, model, data, knowledge_graph, loss_fn)
      # val_loss.append(output['loss'])
      MR.append(output['MR'])
      MRR.append(output['MRR'])
      hits1.append(output['hits'][0])
      hits3.append(output['hits'][1])
      accuracy.append(output['accuracy'])
  # val_outputs['loss'] = np.mean(val_loss)
  outputs['MR'] = np.mean(MR)
  outputs['MRR'] = np.mean(MRR)
  outputs['hits1'] = np.mean(hits1)
  outputs['hits3'] = np.mean(hits3)
  outputs['accuracy'] = np.mean(accuracy)
  return outputs


'''
@brief Train the BoxE model with validation, print result
@param BoxE model, train_loader(torch.utils.data.DataLoader), val_loader(torch.utils.data.DataLoader)
@return None
'''
def train(opt, model, train_loader, val_loader, optimizer, knowledge_graph, loss_fn): # Jiaqi
    #optimizer = Adam(model.parameters(), lr=opt.learning_rate)
    print('Training started')
    best_mrr = -1
    best_model_params = None

    for epoch in range(opt.num_epochs):
        loss_list = []
        for i, train_data in enumerate(train_loader):
            # switch to train mode
            train_data = train_data.to(device)
            negatives = knowledge_graph.sample_negatives(train_data, opt.num_negative_samples)
            optimizer.zero_grad()
            entity_final_emb, box_low, box_high = model(train_data)
            neg_emb, neg_box_low, neg_box_high = model.forward_negatives(negatives)
            loss = loss_fn(entity_final_emb, box_low, box_high, neg_emb, neg_box_low, neg_box_high)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()
        # print mean loss of each epoch
        print('Epoch done')
        #print("epoch: {}; Loss: {}".format(epoch, np.mean(loss_list)))
        # validate on train and validation dataset
        if epoch % opt.val_step == 0:
            print('Validation checkpoint reached')
            # train data  #  this can go away
            # tr_loss = []
            # tr_MR = []
            # tr_MRR = []
            # tr_hits1 = []
            # tr_hits3 = []
            # tr_accuracy = []
            # train_outputs = {}
            # for i, train_data in enumerate(train_loader):
            #     train_data = train_data.to(device)
                # output = evaluate(opt, model, train_data, knowledge_graph, loss_fn)
                # tr_MR.append(output['MR'])
                # tr_MRR.append(output['MRR'])
                # tr_hits1.append(output['hits'][0])
                # tr_hits3.append(output['hits'][1])
                # tr_accuracy.append(output['accuracy'])
                # tr_loss.append(output['loss'])
            # train_outputs['loss'] = np.mean(tr_loss)
            # train_outputs['MR'] = np.mean(tr_MR)
            # train_outputs['MRR'] = np.mean(tr_MRR)
            # #train_outputs['hits'][0] = np.mean(tr_hits1)
            # #train_outputs['hits'][1] = np.mean(tr_hits1)
            # train_outputs['hits1'] = np.mean(tr_hits1)
            # train_outputs['hits3'] = np.mean(tr_hits3)
            # train_outputs['accuracy'] = np.mean(tr_accuracy)
            # print_result(opt, 'train', epoch, train_outputs)

            # valid data
            # val_loss = []
            val_MR = []
            val_MRR = []
            val_hits1 = []
            val_hits3 = []
            val_accuracy = []
            val_outputs = {}
            for i, val_data in enumerate(val_loader):
                val_data = val_data.to(device)
                output = evaluate(opt, model, val_data, knowledge_graph, loss_fn)
                # val_loss.append(output['loss'])
                val_MR.append(output['MR'])
                val_MRR.append(output['MRR'])
                val_hits1.append(output['hits'][0])
                val_hits3.append(output['hits'][1])
                val_accuracy.append(output['accuracy'])
            # val_outputs['loss'] = np.mean(val_loss)
            val_outputs['MR'] = np.mean(val_MR)
            val_outputs['MRR'] = np.mean(val_MRR)
            val_outputs['hits1'] = np.mean(val_hits1)
            val_outputs['hits3'] = np.mean(val_hits3)
            val_outputs['accuracy'] = np.mean(val_accuracy)
            print_result(opt, 'val', epoch, val_outputs)
            if val_outputs['MRR'] > best_mrr:
              best_mrr = val_outputs['MRR']
              best_model_params = model.parameter_dict()
    print('Training complete')
    return best_mrr, best_model_params

def print_result(opt, string, epoch, output):
    print("BoxEModel " + string + "results:")
    print("Number of Epochs: " + str(epoch))
    print("Training for " + str(opt.data_name))
    print("Learning Rate: " + str(opt.learning_rate))
    print("Embedding Dimension: " + str(opt.embedding_dim))
    # print("Loss: " + str(output['loss']))
    print("MR: " + str(output['MR']))
    print("MRR: " + str(output['MRR']))
    print("h@1: " + str(output['hits1']))
    print("h@3: " + str(output['hits3']))
    # print("h@5: " + output['hits'][2])
    # print("h@10: " + output['hits'][3])
    print("Accuracy: " + str(output['accuracy']))


'''
@brief Evaluate the training, validation and test dataset
@param BoxE model, data (train/val/test: [number of facts, 3] (sec_dim: relation, entity1, entity2))
@return outputs: evaluation results (type: a dictionary containing loss, MR, MRR, hits, accuracy)
'''
def evaluate(opt, model, data, knowledge_graph, loss_fn): # Jiaqi
    #model.eval()
    # negative_samples = knowledge_graph.sample_negatives(data, opt.num_negative_samples)
    # with torch.no_grad():
    #     entity_final_emb, box_low, box_high = model(data)
        # neg_emb, neg_box_low, neg_box_high = model.forward_negatives(negative_samples)

    # output evaluation results
    outputs = {}

    # evaluate input data
    # loss =  loss_fn(entity_final_emb, box_low, box_high, neg_emb, neg_box_low, neg_box_high)  # this can go away
    # scores = score_(entity_final_emb, box_low, box_high)

    # data mask of entity pairs in data
    # data_mask = torch.zeros([data.shape[0], opt.num_entity, opt.num_entity])
    # data mask of all entity pairs that need to be calculated for evaluate data
    # data_eval_mask = torch.zeros([data.shape[0], opt.num_entity, opt.num_entity])

    # store the scores of all r(_,t) and r(h,_)
    data_eval_score = torch.zeros([data.shape[0], opt.num_entity, opt.num_entity])
    # data need to be fed into model for evaluation (all data of r(_,t) and r(h,_))
    eval_data = []

    count = 0
    for item in data: # data: two-dimension->[batch_size, 3]  item: one-dimension->(rel, ent1, ent2)  eval_data: two-dimension->[data.shape[0]*opt.num_entity*2, 3]
        # data_mask[count, item[1], item[2]] = 1
        # data_eval_mask[count, item[1], :] = 1
        # data_eval_mask[count, :, item[2]] = 1
        for ent in range(opt.num_entity):
            eval_data.append([item[0], item[1], ent])
        for ent in range(opt.num_entity):
            eval_data.append([item[0], ent, item[2]])
        count += 1
    eval_data = torch.Tensor(eval_data)
    eval_data = eval_data.int().to(device)
    # eval_negatives = knowledge_graph.sample_negatives(eval_data, opt.num_negative_samples)
    eval_entity_final_emb, eval_box_low, eval_box_high = model(eval_data)
    # eval_neg_emb, eval_neg_box_low, eval_neg_box_high = model.forward_negatives(eval_negatives)  # this can go away
    # eval_loss = loss_fn(eval_entity_final_emb, eval_box_low, eval_box_high, eval_neg_emb, eval_neg_box_low, eval_neg_box_high)  # this can go away
    eval_scores = score_(eval_entity_final_emb, eval_box_low, eval_box_high)

    for item in range(data.shape[0]): # item: index of triple in this batch
        for ent in range(opt.num_entity):
              data_eval_score[item, data[item, 1], ent] = eval_scores[item*opt.num_entity*2 + ent]
        for ent in range(opt.num_entity):
              data_eval_score[item, ent, data[item, 2]] = eval_scores[item*opt.num_entity*2 + opt.num_entity + ent]

    mr, mrr, hits, acc = metrics(opt, data, data_eval_score, knowledge_graph)
    '''
    data: [number of facts, 3] (sec_dim: relation, entity1, entity2)
    data_eval_score: [number of facts, opt.num_entity, opt.num_entity]
             an adjacency matrix: (fisrt-dim: number of facts, sec_dim: number of entities, third_dim: number of entities)
             If the nth fact is r(h, t), there will be scores in data_eval_score[n,h,:] and data_eval_score[n,:,t]
    '''

    # outputs['loss'] = loss
    outputs['MR'] = mr
    outputs['MRR'] = mrr
    outputs['hits'] = hits
    outputs['accuracy'] = acc

    return outputs


'''
@brief Calculate the metrics for input data (MR, MRR. Accuracy, Hits)
@param opt parameters, data (train/val/test: [number of facts, 3] (sec_dim: relation, entity1, entity2))
@return calculated metrics
    [MR: Mean rank]
    [MRR: Mean reciprocal rank]
    [hits: a list containing Hits@1 and Hits@3]
    [ret_acc: accuracy of prediction (only consider our prediction to be correct when entity1 and entity2 both being predicted correctly)]
'''
def metrics(opt, data, eval_score, knowledge_graph): # Jiaqi
    num_fact = data.shape[0] # number of input facts
    num_entity = eval_score.shape[1]
    acc = []
    hit1 = []
    hit3 = []
    rank = []
    for i in range(num_fact):
        ent1 = data[i, 1]
        ent2 = data[i, 2]
        # iterate over eval_score[i, ent1, :] and eval_score[i, :, ent2], filter out positive facts except (i, ent1, ent2)
        for j in range(num_entity):
            if (data[i, 0], ent1, j) in knowledge_graph.facts_set and j != ent2:
                eval_score[i, ent1, j] = float('inf')
            if (data[i, 0], j, ent2) in knowledge_graph.facts_set and j != ent1:
                eval_score[i, j, ent2] = float('inf')
        # sort the rank of r(h,_) and r(_,t) (from high to low)
        row_rank = eval_score[i, ent1, :].topk(opt.num_entity, dim=0)[1]
        col_rank = eval_score[i, :, ent2].topk(opt.num_entity, dim=0)[1]
        # reverse the rank order
        row_rank = torch.flip(row_rank.reshape(row_rank.shape[0],1), [0,1])
        row_rank = row_rank.reshape(-1)
        col_rank = torch.flip(col_rank.reshape(col_rank.shape[0],1), [0,1])
        col_rank = col_rank.reshape(-1)
        # calculate the metrics
        rank.append((row_rank == ent2).nonzero().item() + 1)
        rank.append((col_rank == ent1).nonzero().item() + 1)
        hit1.append(1 if ((row_rank == ent2).nonzero().item() < 1) else 0)
        hit1.append(1 if ((col_rank == ent1).nonzero().item() < 1) else 0)
        hit3.append(1 if ((row_rank == ent2).nonzero().item() < 3) else 0)
        hit3.append(1 if ((col_rank == ent1).nonzero().item() < 3) else 0)
        acc.append(1 if (row_rank[0] == ent2 and col_rank[0] == ent1) else 0)
    MR = np.mean(np.array(rank))
    MRR = np.mean(1 / np.array(rank))
    ret_acc = np.mean(np.array(acc))
    ret_hit1 = np.mean(np.array(hit1))
    ret_hit3 = np.mean(np.array(hit3))
    hits = [ret_hit1, ret_hit3]
    return MR, MRR, hits, ret_acc

#----------------------
# model.py
#----------------------

class BoxEModel: #Jiaqi
    def __init__(self, opt):
        self.opt = opt
        self.embedding_dim = opt.embedding_dim
        self.num_entity = opt.num_entity
        self.num_relation = opt.num_relation
        self.data_name = opt.data_name
        self.sqrt_dim = torch.sqrt(torch.tensor(self.embedding_dim + 0.0))
        self.arity = 2

        self.entity_shape = [self.num_entity, self.embedding_dim]
        # initialize entity embedding
        self.entity_points = nn.Embedding(self.num_entity, self.embedding_dim)

        # initialize bump embedding
        self.entity_bumps = nn.Embedding(self.num_entity, self.embedding_dim)

        # Relation Embedding Instantiation
        rel_tbl_shape = [self.num_relation, self.arity, self.embedding_dim]
        scale_multiples_shape = [self.num_relation, self.arity, 1]

        # Variable box shape
        base_shape = rel_tbl_shape
        # the shape is learnable, define variables accordingly
        self.rel_shapes = Variable(torch.randn(rel_tbl_shape), requires_grad=True)
        # TODO: nomalization
        self.norm_rel_shapes = self.rel_shapes

        self.rel_bases, self.rel_deltas = \
            instantiate_box_embeddings(scale_multiples_shape, rel_tbl_shape,
                                       self.norm_rel_shapes)


    def to(self, device):
      self.entity_points = self.entity_points.to(device)
      self.entity_bumps = self.entity_bumps.to(device)
      self.rel_bases.requires_grad = False
      self.rel_bases = self.rel_bases.to(device)
      self.rel_bases.requires_grad = True
      self.rel_deltas.requires_grad = False
      self.rel_deltas = self.rel_deltas.to(device)
      self.rel_deltas.requires_grad = True
      return self

    def __call__(self, data):
      return self.forward(data)

    def forward(self, data):
        entity_base = self.entity_points(data[:, 1: self.arity + 1].to(torch.int64).to(device)).to(device)
        bump_emb = self.entity_bumps(data[:, 1: self.arity + 1].to(torch.int64).to(device)).to(device)
        # Application of bumps
        bump_sum = bump_emb.sum(dim=1, keepdims=True)
        entity_final_emb = entity_base + bump_sum - bump_emb

        # look up relation embedding
        batch_rel_bases = torch.index_select(self.rel_bases, 0, data[:, 0].to(torch.int64)).to(device)
        batch_rel_deltas = torch.index_select(self.rel_deltas, 0, data[:, 0].to(torch.int64)).to(device)
        # batch_rel_mults = torch.index_select(self.rel_multiples, 0, data[:, 0])

        box_low, box_high = compute_box(batch_rel_bases, batch_rel_deltas)
        return entity_final_emb, box_low, box_high

    def forward_negatives(self, negatives):
        entities = []
        box_low = []
        box_high = []
        for data in negatives:
          e, l, h = self.forward(data)
          entities.append(e)
          box_low.append(l)
          box_high.append(h)
        return torch.stack(entities), torch.stack(box_low), torch.stack(box_high)


    def parameters(self):
        return [self.rel_bases, self.rel_deltas, self.entity_points.weight, self.entity_bumps.weight]

    def parameter_dict(self):
      return {'rel_bases': self.rel_bases,
              'rel_deltas': self.rel_deltas,
              # 'rel_multiples': self.rel_multiples,
              'entity_points': self.entity_points.weight,
              'entity_bumps': self.entity_bumps.weight}

    def load_parameters(self, param_dict):
      self.rel_bases = param_dict['rel_bases']
      self.rel_deltas = param_dict['rel_deltas']
      # self.rel_multiples = param_dict['rel_multiples']
      self.entity_points.weight = param_dict['entity_points']
      self.entity_bumps.weight = param_dict['entity_bumps']

def add_padding(input_tensor): # Jiaqi
    return torch.cat([input_tensor, torch.zeros([1, input_tensor.shape[1]])], axis=0)

def instantiate_box_embeddings(scale_mult_shape, rel_tbl_shape, base_norm_shapes): # Jiaqi
    # scale_multiples = Variable(torch.randn(scale_mult_shape), requires_grad=True)
    # scale_multiples = F.elu(scale_multiples) + 1.0

    embedding_base_points = Variable(torch.randn(rel_tbl_shape), requires_grad=True) #box base points
    embedding_deltas = base_norm_shapes
    return embedding_base_points, embedding_deltas
    # embedding_deltas = torch.mul(scale_multiples, base_norm_shapes) #box width
    # return embedding_base_points, embedding_deltas, scale_multiples

def compute_box(box_base, box_delta): #Jiaqi
    box_second = box_base + 0.5 * box_delta
    box_first = box_base - 0.5 * box_delta
    box_low = torch.minimum(box_first, box_second)
    box_high = torch.maximum(box_first, box_second)
    return box_low, box_high

#----------------------
# data.py
#----------------------

RELATION_INDEX = 0
HEAD_INDEX = 1
TAIL_INDEX = 2

class Data:

    def __init__(self, opt, truncate=-1):
        # Directory path and batch size parameters
        self.data_path = opt.data_path
        self.data_name = opt.data_name
        self.batch_size = opt.batch_size

        # Parse data into numpy arrays
        self.train_data = self.parse_kg('train.txt')
        self.valid_data = self.parse_kg('valid.txt')
        self.test_data = self.parse_kg('test.txt')
        if truncate > 0:  # use only first few rows; for debugging/testing
          self.train_data = self.train_data[:truncate]
          self.valid_data = self.valid_data[:truncate]
          self.test_data = self.test_data[:truncate]

        # Set number of entities, relations, and set dictionaries
        self.num_entity, self.num_relation, \
        self.entity_to_id_dict, self.id_to_entity_dict, \
        self.relation_to_id_dict, self.id_to_relation_dict = self.ret_kg_info()

        # Convert data to ids
        self.train_data_ids = self.data_to_ids(self.train_data)
        self.valid_data_ids = self.data_to_ids(self.valid_data)
        self.test_data_ids = self.data_to_ids(self.test_data)

        # Create set of all facts from train, validation, and test
        self.facts_set = self.data_to_set(np.concatenate((self.train_data, self.valid_data, self.test_data), 0))

        # Create data loaders
        self.train_loader, self.valid_loader, self.test_loader = self.get_loaders()

    '''
    Parses txt data file and returns data in numpy format

    @param train_test_or_val either Strings 'train.txt', 'test.txt', or 'valid.txt' to determine file name
    @return data in numpy format
    '''
    def parse_kg(self, train_test_or_val):
        path_to_kg = self.data_path + self.data_name + train_test_or_val
        data = pd.read_table(path_to_kg, delim_whitespace=True, names=('head_entity', 'rel_name', 'tail_entity')).to_numpy()
        data[:, [1, 0]] = data[:, [0, 1]] # Gets data in [relation, head, tail] format

        return data

    '''
    Returns number of relations, entities, and dictionaries between relations, entities, and their ids

    @param None
    @return number of distinct entities in all datasets (train, test, and validation),
            number of distinct relations in all datasets
            dictionary mapping from entity names to entity ids
            dictionary mapping from entity ids to entity names
            dictionary mapping from relation names to relation ids
            dictionary mapping from relation ids to relation names
    '''
    def ret_kg_info(self):
        unique_heads = np.unique(np.concatenate((self.test_data[:,HEAD_INDEX], self.train_data[:,HEAD_INDEX], self.valid_data[:,HEAD_INDEX]), 0))
        unique_tails = np.unique(np.concatenate((self.test_data[:,TAIL_INDEX], self.train_data[:,TAIL_INDEX], self.valid_data[:,TAIL_INDEX]), 0))
        unique_ents = np.unique(np.concatenate((unique_heads, unique_tails), 0))

        unique_rels = np.unique(np.concatenate((self.test_data[:,RELATION_INDEX], self.train_data[:,RELATION_INDEX], self.valid_data[:,RELATION_INDEX]), 0))

        ent_to_id_dict, id_to_ent_dict = self.get_entity_dicts(unique_ents)
        rel_to_id_dict, id_to_rel_dict = self.get_relation_dicts(unique_rels)

        return [len(unique_ents), len(unique_rels), ent_to_id_dict, id_to_ent_dict, rel_to_id_dict, id_to_rel_dict]

    '''
    Creates dictionaries mapping to and from entity names (Strings) and ids

    @param the list of distinct entities
    @return dictionary mapping from entity names to ids and dictionary mapping from entity ids to names
    '''
    def get_entity_dicts(self, unique_ents):
        ent_to_id_dict = {}
        id_to_ent_dict = {}
        for i in range(len(unique_ents)):
            ent_to_id_dict[unique_ents[i]] = i
            id_to_ent_dict[i] = unique_ents[i]

        return [ent_to_id_dict, id_to_ent_dict]

    '''
    Creates dictionaries mapping to and from relation names (Strings) and ids

    @param the list of distinct relation
    @return dictionary mapping from relation names to ids and dictionary mapping from relation ids to names
    '''
    def get_relation_dicts(self, unique_rels):
        rel_to_id_dict = {}
        id_to_rel_dict = {}
        for i in range(len(unique_rels)):
            rel_to_id_dict[unique_rels[i]] = i
            id_to_rel_dict[i] = unique_rels[i]

        return [rel_to_id_dict, id_to_rel_dict]

    '''
    Takes data with string names (entities or relations) and returns data with ids instead

    @param the data to be mapped to ids
    @return numpy holding ids of all data
    '''
    def data_to_ids(self, data):
        ids = []
        for i in range(len(data)):
            arr = np.empty(3)
            arr[RELATION_INDEX] = self.relation_to_id_dict[data[i][RELATION_INDEX]]
            arr[HEAD_INDEX] = self.entity_to_id_dict[data[i][HEAD_INDEX]]
            arr[TAIL_INDEX] = self.entity_to_id_dict[data[i][TAIL_INDEX]]
            ids.append(arr)

        return np.array(ids, dtype=np.int64)

    '''
    Takes numpy array of [relation, head, tail] arrays and turns it into Python set of tuples

    @param data to be transformed
    @return python set of (relation, head, tail) tuples
    '''
    def data_to_set(self, data):
        data_list = data.tolist()
        return set(tuple(x) for x in data_list)

    '''
    Returns data loaders fro training, validation, and testing sets

    @param the batch size for the loades
    @return data loader objects for train, validation, and test
    '''
    def get_loaders(self):
        return [torch.utils.data.DataLoader(self.train_data_ids, self.batch_size, shuffle=True), \
                torch.utils.data.DataLoader(self.valid_data_ids, self.batch_size, shuffle=True), \
                torch.utils.data.DataLoader(self.test_data_ids, self.batch_size, shuffle=True)]

    def resample(self, triples):
      no_replacement_performed = True
      for i, r in enumerate(triples[0]):
        if (r, triples[1, i], triples[2, i]) in self.train_data:
          no_replacement_performed = False
          new_index = torch.randint(self.num_entity, (1,))
          new_e = torch.tensor(list(self.id_to_entity_dict.keys()))[new_index].to(device)
          is_head = torch.randint(2, (1,)) == 1
          if is_head.item():
            triples[1,i] = new_e.item()
          else:
            triples[2,i] = new_e.item()
      return triples, no_replacement_performed


    def sample_negatives(self, triples, nb_samples):
      triples_t = triples.transpose(0,1)
      batch_size = len(triples_t[0])
      triples_rep = torch.repeat_interleave(triples_t, nb_samples, dim=1)
      # sample random entities
      sample_index = torch.randint(self.num_entity, size=(batch_size*nb_samples,))
      sample_ids = torch.tensor(list(self.id_to_entity_dict.keys()))[sample_index].to(device)
      is_head = (torch.randint(2, size=(batch_size*nb_samples,)) == 1) # indicate if head is being replaced (otherwise, replace tail)
      # create sampled triples from sampled entities
      replace_mask = torch.stack((torch.zeros(len(is_head)), is_head, ~is_head)).to(device)
      inverse_replace_mask = torch.stack((torch.ones(len(is_head)), ~is_head, is_head)).to(device)
      replacements = torch.stack((triples_rep[0], sample_ids, sample_ids)).to(device)
      sampled_triples = (replace_mask * replacements + inverse_replace_mask * triples_rep).to(device)
      # filter out and replace known positive triples
      filtering_done = False
      while not filtering_done:
        sampled_triples, filtering_done = self.resample(sampled_triples)
      return sampled_triples.reshape((3, batch_size, nb_samples)).long().transpose(0,2)
