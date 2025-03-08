
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np


IMMUTABLE_TOKENS_SUFFIX = ['Please', '▁make', '▁sure', '▁to', '▁answer', '▁(', 'A', ',', 'B', ',', 'C', ',', '▁or', '▁D', ')', '<0x0A>', 'Answer', '▁is', ':']
IMMUTABLE_TOKENS_PREFIX = ['<s>', '▁The', '▁following', '▁is', '▁multiple', '▁choice', '▁question', '▁(', 'with', '▁answers', ').']

SHOULD_KEEP = True

def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def flip(model, x, token_ids, tokens, y_true,attention_mask,  fracs, flip_case,random_order = False, tokenizer=None, device='cpu'):

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

    y0 = model(inputs0, attention_mask=attention_mask, labels = None)['logits'].squeeze().detach().cpu().float().numpy()
    orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    if flip_case=='generate':
        inds_sorted = np.argsort(x)[::-1]
        inds_sorted2 =  np.argsort(np.abs(x))

    elif flip_case=='pruning':
        inds_sorted =  np.argsort(np.abs(x))
    else:
        raise
   

    inds_sorted = inds_sorted.copy()
    vals = x[inds_sorted]

    mse = []
    evidence = []
    evidence_logit = []
    evidence_insert_least_relevant = [] #calculated for pruning
    evidence_insert_least_relevant_logit = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().cpu().numpy(), 'y0':y0}

    N=len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':

            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        elif flip_case == 'generate':
            inds_generator2 = iter(inds_sorted2)
            n_flip2=int(np.ceil(frac*N))
            inds_flip2 = [next(inds_generator2) for i in range(n_flip2)]
            inputs = UNK_IDX*torch.ones_like(inputs0)
            inputs2 = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0
            inputs2[inputs2==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]
            for i in inds_flip2:
                inputs2[:,i] = inputs0[:,i]

        y = model(inputs, attention_mask = attention_mask, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().float().numpy()
        y = y.squeeze()

        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(softmax(y)[int(y_true)])
        evidence_logit.append(y[int(y_true)])

        if flip_case == "generate":
            y2 = model(inputs2, attention_mask = attention_mask, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().float().numpy()
            y2 = y2.squeeze()
            evidence_insert_least_relevant.append(softmax(y2)[int(y_true)])
            evidence_insert_least_relevant_logit.append(y2[int(y_true)])
      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
       
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
    return mse, evidence, evidence_logit, model_outs, evidence_insert_least_relevant, evidence_insert_least_relevant_logit






def flip_arc(model, x, highest_idx, logit0, token_ids, tokens, y_true, should_keep,attention_mask,  fracs, flip_case,random_order = False, tokenizer=None, device='cpu'):

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

 
    #y0 = model(inputs0)['logits'].squeeze().detach().cpu().float().numpy()
    #orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    if flip_case=='generate':
        inds_sorted = np.argsort(x)[::-1]
        inds_sorted2 =  np.argsort(np.abs(x))

        if should_keep:
            inds_sorted = inds_sorted[(inds_sorted >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted <= (len(inds_sorted) - len(IMMUTABLE_TOKENS_SUFFIX)))]
            inds_sorted2 = inds_sorted2[(inds_sorted2 >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted2 <= (len(inds_sorted2) - len(IMMUTABLE_TOKENS_SUFFIX)))]

      
    elif flip_case=='pruning':
        #CHANGEHERE
        inds_sorted  =  np.argsort(np.abs(x))
        if should_keep:
            inds_sorted = inds_sorted[(inds_sorted >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted <= (len(inds_sorted) - len(IMMUTABLE_TOKENS_SUFFIX)))]

    else:
        raise
   

    inds_sorted = inds_sorted.copy()
   
    mse = []
    evidence = []
    evidence_logit = []

    evidence_insert_least_relevant = [] #calculated for pruning
    evidence_insert_least_relevant_logit = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().float().cpu().numpy()}


    #CHANGEHERE

    N= len(x)
    if should_keep:
        N  = N -len(IMMUTABLE_TOKENS_SUFFIX) - len(IMMUTABLE_TOKENS_PREFIX) +1

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        #print("here")
        #print(N)
        #print(len(inds_sorted))

        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':



            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

            #CHANGEHERE
            if should_keep:
                inputs[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]

        elif flip_case == 'generate':

            inds_generator2 = iter(inds_sorted2)
            n_flip2=int(np.ceil(frac*N))
            inds_flip2 = [next(inds_generator2) for i in range(n_flip2)]
            inputs = UNK_IDX*torch.ones_like(inputs0)
            inputs2 = UNK_IDX*torch.ones_like(inputs0)

            # Set pad tokens
            inputs[inputs0==0] = 0
            inputs2[inputs2==0] = 0


            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

            for i in inds_flip2:
                inputs2[:,i] = inputs0[:,i]
            
            
            #CHANGEHERE
            if should_keep:
                inputs[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]
                inputs2[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs2[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]
        
        
        y = model(inputs)['logits'][0, -1, :].detach().cpu().float().numpy()
        y_pred = y[highest_idx]
        #print(y.shape)
        #print(y_pred)
        #print(logit0)
        #exit(1)
        y_pred = y_pred.squeeze().item()
        #print(logit0)
        #print(y_pred)

        err = np.sum((logit0-y_pred)**2)
        mse.append(err)
        #print(y.shape)
        #print(softmax(y).shape)
        #print(highest_idx)


        evidence.append(softmax(y)[highest_idx])
        evidence_logit.append(y[highest_idx])

        if flip_case == "generate":
            y2 = model(inputs2)['logits'][0, -1, :].detach().cpu().float().numpy()
            y_pred2 = y2[highest_idx]
            y_pred2 = y_pred2.squeeze().item()
            #print("=======================")
            #print(logit0)
            #print(y_pred2)
            #print("=======================")

            evidence_insert_least_relevant.append(softmax(y2)[highest_idx])
            evidence_insert_least_relevant_logit.append(y2[highest_idx])



      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().float().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:



        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
   
    return mse, evidence, evidence_logit, model_outs, evidence_insert_least_relevant, evidence_insert_least_relevant_logit







def get_latest_checkpoint(path):
    import os, re
    checkpoints = [d for d in os.listdir(path) if d.startswith('checkpoint_')]
    if not checkpoints:
        print("no directory here")
        exit(1)
    return os.path.join(path, max(checkpoints, key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1))))