
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
    elif flip_case=='pruning':
        inds_sorted =  np.argsort(np.abs(x))
    else:
        raise
   

    inds_sorted = inds_sorted.copy()
    vals = x[inds_sorted]

    mse = []
    evidence = []
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
            inputs = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

        y = model(inputs, attention_mask = attention_mask, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().float().numpy()
        y = y.squeeze()

        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(softmax(y)[int(y_true)])

      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
    return mse, evidence, model_outs




def get_latest_checkpoint(path):
    import os, re
    checkpoints = [d for d in os.listdir(path) if d.startswith('checkpoint_')]
    if not checkpoints:
        print("no directory here")
        exit(1)
    return os.path.join(path, max(checkpoints, key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1))))