import numpy as np
import math
from copy import deepcopy
import pdb
from sklearn.metrics import (euclidean_distances, f1_score, pairwise,
                             pairwise_distances, precision_score, recall_score)
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_grad_vector, get_future_step_parameters, label_smoothing, CustomLossFunction, mixup, cross_entropy_loss_mu
from collections import Counter
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import numpy as np 
from scipy.spatial.distance import cdist
from buffer import retrieve_by_knn_sv


#---------- Functions ------------#
dist_kl = lambda y, t_s : F.kl_div(F.log_softmax(y, dim=-1), F.softmax(t_s, dim=-1), reduction='mean') * y.size(0)
entropy_fn = lambda x : torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)
cross_entropy = lambda y, t_s : -torch.sum(F.log_softmax(y, dim=-1)*F.softmax(t_s, dim=-1),dim=-1).mean()
mse = torch.nn.MSELoss()


def gradient_wrt_feature(model, source_data, target_data, last=True, criterion=nn.MSELoss()):
    source_data.requires_grad = True
    if last:
        out = model(source_data)
        target = model(target_data).data.clone().detach()
    else:
        out = model.return_hidden(source_data)
        target = model.return_hidden(target_data).data.clone().detach()
    
    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()
    data_grad = source_data.grad.data
    return data_grad.clone().detach()


def Linf_distillation(args, model, dat, target, eps, alpha, steps, last=True, mu=1, momentum=True, rand_start=False):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(args.device)
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)
    g = torch.zeros_like(x_adv)

    # MI-FGSM
    for i in range(steps):
        # Calculate gradient in input space
        grad = gradient_wrt_feature(model, x_adv, target, last)
        with torch.no_grad():
            if momentum:
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                if args.dataset in ['permuted_mnist']:
                    grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1)
                else:
                    grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calculate new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv - alpha * new_grad.sign() # perturb the data to MINIMIZE loss on tgt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Respect image bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def retrieve_mir_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    MIR RAR
    '''
    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model
    
    present = input_y.unique()
    if args.method == 'mir_rar':
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_labels=present ,ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            logits_track_pre = model(bx)
            buffer_hid = model_temp.return_hidden(bx)
            logits_track_post = model_temp.linear(buffer_hid)
            
            hid = model_temp.return_hidden(input_x)
            logits = model_temp.linear(hid)

            pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
            post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
            scores = post_loss - pre_loss
            all_logits = scores
            big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]
            idx = subsample[big_ind]

            buffer_hid = buffer_hid[big_ind].cpu().data.numpy()
            logits_track_post = logits_track_post[big_ind].cpu().data.numpy()
            hid = hid.cpu().data.numpy()
            logits = logits.cpu().data.numpy()    

        mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]
        if args.d_last:
            dist_mat = cdist(logits_track_post, logits, metric=args.sim_metric)
        else:
            dist_mat = cdist(buffer_hid, hid, metric=args.sim_metric)

        similar_indices = list(dist_mat.argmin(axis=1))
        # prepare target input
        input_x = input_x[similar_indices, :]
        # print (input_x.shape, mem_x.shape)
        linf_distilled = Linf_distillation(args, model_temp, mem_x, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = F.cross_entropy(distilled_logits, mem_y)

    else:
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_labels=present )

    logits_buffer = model(mem_x)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, mem_y) + args.d_coeff * loss_distilled
    overall_loss.backward()

    if updated_inds is not None:
        buffer.logits[subsample[updated_inds]] = deepcopy(logits_track_pre[updated_inds])

    opt.step()
    return model


def retrieve_er_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ER RAR
    '''
    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)
    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model
    present = input_y.unique()
    if args.method == 'er_rar':
        assert (args.subsample == args.buffer_batch_size)
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_labels=present, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            if args.d_last:
                buffer_logits = model_temp(bx).cpu().data.numpy()
                batch_logits = model_temp(input_x).cpu().data.numpy()
            else:
                buffer_logits = model_temp.return_hidden(bx).cpu().data.numpy() # not logits, but pen-ultimate representation
                batch_logits = model_temp.return_hidden(input_x).cpu().data.numpy()

        dist_mat = cdist(buffer_logits, batch_logits, metric=args.sim_metric)
        similar_indices = list(dist_mat.argmin(axis=1))
        
        # prepare target input
        input_x = input_x[similar_indices, :]
        linf_distilled = Linf_distillation(args, model_temp, bx, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = F.cross_entropy(distilled_logits, by)

    else:
        bx, by, bt = buffer.sample(args.buffer_batch_size)

    logits_buffer = model(bx)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, by) + args.d_coeff * loss_distilled
    overall_loss.backward()

    opt.step()
    return model


def retrieve_mir_mixup_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    MIR mixup RAR
    '''
    
    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model
    
    present = input_y.unique()
    if args.method == 'mir_mixup_rar':
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_labels=present, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            logits_track_pre = model(bx)
            buffer_hid = model_temp.return_hidden(bx)
            logits_track_post = model_temp.linear(buffer_hid)
            
            hid = model_temp.return_hidden(input_x)
            logits = model_temp.linear(hid)

            pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
            post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
            scores = post_loss - pre_loss
            all_logits = scores
            big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]
            idx = subsample[big_ind]
            hid = hid.cpu().data.numpy()
            logits = logits.cpu().data.numpy()    

        mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]
        # mixup samples from mem_x
        mixup_mem_x, mixup_mem_y = mixup(args, mem_x, mem_y, 1.0, args.n_classes)
        buffer_hid = model_temp.return_hidden(mixup_mem_x)
        logits_track_post = model_temp.linear(buffer_hid)

        logits_track_post = logits_track_post.cpu().data.numpy()
        buffer_hid = buffer_hid.cpu().data.numpy()

        if args.d_last:
            dist_mat = cdist(logits_track_post, logits, metric=args.sim_metric)
        else:
            dist_mat = cdist(buffer_hid, hid, metric=args.sim_metric)
        similar_indices = list(dist_mat.argmin(axis=1))

        # prepare target input
        input_x = input_x[similar_indices, :]
        linf_distilled = Linf_distillation(args, model_temp, mixup_mem_x, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = cross_entropy_loss_mu(distilled_logits, mixup_mem_y)

    else:
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_labels=present)

    logits_buffer = model(mem_x)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, mem_y) + args.d_coeff * loss_distilled
    overall_loss.backward()

    if updated_inds is not None:
        buffer.logits[subsample[updated_inds]] = deepcopy(logits_track_pre[updated_inds])

    opt.step()
    return model


def retrieve_er_mixup_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ER mixup RAR
    '''
    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model
    
    present = input_y.unique()
    if args.method == 'er_mixup_rar':
        assert (args.subsample == args.buffer_batch_size)
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_labels=present, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():            
            hid = model_temp.return_hidden(input_x)
            logits = model_temp.linear(hid)
            hid = hid.cpu().data.numpy()
            logits = logits.cpu().data.numpy()    

        # mixup samples from mem_x
        mixup_mem_x, mixup_mem_y = mixup(args, bx, by, 1.0, args.n_classes)
        buffer_hid = model_temp.return_hidden(mixup_mem_x)
        logits_track_post = model_temp.linear(buffer_hid)

        logits_track_post = logits_track_post.cpu().data.numpy()
        buffer_hid = buffer_hid.cpu().data.numpy()

        if args.d_last:
            dist_mat = cdist(logits_track_post, logits, metric=args.sim_metric)
        else:
            dist_mat = cdist(buffer_hid, hid, metric=args.sim_metric)
        similar_indices = list(dist_mat.argmin(axis=1))

        # prepare target input
        input_x = input_x[similar_indices, :]
        linf_distilled = Linf_distillation(args, model_temp, mixup_mem_x, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = cross_entropy_loss_mu(distilled_logits, mixup_mem_y)

    else:
        bx, by, bt = buffer.sample(args.buffer_batch_size, exclude_labels=present)

    logits_buffer = model(bx)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, by) + args.d_coeff * loss_distilled
    overall_loss.backward()

    opt.step()
    return model


def retrieve_mir_mixup(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    MIR mixup replay
    '''

    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'mir_mixup':
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_labels=present, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            logits_track_pre = model(bx)
            buffer_hid = model_temp.return_hidden(bx)
            logits_track_post = model_temp.linear(buffer_hid)

            pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
            post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
            scores = post_loss - pre_loss
            all_logits = scores
            big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]
            idx = subsample[big_ind]

        mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]
    else:
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_labels=present)

    # mixup samples from mem_x
    mixup_mem_x, mixup_mem_y = mixup(args, mem_x, mem_y, 1.0, args.n_classes)    
    logits_buffer = model(mixup_mem_x)
    loss_distilled = cross_entropy_loss_mu(logits_buffer, mixup_mem_y)
    loss_distilled.backward()
    opt.step()
    return model


def retrieve_er_mixup(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ER mixup replay
    '''

    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'er_mixup':
        assert (args.subsample == args.buffer_batch_size)
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_labels=present)

    # mixup samples from mem_x
    mixup_mem_x, mixup_mem_y = mixup(args, mem_x, mem_y, 1.0, args.n_classes)    
    logits_buffer = model(mixup_mem_x)
    loss_distilled = cross_entropy_loss_mu(logits_buffer, mixup_mem_y)
    loss_distilled.backward()
    opt.step()
    return model



def retrieve_aser(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ASER replay
    '''

    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'aser':
        model_temp = deepcopy(model)
        mem_x, mem_y = retrieve_by_knn_sv(args, model_temp, buffer, input_x, input_y)

    logits_buffer = model(mem_x)
    F.cross_entropy(logits_buffer, mem_y).backward()
    opt.step()
    return model


def retrieve_aser_mixup(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ASER mixup replay
    '''
    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'aser_mixup':
        model_temp = deepcopy(model)
        mem_x, mem_y = retrieve_by_knn_sv(args, model_temp, buffer, input_x, input_y)

    # mixup samples from mem_x
    mixup_mem_x, mixup_mem_y = mixup(args, mem_x, mem_y, 1.0, args.n_classes)    
    logits_buffer = model(mixup_mem_x)
    loss_distilled = cross_entropy_loss_mu(logits_buffer, mixup_mem_y)
    loss_distilled.backward()
    opt.step()
    return model


def retrieve_aser_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ASER RAR replay
    '''

    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'aser_rar':
        model_temp = deepcopy(model)
        mem_x, mem_y = retrieve_by_knn_sv(args, model_temp, buffer, input_x, input_y)
        with torch.no_grad():
            buffer_hid_pre = model.return_hidden(mem_x)
            logits_track_pre = model.linear(buffer_hid_pre)

            buffer_hid_pre = buffer_hid_pre.cpu().data.numpy()
            logits_track_pre = logits_track_pre.cpu().data.numpy()
            hid = hid.cpu().data.numpy()
            logits = logits.cpu().data.numpy()

        if args.d_last:
            dist_mat = cdist(logits_track_pre, logits, metric=args.sim_metric)
        else:
            dist_mat = cdist(buffer_hid_pre, hid, metric=args.sim_metric)
        similar_indices = list(dist_mat.argmin(axis=1))

        # prepare target input
        input_x = input_x[similar_indices, :]
        model_temp = deepcopy(model)
        linf_distilled = Linf_distillation(args, model_temp, mem_x, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = F.cross_entropy(distilled_logits, mem_y)                       

    logits_buffer = model(mem_x)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, mem_y) + args.d_coeff * loss_distilled
    overall_loss.backward()
    opt.step()
    return model


def retrieve_aser_mixup_rar(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    '''
    ASER mixup RAR replay
    '''

    updated_inds = None
    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    present = input_y.unique()
    if args.method == 'aser_mixup_rar':
        model_temp = deepcopy(model)
        mem_x, mem_y = retrieve_by_knn_sv(args, model_temp, buffer, input_x, input_y)
        # mixup samples from mem_x
        mixup_mem_x, mixup_mem_y = mixup(args, mem_x, mem_y, 1.0, args.n_classes)
        model_temp = deepcopy(model)

        with torch.no_grad():
            buffer_hid_pre = model_temp.return_hidden(mixup_mem_x)
            logits_track_pre = model_temp.linear(buffer_hid_pre)

            buffer_hid_pre = buffer_hid_pre.cpu().data.numpy()
            logits_track_pre = logits_track_pre.cpu().data.numpy()
            hid = hid.cpu().data.numpy()
            logits = logits.cpu().data.numpy()

        if args.d_last:
            dist_mat = cdist(logits_track_pre, logits, metric=args.sim_metric)
        else:
            dist_mat = cdist(buffer_hid_pre, hid, metric=args.sim_metric)
        similar_indices = list(dist_mat.argmin(axis=1))

        # prepare target input
        input_x = input_x[similar_indices, :]
        linf_distilled = Linf_distillation(args, model_temp, mixup_mem_x, input_x, eps=args.d_eps, alpha=args.d_alpha, 
                                            steps=args.d_steps, last=args.d_last, momentum=args.d_momentum, rand_start=args.d_rand_start)
        distilled_logits = model(linf_distilled)
        loss_distilled = cross_entropy_loss_mu(distilled_logits, mixup_mem_y)

    logits_buffer = model(mem_x)
    overall_loss = (1-args.d_coeff) * F.cross_entropy(logits_buffer, mem_y) + args.d_coeff * loss_distilled
    overall_loss.backward()
    opt.step()
    return model