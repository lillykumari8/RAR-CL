import numpy as np
from copy import deepcopy
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_knn_sv


class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args
        self.k    = 0.03

        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        # TODO(change this:)
        if args.gen:
            if 'mnist' in args.dataset:
                img_size = 784
                economy = img_size // input_size[0]
            elif 'cifar' in args.dataset:
                img_size = 32 * 32 * 3
                economy = img_size // (input_size[0] ** 2)
            elif 'imagenet' in args.dataset:
                img_size = 84 * 84 * 3
                economy = img_size // (input_size[0] ** 2)
        else:
            economy = 1

        buffer_size = economy  * args.mem_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)
        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)

        if args.cuda:
            bx = bx.to(args.device)
            by = by.to(args.device)
            bt = bt.to(args.device)
            logits = logits.to(args.device)

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)

        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        #Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)


            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).uniform_(0, self.n_seen_so_far)
        indices = indices.to(x.device)
        # print("runs fine!!!")
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data].long()

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.by.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.bt.size(0), pdb.set_trace()

        assert idx_new_data.max() < x.size(0), pdb.set_trace()
        assert idx_new_data.max() < y.size(0), pdb.set_trace()

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]


    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno    = one_hot_y.sum(dim=0)
            per_class_acc     = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc    = per_class_acc
            self.valid_deno   = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]


    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.br = self.bt[:remove_after_this_idx]

    def sample(self, amt, exclude_task = None, exclude_labels=None, ret_ind = False):
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bt = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices]
        elif exclude_labels is not None:
            # all true tensor
            valid_indices = self.bt > -1
            for label in exclude_labels:
                valid_indices = valid_indices & (self.by != label)
            valid_indices = valid_indices.nonzero().squeeze()
            bx = self.bx[valid_indices]
            by = self.by[valid_indices]
            bt = self.bt[valid_indices]            
        else:
            bx, by, bt = self.bx[:self.current_index], self.by[:self.current_index], self.bt[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

            if self.args.cuda:
                indices = indices.to(self.args.device)

            if ret_ind:
                return bx[indices], by[indices], bt[indices], indices
            else:
                return bx[indices], by[indices], bt[indices]


    def sample_random(self, amt, exclude_task = None, exclude_labels=None, ret_ind = False):
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bt = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices]
        elif exclude_labels is not None:
            # all true tensor
            valid_indices = self.bt > -1
            for label in exclude_labels:
                valid_indices = valid_indices & (self.by != label)
            valid_indices = valid_indices.nonzero().squeeze()
            bx = self.bx[valid_indices]
            by = self.by[valid_indices]
            bt = self.bt[valid_indices]            
        else:
            valid_indices = None
            bx, by, bt = self.bx[:self.current_index], self.by[:self.current_index], self.bt[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                if valid_indices is None:
                    return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, bt, valid_indices
            else:
                return bx, by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

            if self.args.cuda:
                indices = indices.to(self.args.device)

            if ret_ind:
                if valid_indices is None:
                    return bx[indices], by[indices], bt[indices], indices
                else:
                    return bx[indices], by[indices], bt[indices], valid_indices[indices]
            else:
                return bx[indices], by[indices], bt[indices]


    def sample_class_balanced(self, amt, exclude_indices=None, ret_ind = False):
        filled_indices = np.arange(self.current_index)
        if exclude_indices is not None:
            exclude_indices = list(exclude_indices)
        else:
            exclude_indices = []
        valid_indices = np.setdiff1d(filled_indices, np.array(exclude_indices))

        bx = self.bx[valid_indices]
        by = self.by[valid_indices]
        bt = self.bt[valid_indices]
        
        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, bt
        else:
            class_count = by.bincount()
            # a sample's prob. of being sample is inv. prop to its class abundance
            class_sample_p = 1. / class_count.float() / class_count.size(0)
            per_sample_p   = class_sample_p.gather(0, by)
            balanced_indices = torch.multinomial(per_sample_p, amt)

            if self.args.cuda:
                balanced_indices = balanced_indices.to(self.args.device)
            if ret_ind:
                return bx[balanced_indices], by[balanced_indices], bt[balanced_indices], balanced_indices
            else:
                return bx[balanced_indices], by[balanced_indices], bt[balanced_indices]


    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]


def retrieve_by_knn_sv(args, model_temp, buffer, input_x, input_y):
    present = input_y.unique()
    cand_x, cand_y, cand_t, cand_ind = buffer.sample_random(args.subsample, exclude_labels=present, ret_ind=True)

    # Type 1 - Adversarial SV
    # Get evaluation data for type 1 (i.e., eval <- current input)
    eval_adv_x, eval_adv_y = input_x, input_y
    
    # Compute adversarial Shapley value of candidate data (i.e., sv wrt current input)    
    sv_matrix_adv = compute_knn_sv(args, model_temp, eval_adv_x, eval_adv_y, cand_x, cand_y)

    if args.aser_type != "neg_sv":
        excl_indices = set(cand_ind.tolist())
        eval_coop_x, eval_coop_y, _ = buffer.sample_class_balanced(args.coop_size, exclude_indices=excl_indices, ret_ind=False)
        # Compute Shapley value
        sv_matrix_coop = compute_knn_sv(args, model_temp, eval_coop_x, eval_coop_y, cand_x, cand_y)
        if args.aser_type == "asv":
            # Use extremal SVs for computation
            sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
        else: #asvu
            # Use mean variation for aser_type == "asvm" or anything else
            sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)                
    else:
        # aser_type == "neg_sv"
        # No Type 1 - Cooperative SV; Use sum of Adversarial SV only
        sv = sv_matrix_adv.sum(0) * -1

    ret_ind = sv.argsort(descending=True)
    ret_x = cand_x[ret_ind][:args.aser_num_retrieve]
    ret_y = cand_y[ret_ind][:args.aser_num_retrieve]
    return ret_x, ret_y