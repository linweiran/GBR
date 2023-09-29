import numpy as np
import torch
import torch.nn as nn

import time
import os
import sys
import torch.nn.functional as F
import pickle
import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import json
import math
import requests
import torch
from collections import OrderedDict

def test_hit(model, x,y_targets,batch_size,device='cpu'):
    model.eval()
    model.to(device)
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    success=0
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=torch.tensor(x[start_index:end_index]).float().to(device)
        pred=torch.argmax(model(x_batch),dim=1)
        select=torch.sum(sum(pred==j for j in y_targets).bool()).item()
        success+=select
    return success/total




def pgd_attack(model, x,y, batch_size=1024,eps=8/255, alpha=2/255, iters=40,device='cpu',L_dist='L_inf') :
    loss = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    adv=np.zeros(x.shape)
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=torch.tensor(x[start_index:end_index]).float().to(device)
        y_batch=torch.tensor(y[start_index:end_index]).long().to(device)
        adv_batch=x_batch.detach().clone()
        for j in range(iters) :
            adv_batch.requires_grad = True
            pred = model(adv_batch)
            model.zero_grad()
            cost = loss(pred, y_batch)
            cost.backward()
            grads=adv_batch.grad
            with torch.no_grad():
                if L_dist=='L_inf':
                    adv_batch = adv_batch + alpha*grads.sign()
                    eta = torch.clamp(adv_batch - x_batch, min=-eps, max=eps)
                    adv_batch = torch.clamp(x_batch + eta, min=0, max=1).detach().clone()
                    #adv_batch = torch.clamp(x_batch + eta, min=0, max=255).detach().clone()
                elif L_dist=='L_2':
                    st=time.time() 
                    x_adv_1 = adv_batch + alpha * grads / ((grads ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    #print (time.time()-st)
                    adv_batch = torch.clamp(x_batch + (x_adv_1 - x_batch) / (((x_adv_1 - x_batch) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15) * torch.min(
                        eps * torch.ones(x_batch.shape).to(device).detach(), ((x_adv_1 - x_batch) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    #print (time.time()-st)
                    """
                    #delta= alpha * grads /( torch.norm(grads.reshape(x_batch.shape[0], -1), p=2, dim=1) + 1e-15).reshape(-1, 1, 1, 1)
                    delta=alpha*grads/((grads ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    print (time.time()-st)
                    #delta_norm=torch.norm(delta.reshape(x_batch.shape[0], -1), p=2, dim=1)
                    delta_norm=((delta ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    print (time.time()-st)
                    factor=eps/delta_norm
                    print (time.time()-st)
                    factor = torch.min(factor, torch.ones_like(delta_norm))
                    print (time.time()-st)
                    delta = delta * factor.reshape(-1, 1, 1, 1)
                    print (time.time()-st)
                    adv_batch = torch.clamp(x_batch+delta,min=0, max=1)
                    print (time.time()-st)
                    """
                else:
                    raise NotImplementedError
        
        adv[start_index:end_index]=adv_batch.cpu().numpy()
    return adv
from basic_operations import test
#from autoattack.autoattack import AutoAttack
def autopgd(model,X,Y,seed,eps=8./255,device='cuda',bs=512,norm='Linf',glass=None):
    model.eval()
    print (Y.shape)
    attack=AutoAttack(model,eps=eps,device=device,seed=seed,attacks_to_run = ['apgd-dlr'],version='custom',norm=norm)#['apgd-dlr'],version='custom')
    adv_test=attack.run_standard_evaluation(torch.from_numpy(X).float(), torch.from_numpy(Y).long(),bs=bs,glass=glass)
    robustness=test(model,adv_test,Y,bs,device=device)
    return robustness

class PGD():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,n_iter_2_p=.22, n_iter_min_p=.06, size_decr_p=.03,
                 seed=0, eot_iter=1, nclass=10,verbose=False, device='cuda',alpha=None,loss='ce',rand_t=False):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.verbose = verbose
        self.device = device
        self.loss=loss
        self.alpha=alpha
        self.nclass=nclass
        self.thr_decr=0.75
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(n_iter_2_p * self.n_iter), 1), max(int(n_iter_min_p * self.n_iter), 1), max(int(size_decr_p * self.n_iter), 1)
        self.rand_t=rand_t

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def ce_loss(self, x, y_target):
        celoss=nn.CrossEntropyLoss(reduce=False, reduction='none')
        return celoss(x, y_target)
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0]).to(self.device)
        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def md_loss_single(self, x, y_target):
        #wr loss
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
        return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)


    def md_loss_max(self, x, y_targets):
        #wr loss max
        a=torch.arange(self.nclass)
        b=np.arange(self.nclass)
        y_nontargets=a[np.delete(b,y_targets.numpy())]
        x_targets=torch.index_select(x,1,y_targets.to(self.device))
        x_nontargets=torch.index_select(x,1,y_nontargets.to(self.device))
        x_target_max=torch.unsqueeze(torch.max(x_targets,1)[0],1).expand(-1,x_nontargets.shape[1])
        return torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target_max),dim=1)

    def md_loss_group(self, x, y_targets):
        #wr loss group
        a=torch.arange(self.nclass)
        b=np.arange(self.nclass)
        y_nontargets=a[np.delete(b,y_targets.numpy())]
        x_nontargets=torch.index_select(x,1,y_nontargets.to(self.device))
        ret=torch.ones(x.shape[0]).to(self.device)
        for y_target in y_targets:
            x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x_nontargets.shape[1])
            #ret*=torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1)
            ret+=torch.log(torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1))
        return ret

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] < x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def attack_single_run(self, x_in, y_in,y_targets,y_t=None):
        if self.loss=='ce':
            lo=self.ce_loss
        if self.loss=='md_single':
            lo=self.md_loss_single
        if self.loss=='md_max':
            lo=self.md_loss_max
        if self.loss=='md_group':
            lo=self.md_loss_group

        #loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        #loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        #acc_steps = torch.zeros_like(loss_best_steps)

        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        ret=x.clone().detach()
        #ret=torch.ones(x.shape[0])*(self.n_iter+1)
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))

        for i in range(self.n_iter+1):
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    if self.loss=='md_single' or self.loss=='ce':
                        loss_indiv = lo(logits,y_t)
                    else:
                        loss_indiv = lo(logits,y_targets)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            grad /= float(self.eot_iter)
            with torch.no_grad():
                if self.norm == 'Linf':
                    x_adv = x_adv - self.alpha * torch.sign(grad)
                    x_adv = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)
                pred = torch.argmax(logits, dim=1)
                select=torch.where(sum(pred==j for j in y_targets).bool())[0]
                ret[select]=x_adv[select].clone().detach()
                #ret[select]=torch.minimum(ret[select],i*torch.ones(x.shape[0])[select])
                #print (i,select.size())
        return ret

    def attack_single_run_auto(self, x_in, y_in,y_targets,y_t=None):
        if self.loss=='ce':
            lo=self.ce_loss
        if self.loss=='md_single':
            lo=self.md_loss_single
        if self.loss=='md_max':
            lo=self.md_loss_max
        if self.loss=='md_group':
            lo=self.md_loss_group
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        unit=torch.arange(x.size()[0]).to(self.device)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        x_best_adv=torch.round(x_best_adv*255)/255.
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        output = self.model(x)
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        """
        record_pert=np.zeros((100,x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        images=x.cpu().numpy()
        record_pred=np.zeros((100,x.shape[0]))
        """
        #losses=np.zeros((100,x.shape[0]))
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) 
                if self.loss=='md_single' or self.loss=='ce':
                    loss_indiv = lo(logits,y_t)
                else:
                    loss_indiv = lo(logits,y_targets)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() 
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0
                if self.norm == 'Linf':
                    x_adv_1 = x_adv - step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                elif self.norm == 'L2':
                    #print (((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12).detach().cpu().numpy())
                    """
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    """
                    x_adv_1 = x_adv - step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)



                x_adv = x_adv_1 + 0.
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    if self.loss=='md_single' or self.loss=='ce':
                        loss_indiv = lo(logits,y_t)
                    else:
                        loss_indiv = lo(logits,y_targets)
                    loss = loss_indiv.sum()
                    #losses[i]=loss_indiv.detach().cpu().numpy()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            grad /= float(self.eot_iter)
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 < loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              counter3 += 1
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() <= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      fl_oscillation = np.where(fl_oscillation)
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
            
            pert=x_adv-x
            pert_sign=pert.sign()
            pert_magnitude=pert.abs()
            pert=pert_sign*torch.floor(pert_magnitude*255)/255.
            #print (self.model(x+pert).max(1)[1].cpu().numpy())
            #pert=pert_sign*pert_magnitude
            #record_pert[i]=pert.detach().cpu().numpy()
            x_test=torch.clamp(x+pert,0,1)
            pred=self.model(x_test).max(1)[1]
            #record_pred[i]=pred.detach().cpu().numpy()
            #print (self.model(x_test)[0].detach().cpu().numpy())
            #print(pred.cpu().numpy())
            if self.rand_t:
                select=(pred==y_t)
            else:
                select=sum(pred==j for j in y_targets).bool()
            unchangable=torch.masked_select(unit,select)
            x_best_adv[unchangable]=x_test[unchangable]
            
            #x_best_adv=torch.round(x_best_adv*255)/255.
            #x_best_adv=torch.clamp(x_best_adv,0,1)
            #print ((self.model(x_best_adv).max(1)[1] != y_target).sum().item())
        x_best_adv = torch.clamp(torch.min(torch.max(x_best_adv, x - self.eps), x + self.eps), 0.0, 1.0)
        x_best_adv=torch.round(x_best_adv*255)/255.
        #with open('record','wb') as f:
            #pickle.dump((images,record_pred,record_pert),f)
        #with open('losses','wb') as f:
            #pickle.dump(losses,f)
        return x_best_adv,x_adv#, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in,y_targets,bs=512,auto=True):
        assert self.norm in ['Linf', 'L2']
        self.model.eval()
        self.model.to(self.device)
        #x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        #y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        x=torch.tensor(x_in).float().to(self.device)
        y=torch.tensor(y_in).long().to(self.device)

        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        if self.loss=='md_single' or self.loss=='ce':
            if self.rand_t :
                y_targets=torch.tensor(y_targets).long().to(self.device)
                n_batches = int(np.ceil(x.shape[0] / bs))
                ret=np.zeros(x.shape[0])
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x.shape[0])
                    x_to_fool, y_to_fool = x[start_idx:end_idx, :].clone().to(self.device), y[start_idx:end_idx].clone().to(self.device)
                    y_t=y_targets[start_idx:end_idx].clone().to(self.device)
                    if auto:
                        adv_samples,tmp=self.attack_single_run_auto(x_to_fool, y_to_fool,y_targets=y_t,y_t=y_t)
                    else:
                        adv_samples,tmp=self.attack_single_run(x_to_fool, y_to_fool,y_targets=y_t,y_t=y_t)
                    if self.verbose:
                        print('cum. time: {:.1f} s'.format(time.time() - startt))
                    pred=torch.argmax(self.model(adv_samples),dim=1)==y_t
                    ret[start_idx:end_idx]=pred.cpu().numpy()
            else:
                n_batches = int(np.ceil(x.shape[0] / bs))
                #adv_test=x.clone().detach()
                ret=np.zeros((y_targets.shape[0],x.shape[0]))
                c=0
                for y_t in y_targets:
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * bs
                        end_idx = min( (batch_idx + 1) * bs, x.shape[0])
                        x_to_fool, y_to_fool = x[start_idx:end_idx, :].clone().to(self.device), y[start_idx:end_idx].clone().to(self.device)
                        #adv_test[start_idx:end_idx] = self.attack_single_run(x_to_fool, y_to_fool,y_targets=y_targets,y_t=y_t)
                        if auto:
                            adv_samples,tmp=self.attack_single_run_auto(x_to_fool, y_to_fool,y_targets=y_targets,y_t=y_t)
                        else:
                            adv_samples,tmp=self.attack_single_run(x_to_fool, y_to_fool,y_targets=y_targets,y_t=y_t)
                        if self.verbose:
                            print('cum. time: {:.1f} s'.format(time.time() - startt))
                        pred=torch.argmax(self.model(adv_samples),dim=1)
                    
                        select=sum(pred==j for j in y_targets).bool().cpu().numpy()
                        ret[c][start_idx:end_idx]=select
                        #with open('record','wb') as f:
                        #    pickle.dump(adv_samples.detach().cpu().numpy(),f)
                    c+=1
        else:
            n_batches = int(np.ceil(x.shape[0] / bs))
            ret=np.zeros(x.shape[0])#x.clone().detach()
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x.shape[0])
                x_to_fool, y_to_fool = x[start_idx:end_idx, :].clone().to(self.device), y[start_idx:end_idx].clone().to(self.device)
                if auto:
                    adv_samples,_ = self.attack_single_run_auto(x_to_fool, y_to_fool,y_targets=y_targets,y_t=None)
                else:
                    adv_samples,_ = self.attack_single_run(x_to_fool, y_to_fool,y_targets=y_targets,y_t=None)
                if self.verbose:
                    print('cum. time: {:.1f} s'.format(time.time() - startt))
                pred=torch.argmax(self.model(adv_samples),dim=1)
                ret[start_idx:end_idx]=sum(pred==j for j in y_targets).bool().cpu().numpy()
        return ret#.cpu().numpy()

def APGD_caller(p,pairs,x_in, y_in,bs=512,auto=True):
    #p: attack instance
    #pairs: pairs of (source, target classes)
    dic={}
    for (s,t) in pairs:
        if s not in dic:
            dic[s]=list()
        dic[s].append(t)
    succ=0
    best=0
    worst=0
    num=0
    for source in dic:
        targets=dic[source]
        #print ([source],torch.tensor(targets))
        select=np.where(y_in==source)
        x=x_in[select]
        y=y_in[select]
        result=p.perturb( x, y,torch.tensor(targets),bs=bs,auto=True)
        n_samples=y.shape[0]
        if n_samples>0:
            if p.loss=='md_single':
                succ_current=np.mean(result)
                best_current=np.mean(np.sign(np.sum(result,axis=0)))
                worst_current=np.mean(1-np.sign(np.sum(1-result,axis=0)))
                succ=(succ*num)+succ_current*n_samples
                best=(best*num)+best_current*n_samples
                worst=(worst*num)+worst_current*n_samples
                num+=n_samples
                succ/=num
                best/=num
                worst/=num
            else:
                succ_current=np.mean(result)
                succ=(succ*num)+succ_current*n_samples
                num+=n_samples
                succ/=num
    if p.loss=='md_single':
        #print (succ,best,worst)
        return succ,best,worst
    else:
        #print (succ)
        return succ

def APGD_targeted(model,x,y,seed,num_classes,eps,norm,bs=512,device="cuda",n_iter=100,array_flag=False,target=None):
    #torch.random.manual_seed(0)
    #torch.cuda.random.manual_seed(0)
    model.eval()
    np.random.seed(0)
    if target is None:
    	target_offset=np.floor(np.random.rand(x.shape[0])*(num_classes-1))+1
    	targets=(y+target_offset) % num_classes
    else:
        targets=np.ones(x.shape[0])*target
    result=np.zeros(x.shape[0])


    apgd=PGD(model=model,nclass=num_classes,eps=eps,loss='md_single',norm=norm,rand_t=True,device=device,n_iter=n_iter)
    result=apgd.perturb( x, y,targets,bs=bs,auto=True)
    if (array_flag):
    	return result
    ret=np.mean(result)
    return ret




