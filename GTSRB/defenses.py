import numpy as np
import torch
import torch.nn as nn
import sys
import time
import torch.nn.functional as F
from basic_operations import test,train_epoch
from attacks import pgd_attack,autopgd, APGD_caller,PGD,APGD_targeted
import pickle
import numpy as np
def FreeAdvTrain_epoch(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf'):
    model.train()
    model.to(device)
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    delta_batch = torch.zeros(x[0:batch_size].shape).to(device)
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=x[start_index:end_index]
        y_batch=y[start_index:end_index]
        delta_batch=delta_batch[0:end_index-start_index]
        for j in range(m):
            optim.zero_grad()
            model.zero_grad()
            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)
            adv_batch.requires_grad = True
            output = model(adv_batch)
            loss=F.cross_entropy(output,y_batch)
            loss.backward()
            optim.step()
            if L_dist=='L_inf':
                delta_batch = delta_batch + eps*adv_batch.grad.sign()
                delta_batch = torch.clamp(delta_batch, min=-eps, max=eps)
            elif L_dist=='L_2':
                grads=adv_batch.grad
                with torch.no_grad():
                    delta_batch+=eps*grads / ((grads ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    delta_batch=delta_batch/ ((delta_batch ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)* torch.min(
                        eps * torch.ones(delta_batch.shape).to(device).detach(), ((delta_batch) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt())

            else:
                raise NotImplementedError
        
        

        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()

    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim


def md_loss_group(x, y_targets,nclass,device):
    #mdmul loss
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.ones(x.shape[0]).to(device)
    for y_target in y_targets:
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x_nontargets.shape[1])
        ret+=torch.log(torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1))
    return ret.sum()



def md_group_train(x, y_targets,nclass,device):
    train loss
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.zeros(x.shape[0]).to(device)
    x_nontarget_max=torch.max(x_nontargets,1)[0]
    for y_target in y_targets:
        x_target=x[np.arange(x.shape[0]), y_target]
        ret+=nn.functional.relu(x_target+1e-15-x_nontarget_max)
    return ret.sum()


def md_loss_targeted(x,y_target):
    x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
    return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)




def GroupAdvTrain_epoch(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',v=0.1):
    model.train()
    model.to(device)
    origins=[0,1,2,3,4,5,6,7,8]
    pairs=[(0,14),(0,15),(0,17),(1,14),(1,15),(1,17),(2,0),(2,14),(2,15),(2,17),(3,0),(3,1),(3,14),(3,15),(3,17),(4,0),(4,1),(4,14),(4,15),(4,17),(5,0),(5,1),(5,14),(5,15),(5,17),(6,0),(6,1),(6,14),(6,15),(6,17),(7,0),(7,1),(7,2),(7,14),(7,15),(7,17),(8,0),(8,1),(8,2),(8,3),(8,14),(8,15),(8,17)]
    targets={}
    for pair in pairs:
        if pair[0] not in targets:
            targets[pair[0]]=list()
        targets[pair[0]].append(pair[1])
    total=x.shape[0]
    batches=np.floor(total/batch_size).astype(int)
    l1=0
    l2=0
    for i in range(batches):

        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=x[start_index:end_index]
        y_batch=y[start_index:end_index]

        alllength=torch.isin(y, torch.tensor(origins).long().to(device)).sum().item()
        position=x_batch.shape[0]
        
        for origin in origins:
            select= (y==origin)
            x_select=x[select]
            y_select=y[select]
            length=select.sum().item()


            N=np.ceil(128*length/alllength).astype(int)
            start=np.random.randint(length-N)
            x_batch=torch.cat((x_batch,x_select[start:start+N]),0)
            y_batch=torch.cat((y_batch,y_select[start:start+N]),0)
        delta_batch = torch.zeros(x_batch.shape).to(device)
        for j in range(m):
            model.train()

            optim.zero_grad()
            model.zero_grad()

            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)

            adv_batch.requires_grad = True
            output = model(adv_batch)
            loss=F.cross_entropy(output[:position],y_batch[:position])
            l1+=loss.item()
            outputp=output[position:]
            y_batchp=y_batch[position:]
            for origin in origins:
                loss+=md_group_train(outputp[y_batchp==origin], torch.tensor(targets[origin]).long(),43,device)*v
            l2+=loss.item()
            loss.backward()
            optim.step()

            model.eval()
            model.zero_grad()

            adv_batch.requires_grad = True
            output = model(adv_batch)
            loss=0
            for origin in origins:
                loss-=md_loss_group(output[y_batch==origin], torch.tensor(targets[origin]).long(),43,device)
            grads=torch.autograd.grad(loss, [adv_batch])[0].detach()
            if L_dist=='L_inf':
                delta_batch = delta_batch + eps*grads.sign()
                delta_batch = torch.clamp(delta_batch, min=-eps, max=eps)
            elif L_dist=='L_2':
                with torch.no_grad():
                    delta_batch+=eps*grads / ((grads ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    delta_batch=delta_batch/ ((delta_batch ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)* torch.min(
                        eps * torch.ones(delta_batch.shape).to(device).detach(), ((delta_batch) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt())

            else:
                raise NotImplementedError


            delta_batch[:position]=0
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim






def FreeAdvTrain(model,optim, trainx,trainy,testx,testy,batch_size,max_iters=5,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',nclass=10,x_val=None,y_val=None,origins=None,pairs=None):
    if L_dist=='L_inf':
        norm='Linf'
    else:
        norm='L2'

    x=torch.tensor(trainx).float().to(device)
    y=torch.tensor(trainy).long().to(device)
    for i in range(max_iters):
        model,optim=FreeAdvTrain_epoch(model,optim,x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
    return model


def GroupTrain(model,optim, trainx,trainy,testx,testy,batch_size,max_iters=5,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',nclass=10,x_val=None,y_val=None,origins=None,pairs=None,v=0.1):
    if L_dist=='L_inf':
        norm='Linf'
    else:
        norm='L2'

    x=torch.tensor(trainx).float().to(device)
    y=torch.tensor(trainy).long().to(device)
    for i in range(max_iters):
        model,optim=GroupAdvTrain_epoch(model,optim, x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist,v=v)
        model.eval()

    return model


