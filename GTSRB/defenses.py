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
                    """
                    grads_norms = torch.norm(grads.reshape(x_batch.shape[0], -1), p=2, dim=1) + 1e-15
                    grads = grads / grads_norms.reshape(x_batch.shape[0], 1, 1, 1)
                    delta_batch = delta_batch + grads * eps
                    """
                    delta_batch+=eps*grads / ((grads ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)
                    delta_batch=delta_batch/ ((delta_batch ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-15)* torch.min(
                        eps * torch.ones(delta_batch.shape).to(device).detach(), ((delta_batch) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt())

            else:
                raise NotImplementedError
        
        
        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()
        #torch.save(model.state_dict(), 'models/'+dataset+'ADVmodel.pth')
        #torch.save(optim.state_dict(), 'models/'+dataset+'ADVoptimizer.pth')
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim


def md_loss_group(x, y_targets,nclass,device):
    #wr loss group
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.ones(x.shape[0]).to(device)
    for y_target in y_targets:
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x_nontargets.shape[1])
        ret+=torch.log(torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1))
    return ret.sum()

def md_loss_group_train(x, y_targets,nclass,device):
    #wr loss group
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.zeros(x.shape[0]).to(device)
    for y_target in y_targets:
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x_nontargets.shape[1])
        ret+=torch.log(torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1))
    ret=torch.clip(ret,-40,0)
    ret=nn.functional.relu(-ret-25)
    
    return ret.sum()


def md_group_train_log(x, y_targets,nclass,device):
    #wr loss group
    x=torch.exp(x)
    s=torch.sum(x,axis=1)
    s=torch.unsqueeze(s, 1).repeat(1,nclass)
    x=x/s
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.zeros(x.shape[0]).to(device)
    x_nontarget_max=torch.max(x_nontargets,1)[0]
    for y_target in y_targets:
        x_target=x[np.arange(x.shape[0]), y_target]
        ret+=nn.functional.relu(x_target+1e-15-x_nontarget_max)
    ret=torch.log(ret+1e-15)
    
    return ret.sum()


def md_group_train(x, y_targets,nclass,device):
    #wr loss group
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets.numpy())]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.zeros(x.shape[0]).to(device)
    x_nontarget_max=torch.max(x_nontargets,1)[0]
    for y_target in y_targets:
        x_target=x[np.arange(x.shape[0]), y_target]
        ret+=nn.functional.relu(x_target+1e-15-x_nontarget_max)
        #ret+=(x_target+1e-15-x_nontarget_max)**2
    return ret.sum()


def md_loss_targeted(x,y_target):
    x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
    return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)


def FreeAdvTrain_epoch2(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf'):
    origins=[0,1,2,3,4,5,6,7,8]
    pairs=[(0,14),(0,15),(0,17),(1,14),(1,15),(1,17),(2,0),(2,14),(2,15),(2,17),(3,0),(3,1),(3,14),(3,15),(3,17),(4,0),(4,1),(4,14),(4,15),(4,17),(5,0),(5,1),(5,14),(5,15),(5,17),(6,0),(6,1),(6,14),(6,15),(6,17),(7,0),(7,1),(7,2),(7,14),(7,15),(7,17),(8,0),(8,1),(8,2),(8,3),(8,14),(8,15),(8,17)]
    targets={}
    for pair in pairs:
        if pair[0] not in targets:
            targets[pair[0]]=list()
        targets[pair[0]].append(pair[1])
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


        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim


def GroupAdvTrain_epoch(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',v=1):
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
    batches=np.ceil(total/batch_size).astype(int)
    #batches=np.floor(total/batch_size).astype(int)
    lookup={}
    c=43
    for pair in pairs:
        lookup[pair]=c
        c+=1

    for i in range(batches):
        
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=x[start_index:end_index]
        y_batch=y[start_index:end_index]
        
        select= torch.logical_not(torch.isin(y_batch, torch.tensor(origins).long().to(device)))
        x_batch=x_batch[select]
        y_batch=y_batch[select]
        
        alllength=torch.isin(y, torch.tensor(origins).long().to(device)).sum().item()
        position=x_batch.shape[0]
        ys=list()
        for origin in origins:
            select= (y==origin)
            x_select=x[select]
            y_select=y[select]
            length=select.sum().item()

            N=np.ceil(128*length/alllength).astype(int)
            start=min(i*N % length, length-N)
            x_batch=torch.cat((x_batch,x_select[start:start+N]),0)
            y_batch=torch.cat((y_batch,y_select[start:start+N]),0)
        #delta_batch=delta_batch[0:end_index-start_index]
        delta_batch = torch.zeros(x_batch.shape).to(device)
        
        for j in range(m):
            model.train()

            optim.zero_grad()
            model.zero_grad()

            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)
            
            adv_batch.requires_grad = True
            output = model(adv_batch)
            loss=F.cross_entropy(output,y_batch)
            #loss+=F.cross_entropy(output[position:],y_batch[position:])*v     
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

            #select= torch.logical_not(torch.isin(y_batch, torch.tensor(origins).long().to(device)))
            delta_batch[:position]=0
        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()
    

        #torch.save(model.state_dict(), 'models/'+dataset+'ADVmodel.pth')
        #torch.save(optim.state_dict(), 'models/'+dataset+'ADVoptimizer.pth')
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim

def GroupAdvTrain_epoch3(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',v=1):
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
    
    alllength=torch.isin(y, torch.tensor(origins).long().to(device)).sum().item()
    batches=np.ceil(alllength/batch_size).astype(int)
    select= (torch.isin(y, torch.tensor(origins).long().to(device)))
    x_select=x[select]
    y_select=y[select]

    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=x_select[start_index:end_index]
        y_batch=y_select[start_index:end_index]
        delta_batch = torch.zeros(x_batch.shape).to(device)

        for j in range(m):
            model.train()
            optim.zero_grad()
            model.zero_grad()

            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)

            
            refetch=np.random.choice(x.shape[0], size=x_batch.shape[0], replace=False)
            x_BAT=x[refetch]
            y_BAT=y[refetch]
            
            adv_batch.requires_grad = True
            optim.zero_grad()
            model.zero_grad()
            output = model(x_BAT)
            loss=F.cross_entropy(output,y_BAT)
            loss.backward()
            optim.step()

            adv_batch.requires_grad = True
            output = model(adv_batch)
            if j<0:
                loss=F.cross_entropy(output,y_batch)
            else:
                loss=0
                for origin in origins:
                    loss+=md_group_train(output[y_batch==origin], torch.tensor(targets[origin]).long(),43,device)
            
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
            grads=adv_batch.grad
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

            #select= torch.logical_not(torch.isin(y_batch, torch.tensor(origins).long().to(device)))

        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()


        #torch.save(model.state_dict(), 'models/'+dataset+'ADVmodel.pth')
        #torch.save(optim.state_dict(), 'models/'+dataset+'ADVoptimizer.pth')
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim




def GroupAdvTrain_epoch2(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',v=1):
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
    #batches=np.ceil(total/batch_size).astype(int)
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

            #N=np.ceil(64*length/alllength).astype(int)
            N=np.ceil(128*length/alllength).astype(int)
            #start=min(i*N % length, length-N)
            start=np.random.randint(length-N)
            x_batch=torch.cat((x_batch,x_select[start:start+N]),0)
            y_batch=torch.cat((y_batch,y_select[start:start+N]),0)
        #delta_batch=delta_batch[0:end_index-start_index]
        delta_batch = torch.zeros(x_batch.shape).to(device)
        for j in range(m):
            model.train()

            optim.zero_grad()
            model.zero_grad()

            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)

            adv_batch.requires_grad = True
            output = model(adv_batch)
            #loss=F.cross_entropy(output,y_batch)
            #loss+=F.cross_entropy(output[position:],y_batch[position:])*v
            loss=F.cross_entropy(output[:position],y_batch[:position])
            l1+=loss.item()
            #print (loss)
            outputp=output[position:]
            y_batchp=y_batch[position:]
            for origin in origins:
                loss+=md_group_train(outputp[y_batchp==origin], torch.tensor(targets[origin]).long(),43,device)*0.1
            #print (loss)
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

            #select= torch.logical_not(torch.isin(y_batch, torch.tensor(origins).long().to(device)))
            delta_batch[:position]=0
        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()


        #torch.save(model.state_dict(), 'models/'+dataset+'ADVmodel.pth')
        #torch.save(optim.state_dict(), 'models/'+dataset+'ADVoptimizer.pth')
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim,l1,l2



def GroupAdvTrain_epoch4(model,optim, x,y,batch_size,iters,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf'):
    origins=[0,1,2,3,4,5,6,7,8]
    pairs=[(0,14),(0,15),(0,17),(1,14),(1,15),(1,17),(2,0),(2,14),(2,15),(2,17),(3,0),(3,1),(3,14),(3,15),(3,17),(4,0),(4,1),(4,14),(4,15),(4,17),(5,0),(5,1),(5,14),(5,15),(5,17),(6,0),(6,1),(6,14),(6,15),(6,17),(7,0),(7,1),(7,2),(7,14),(7,15),(7,17),(8,0),(8,1),(8,2),(8,3),(8,14),(8,15),(8,17)]
    targets={}
    for pair in pairs:
        if pair[0] not in targets:
            targets[pair[0]]=list()
        targets[pair[0]].append(pair[1])
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
        
        isin=torch.isin(y_batch, torch.tensor(origins).long().to(device))
        rands=torch.rand(x_batch.shape[0])<0.1
        rands=rands.to(device)
        for j in range(m):
            #rands=torch.rand(x_batch.shape[0])<0.01
            rands=rands.to(device)
            optim.zero_grad()
            model.zero_grad()
            adv_batch = x_batch + delta_batch
            adv_batch = torch.clamp(adv_batch, 0, 1)
            adv_batch.requires_grad = True
            output = model(adv_batch)
            loss=F.cross_entropy(output,y_batch)

            
            lucky=torch.logical_and(isin,rands)
            loss-=F.cross_entropy(output[lucky],y_batch[lucky])
            output_lucky=output[lucky]
            y_lucky=y_batch[lucky]
            for origin in origins:
                loss-=md_loss_group(output_lucky[y_lucky==origin], torch.tensor(targets[origin]).long(),43,device)
    
            
            
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


        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.flush()
    print (str(iters+1)+' iterations training complete------------------------------')
    return model,optim




def FreeAdvTrain(model,optim, trainx,trainy,testx,testy,batch_size,max_iters=5,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',nclass=10,x_val=None,y_val=None,origins=None,pairs=None):
    accl=list()
    #robustnessl=list()
    #robustnesslPGD=list()
    #acclv=list()
    #robustnesslv=list()
    #robustnesslvPGD=list()
    robustnesslAPGD=list()
    robustnesslAPGDT=list()
    #robustnesslvAPGD=list()
    acclS=list()
    robustnesslAPGDS=list()
    if L_dist=='L_inf':
        norm='Linf'
    else:
        norm='L2'

    robustnesslAPGDSt_best=list()
    robustnesslAPGDSt_worst=list()
    robustnesslAPGDSt_average=list()
    robustnesslAPGDSt_max=list()
    robustnesslAPGDSt_group=list()

    max_perform=0
    performance={}
    x=torch.tensor(trainx).float().to(device)
    y=torch.tensor(trainy).long().to(device)
    record=np.zeros((max_iters,2))
    for i in range(max_iters):
        print (testy.shape)
        model,optim=FreeAdvTrain_epoch(model,optim,x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
        #model,optim=GroupAdvTrain_epoch(model,optim, x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
        model.eval()
        """
        record[i][0]=(test(model,testx,testy,batch_size,device=device))
        targets=list()
        for pair in pairs:
            if pair[1] not in targets:
                targets.append(pair[1])
        print (targets)

        test_select=np.isin(testy,targets)
        x_t, y_t=testx[test_select], testy[test_select]

        record[i][1]=(test(model,x_t,y_t,batch_size,device=device))
        """
        #test_select=np.isin(testy,origins)
        #x_t, y_t=testx[test_select], testy[test_select]

        #record[i][1]=(test(model,x_t,y_t,batch_size,device=device))
        #record[i][2]=1-autopgd(model,testx,testy,0,eps=eps,device='cuda',bs=512,norm=norm)
        #record[i][3]=1-autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=512,norm=norm)
        #record[i][4]=1-APGD_targeted(model,testx,testy,0,nclass,eps,norm,bs=512)
        #record[i][5]=1-APGD_targeted(model,x_t,y_t,0,nclass,eps,norm,bs=512)
        #with open("others2F512",'wb') as f:
        #    pickle.dump(record,f)
        
        
        
        
        """
        record[i][2]=(test(model,testx,testy,batch_size,device=device))
        test_select=np.isin(testy,origins)
        x_t, y_t=testx[test_select], testy[test_select]
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        record[i][3]= (1-best)
        record[i][0]=test(model,trainx,trainy,batch_size,device=device)
        test_select=np.isin(trainy,origins)
        x_t, y_t=trainx[test_select], trainy[test_select]
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        record[i][1]=1-best
        with open("recordF512",'wb') as f:
            pickle.dump(record,f)
        #acc=test(model,testx,testy,batch_size,device=device)
        #print (acc)
        
        adv_examples=pgd_attack(model, testx,testy, eps=eps, alpha=eps, iters=1,device=device,L_dist=L_dist)#fgsm
        robustness=test(model,adv_examples,testy,batch_size,device=device)
        print ('benign accuracy: '+str(acc)+' robustness: '+str(robustness))
        accl.append(acc)
        robustnessl.append(robustness)
        print (testy.shape,y_val.shape)
        #pgd robustness
        adv_examples=pgd_attack(model, testx,testy, eps=eps, alpha=1./255, iters=100,device=device,L_dist=L_dist)
        robustness=test(model,adv_examples,testy,batch_size,device=device)
        robustnesslPGD.append(robustness)
        acc=test(model,x_val,y_val,batch_size,device=device)
        acclv.append(acc)
        adv_examples=pgd_attack(model,x_val ,y_val, eps=eps, alpha=eps, iters=1,device=device,L_dist=L_dist)
        robustness=test(model,adv_examples,y_val,batch_size,device=device)
        robustnesslv.append(robustness)
        adv_examples=pgd_attack(model, x_val,y_val, eps=eps, alpha=1./255, iters=100,device=device,L_dist=L_dist)
        robustness=test(model,adv_examples,y_val,batch_size,device=device)
        robustnesslvPGD.append(robustness)


        print (testy.shape,y_val.shape)
        robustnesslAPGD.append(autopgd(model,testx,testy,0,eps=eps,device='cuda',bs=512))
        robustnesslvAPGD.append(autopgd(model,x_val,y_val,0,eps=eps,device='cuda',bs=512))
        print (testy.shape,y_val.shape)
        
        if max_perform < acc + robustness:
            max_perform = acc + robustness
            torch.save(model.state_dict(), 'models/'+dataset+'-ADVmodel1L_2.pth')
            torch.save(optim.state_dict(), 'models/'+dataset+'-ADVoptimizer1L_2.pth')
    
        with open('models/'+dataset+'-performance7L_inf-1B','wb') as f:
            performance['acc']=accl
            performance['robustness']=robustnessl
            
            #additional data
            performance['robustnessPGD']=robustnesslPGD
            performance['accv']=acclv
            performance['robustnessv']=robustnesslv
            performance['robustnessPGDv']=robustnesslvPGD
            performance['robustnessAPGD']=robustnesslAPGD
            performance['robustnessAPGDv']=robustnesslvAPGD        
            pickle.dump(performance,f)
        
        st=time.time()
        robustness=autopgd(model,testx[:512],testy[:512],0,eps=eps,device='cuda',bs=512)
        print (time.time()-st)
         
        accl.append(test(model,testx,testy,batch_size,device=device))
        
        robustnesslAPGD.append(autopgd(model,testx,testy,0,eps=eps,device='cuda',bs=batch_size,norm=norm))
        test_select=np.isin(testy,origins)
        x_t, y_t=testx[test_select], testy[test_select]
        acclS.append(test(model,x_t,y_t,batch_size,device='cuda'))
        robustnesslAPGDS.append(autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=batch_size,norm=norm))
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_max',norm=norm)
        succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_max.append(1-succ)
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_group',norm=norm)
        succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_group.append(1-succ)
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_average.append(1-succ)
        robustnesslAPGDSt_best.append(1-best)
        robustnesslAPGDSt_worst.append(1-worst)
        
        robustnesslAPGDT.append(1-APGD_targeted(model,testx,testy,0,nclass,eps,norm,bs=batch_size))
        with open('models/catch2/'+dataset+'-performance4'+L_dist+'-1C','wb') as f:
            performance['robustnessAPGDT']=robustnesslAPGDT
            
            performance['acc']=accl
            
            performance['robustnessAPGD']=robustnesslAPGD
            performance['accS']=acclS
            performance['robustnessAPGDS']=robustnesslAPGDS
            performance['robustnessAPGDSt_max']=robustnesslAPGDSt_max
            performance['robustnessAPGDSt_group']=robustnesslAPGDSt_group
            performance['robustnessAPGDSt_average']=robustnesslAPGDSt_average
            performance['robustnessAPGDSt_best']=robustnesslAPGDSt_best
            performance['robustnessAPGDSt_worst']=robustnesslAPGDSt_worst
            
            pickle.dump(performance,f)
            
        """
        """
        if ((robustness>=0.455) and (robustness<=0.465) and (acc>=0.81) and (acc<=0.82)):
            #if ((robustness>=0.32) and (robustness<=0.33) and (acc>=0.67) and (acc<=0.68)):
            torch.save(model.state_dict(), 'models/catch/'+dataset+'-ADVmodel7L_inf-2A'+str(i)+'.pth')
            #torch.save(optim.state_dict(), 'models/catch/'+dataset+'-ADVoptimizer3L_inf-1A'+str(i)+'.pth')
        """

    return model


def NormalTrain(model,optim, trainx,trainy,testx,testy,batch_size,max_iters=5,eps=16/255,m=8,device='cpu',dataset="mnist",L_dist='L_inf',nclass=10,x_val=None,y_val=None,origins=None,pairs=None):
    performance={}
    accl=list()
    robustnesslAPGD=list()
    robustnesslAPGDT=list()
    acclS=list()
    robustnesslAPGDS=list()
    if L_dist=='L_inf':
        norm='Linf'
    else:
        norm='L2'
    robustnesslAPGDSt_best=list()
    robustnesslAPGDSt_worst=list()
    robustnesslAPGDSt_average=list()
    robustnesslAPGDSt_max=list()
    robustnesslAPGDSt_group=list()
    x=torch.tensor(trainx).float().to(device)
    y=torch.tensor(trainy).long().to(device)
    v=0
    best_rob=0
    c=0
    loss=np.zeros((max_iters,2))
    record=np.zeros((max_iters,8))
    for i in range(max_iters):
        model,optim,loss[i][0],loss[i][1]=GroupAdvTrain_epoch2(model,optim, x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
        model.eval()
        #with open("lossSDC384m8x3-e2log",'wb') as f:
        #    pickle.dump(loss,f)
    
    
        print (testy.shape)
        #model=FreeAdvTrain_epoch(model,optim,x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
        #model=train_epoch(model,optim,x,y,batch_size,device=device,dataset=dataset)
        
        #model,optim=GroupAdvTrain_epoch2(model,optim, x,y,batch_size,i,eps=eps,m=m,device=device,dataset=dataset,L_dist=L_dist)
        
        #model.eval()
        
        record[i][0]=(test(model,testx,testy,batch_size,device=device))
        targets=list()
        for pair in pairs:
            if pair[1] not in targets:
                targets.append(pair[1])
        print (targets)
        
        test_select=np.isin(testy,origins)
        x_t, y_t=testx[test_select], testy[test_select]
        
        record[i][1]=(test(model,x_t,y_t,batch_size,device=device))
        record[i][2]=1-autopgd(model,testx,testy,0,eps=eps,device='cuda',bs=512,norm=norm)
        record[i][3]=1-autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=512,norm=norm)
        record[i][4]=1-APGD_targeted(model,testx,testy,0,nclass,eps,norm,bs=512)
        record[i][5]=1-APGD_targeted(model,x_t,y_t,0,nclass,eps,norm,bs=512)
        
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        record[i][6]= (1-best)
        test_select=np.isin(testy,targets)
        x_t, y_t=testx[test_select], testy[test_select]
        record[i][7]=(test(model,x_t,y_t,batch_size,device=device))
        """
        
        record[i][0]=test(model,trainx,trainy,batch_size,device=device)
        test_select=np.isin(trainy,origins)
        x_t, y_t=trainx[test_select], trainy[test_select]
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        record[i][1]=1-best
        """
        
        
        with open("50+SDC384m6x0.1",'wb') as f:
            pickle.dump(record,f)
    
    """
        #torch.save(model.state_dict(), norm+"NGROUP3-3")
        
        print (test(model,x_t,y_t,batch_size,device='cuda'))
        print(autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=512,norm=norm))
        
        train_select=np.isin(trainy,origins)
        x_t, y_t=trainx[train_select], trainy[train_select]
        rob=autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=512,norm=norm)
        
        
        if rob>best_rob:
            c=0
            best_rob=rob
            torch.save(model.state_dict(), norm+"NGROUP7")
        else:
            v*=4
            c+=1
        if (c>1):
            break
        
        
               
        accl.append(test(model,testx,testy,batch_size,device=device))

        robustnesslAPGD.append(autopgd(model,testx,testy,0,eps=eps,device='cuda',bs=batch_size,norm=norm))
        test_select=np.isin(testy,origins)
        x_t, y_t=testx[test_select], testy[test_select]
        acclS.append(test(model,x_t,y_t,batch_size,device='cuda'))
        robustnesslAPGDS.append(autopgd(model,x_t,y_t,0,eps=eps,device='cuda',bs=batch_size,norm=norm))
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_max',norm=norm)
        succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_max.append(1-succ)
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_group',norm=norm)
        succ=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_group.append(1-succ)
        apgd=PGD(model,nclass=nclass,eps=eps,loss='md_single',norm=norm)
        succ,best,worst=APGD_caller(apgd,pairs,x_t, y_t,bs=batch_size,auto=True)
        robustnesslAPGDSt_average.append(1-succ)
        robustnesslAPGDSt_best.append(1-best)
        robustnesslAPGDSt_worst.append(1-worst)
        
        robustnesslAPGDT.append(1-APGD_targeted(model,testx,testy,0,nclass,eps,norm,bs=batch_size))
        with open('models/catch2/'+dataset+'-performance6'+L_dist+'-1D','wb') as f:
            performance['robustnessAPGDT']=robustnesslAPGDT

            performance['acc']=accl

            performance['robustnessAPGD']=robustnesslAPGD
            performance['accS']=acclS
            performance['robustnessAPGDS']=robustnesslAPGDS
            performance['robustnessAPGDSt_max']=robustnesslAPGDSt_max
            performance['robustnessAPGDSt_group']=robustnesslAPGDSt_group
            performance['robustnessAPGDSt_average']=robustnesslAPGDSt_average
            performance['robustnessAPGDSt_best']=robustnesslAPGDSt_best
            performance['robustnessAPGDSt_worst']=robustnesslAPGDSt_worst

            pickle.dump(performance,f)
    """
    return model


