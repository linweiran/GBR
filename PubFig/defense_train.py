# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
We use this file to train the clean model to classify the images.
We do not tune a lot of hyperparmeters, all default hyperparameters showed below.
We obtain more than 98% of accuracy
type 'python origin_train.py', note that we do not provide the VGG_FACE.t7, but you can download the model through  
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz move it to glass/experiment file
'''
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy 
import torch.nn.functional as F
import torchfile
import pickle
import numpy as np
from new_vgg_face import VGG_16
import sys
#from save_image import save_image 
#uncomment to show images 

def k_search_helper(a,i,t,l):

    if len(set(l)) != len(l):
        return False
    if t==0:
        return True
    if t>a.shape[0]-i:
        return False
    if k_search_helper(a,i+1,t,l):
        return True
    js=np.where(a[i]>0)[0].tolist()
    for j in js:
        if k_search_helper(a,i+1,t-1,l+[j]):
            return True
    return False

def k_search(a,t):
    if (a.shape[0]>a.shape[1]):
        a=np.swapaxes(a,0,1)
    return k_search_helper(a,0,t,[])


def md_group_train(x, y_targets,nclass,device):
    #wr loss group
    a=torch.arange(nclass)
    b=np.arange(nclass)
    y_nontargets=a[np.delete(b,y_targets)]
    x_nontargets=torch.index_select(x,1,y_nontargets.to(device))
    ret=torch.zeros(x.shape[0]).to(device)
    x_nontarget_max=torch.max(x_nontargets,1)[0]
    for y_target in y_targets:
        x_target=x[np.arange(x.shape[0]), y_target]
        ret+=nn.functional.relu(x_target+1e-15-x_nontarget_max)
    return ret.sum()


def md_loss_targeted(x, y_target):
    x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
    return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)

def choose_color_targeted(model,X,t,glass,mean):
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    min_loss = (torch.ones(X.shape[0])*1000000).to(X.device)
    min_delta = torch.zeros_like(X)

    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(X.device)


        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        all_loss = md_loss_targeted(model(X+delta1-mean),t)
        min_delta[all_loss < min_loss] = delta1.detach()[all_loss < min_loss]
        min_loss = torch.min(min_loss, all_loss)

    return min_delta

def glass_attack_targeted(model, X, t, alpha=20, num_iter=300,momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""
    glass=torchvision.io.read_image('silhouette.png').float()
    glass[glass>0.5]=1
    glass[glass<0.5]=0
    device1 = torch.device("cuda")
    glass = glass.to(device1)
    X=X.to(device1)
    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    mean = mean.to(device1)
    X1 = (X+mean)*(1-glass)

    with torch.set_grad_enabled(False):
        color_glass = choose_color_targeted(model,X1,t,glass,mean)
    with torch.set_grad_enabled(True):
        X1= X1+color_glass-mean


        for i in range(num_iter):
            X1.requires_grad=True
            loss = md_loss_targeted(model(X1), t)
            #print (loss[:10].cpu().detach())
            grad=torch.autograd.grad(loss.sum(), [X1])[0].detach()
            X1.requires_grad=False
            delta_change =  grad*glass
            max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
            r = alpha * delta_change /max_val[:,None,None,None]
            r[r.isnan()]=0
            if i == 0:
                delta = r
            else:
                delta = momentum * delta.detach() + r

            delta[(-delta.detach() + X1.detach() + mean) >255] = 0
            delta[(-delta.detach() + X1.detach() + mean) < 0 ] = 0
            X2 = (X1.detach() - delta.detach())
            X1[loss>0]=X2[loss>0]
            X1 = torch.round(X1.detach()+mean) - mean

    with torch.no_grad():
        Xadv = (X1).detach()
        model.eval()

        outputs = model(Xadv)

        _, predicted = torch.max(outputs.data, 1)
    return (predicted == t).detach().cpu().numpy()


def md_loss_group( x, y_targets):
    a=torch.arange(60)
    b=np.arange(60)
    y_nontargets=a[np.delete(b,y_targets)]
    x_nontargets=torch.index_select(x,1,y_nontargets.to('cuda'))
    ret=torch.ones(x.shape[0]).to('cuda')
    for y_target in y_targets:
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x_nontargets.shape[1])
        ret+=torch.log(torch.sum(nn.functional.relu(x_nontargets+1e-15-x_target),dim=1))
    return ret

def choose_color_group(model,X,t,glass,mean):
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    min_loss = (torch.ones(X.shape[0])*1000000).to(X.device)
    min_delta = torch.zeros_like(X)


    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(X.device)


        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        all_loss = md_loss_group(model(X+delta1-mean),t)
        min_delta[all_loss < min_loss] = delta1.detach()[all_loss < min_loss]
        min_loss = torch.min(min_loss, all_loss)

    return min_delta
def glass_attack_group(model, X, t, glass, alpha=20, num_iter=300,momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""
    """
    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    X1 = (X+mean)*(1-glass)
    """
    """ Construct glass frame adversarial examples on the examples X"""
    glass=torchvision.io.read_image('silhouette.png').float()
    glass[glass>0.5]=1
    glass[glass<0.5]=0
    device1 = torch.device("cuda")
    glass = glass.to(device1)
    X=X.to(device1)
    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    mean = mean.to(device1)
    X1 = (X+mean)*(1-glass)

    with torch.set_grad_enabled(False):
        color_glass = choose_color_group(model,X1,t,glass,mean)
    with torch.set_grad_enabled(True):
        X1= X1+color_glass-mean
        for i in range(num_iter):
            X1.requires_grad=True
            loss = md_loss_group(model(X1), t)
            #print (loss[:10].cpu().detach())
            grad=torch.autograd.grad(loss.sum(), [X1])[0].detach()
            X1.requires_grad=False
            delta_change =  grad*glass
            max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
            r = alpha * delta_change /max_val[:,None,None,None]
            r[r.isnan()]=0
            if i == 0:
                delta = r
            else:
                delta = momentum * delta.detach() + r

            delta[(-delta.detach() + X1.detach() + mean) >255] = 0
            delta[(-delta.detach() + X1.detach() + mean) < 0 ] = 0
            X2 = (X1.detach() - delta.detach())
            X1[loss>0]=X2[loss>0]
            X1 = torch.round(X1.detach()+mean) - mean
            #X1.zero_grad()
        return (X1).detach()

# process the data, can use random crop and RandomHorizontalFlip()
def data_process(batch_size=64):
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (224,224)), 
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #do not have transforms before
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    }
                                
    #data_dir = '../Data'   # change this if the data is in different loaction 
    #data_dir='done/balaned_data_set_36'
    data_dir='done/dev_set_60' 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'val','test']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes

    print(class_names)
    print(dataset_sizes)
    return dataloaders,dataset_sizes



def train_model(model,x_train,y_train,x_test,y_test,perm,optimizer, scheduler, num_epochs=2):
    since = time.time()
    recordl=list()
    glass=torchvision.io.read_image('silhouette.png').float()
    glass[glass>0.5]=1
    glass[glass<0.5]=0
    device1 = torch.device("cuda")
    glass = glass.to(device1)
    criterion = nn.CrossEntropyLoss()
    N=5
    for epoch in range(num_epochs):
        rep=torch.randperm(x_train.shape[0])
        x_train=x_train[rep]
        y_train=y_train[rep]
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #scheduler.step()
        #running_loss = 0.0
        #select_train=torch.isin(y_train,torch.tensor(perm[:5]))
        # Iterate over data.

        """
        model.eval()   # Set model to evaluate mode
        batch_size=64
        batches=np.ceil(5437/batch_size).astype(int)
        acc=0
        for i in range(batches):
            start_index=i*batch_size
            end_index=np.minimum((i+1)*batch_size,5437)
            inputs=x_train[start_index:end_index]
            labels=y_train[start_index:end_index]
            inputs = inputs.to(device1)
            labels = labels.long().to(device1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            acc+=torch.sum(preds==labels).item()/5437
        acc1=acc
        batches=np.ceil(744/batch_size).astype(int)
        acc=0
        for i in range(batches):
            start_index=i*batch_size
            end_index=np.minimum((i+1)*batch_size,744)
            inputs=x_test[start_index:end_index]
            labels=y_test[start_index:end_index]
            inputs = inputs.to(device1)
            labels = labels.long().to(device1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            acc+=torch.sum(preds==labels).item()/744
        acc2=acc
        print (acc1,acc2)
        """
        
        """
        if epoch==0:
            #from free adv training
            tags=np.array([0,1,4])
        else:
            c0,c1,caters=eval_model(model,x_train,y_train,perm)
            tags=np.sum(caters,axis=1).argsort()[:3]
        tags=perm[tags+55]
        """
        tags=perm[60-N:]
        #tags=tags[[1,3,4]]
        #tags=perm[[56,59]]

        select_train=torch.isin(y_train,torch.tensor(perm[:N]))
        select_test=torch.isin(y_test,torch.tensor(perm[:N]))
        x_train_select=x_train[select_train]
        y_train_select=y_train[select_train]
        
        l=list()
        for i in range(N):
            l.append(np.arange(x_train_select.shape[0])[(y_train_select==perm[i]).numpy()])
            print (len(l[i]))

        
        batch_size=64
        batches=np.ceil(x_train.shape[0]/batch_size).astype(int)
        
        choices=np.zeros((batches,N))
        for i in range(N):
            choices[:,i]=np.random.choice(len(l[i]),batches)
            
        
        
        for i in range(batches):
            sys.stdout.write("ADV training "+str(i+1)+'/'+str(batches)+" batches \r")
            sys.stdout.flush()
            start_index=i*batch_size
            end_index=np.minimum((i+1)*batch_size,x_train.shape[0])
            inputs=x_train[start_index:end_index]
            labels=y_train[start_index:end_index]

            
            model.eval()
            attacks=glass_attack_group(model, x_train_select[choices[i]], tags, 20 ,300)            
            

            inputs = inputs.to(device1)
            labels = labels.long().to(device1)
            # zero the parameter gradients
            optimizer.zero_grad()
            model.train()
            # forward
            # track history if only in train
            inputs.requires_grad = True    
            outputs = model(inputs)
            loss = criterion(outputs[:batch_size], labels)
            attacks.requires_grad = True
            outputs2=model(attacks)
            loss+= md_group_train(outputs2,tags,60,device1)*.045

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()   # Set model to evaluate mode
        batch_size=64
        length=x_train.shape[0]
        batches=np.ceil(length/batch_size).astype(int)
        acc=0
        for i in range(batches):
            start_index=i*batch_size
            end_index=np.minimum((i+1)*batch_size,length)
            inputs=x_train[start_index:end_index]
            labels=y_train[start_index:end_index]
            inputs = inputs.to(device1)
            labels = labels.long().to(device1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            acc+=torch.sum(preds==labels).item()/length
        acc1=acc
        length=x_test.shape[0]
        batches=np.ceil(length/batch_size).astype(int)
        acc=0
        for i in range(batches):
            start_index=i*batch_size
            end_index=np.minimum((i+1)*batch_size,length)
            inputs=x_test[start_index:end_index]
            labels=y_test[start_index:end_index]
            inputs = inputs.to(device1)
            labels = labels.long().to(device1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            acc+=torch.sum(preds==labels).item()/length
        acc2=acc
        
        print (acc,acc2)
        #c0,c1,caters=eval_model(model,x_test,y_test,perm)
        #print (acc1,acc2,c0,c1,caters)
        #recordl.append((acc1,acc2,c0,c1,caters))
        #with open("Gtrainl-Bx.14R-134-T5-decay0.9-lr.003",'wb') as f:
        #    pickle.dump(recordl,f)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def eval_model(model,x_test,y_test,perm):
    N=5
    x_test_k=x_test[torch.isin(y_test,torch.tensor(perm[:N]))]
    y_test_k=y_test[torch.isin(y_test,torch.tensor(perm[:N]))]
    l=list()
    for i in range(N):
        l.append(np.arange(x_test_k.shape[0])[(y_test_k==perm[i]).numpy()])
    choices=np.zeros((10000,N))
    for i in range(N):
        choices[:,i]=np.random.choice(len(l[i]),10000)
    print (torch.sum(torch.isin(y_test,torch.tensor(perm[:N]))))
    #print (torch.sum(torch.isin(y_train,torch.tensor(perm[:5]))))

    #x_test_k=x_test[torch.isin(y_test,torch.tensor(perm[:5]))]
    #y_test_k=y_test[torch.isin(y_test,torch.tensor(perm[:5]))]
    #x_train_k=x_train[torch.isin(y_train,torch.tensor(perm[:5]))]
    #y_train_k=y_train[torch.isin(y_train,torch.tensor(perm[:5]))]

    results=np.zeros((N,x_test_k.shape[0]))
    caters=np.zeros((N,N))
    for i in range(N):
        results[i]=glass_attack_targeted(model, x_test_k.to(device), perm[60-N+i])
        for j in range(N):
            select=torch.isin(y_test_k,torch.tensor(perm[j])).numpy()
            result_select=results[i]
            result_select=result_select[select]
            caters[i][j]=(result_select.sum())/(result_select.shape[0])
    c0=0
    c1=0
    for i in range(10000):
        choice=choices[i].astype(dtype='int64')
        result_part=results[:,choice]
        c0+= (np.sum(np.sum(result_part,axis=1)>0)>=N-2)
        c1+= k_search(result_part,N-2)
    return c0,c1,caters

if __name__ == "__main__":
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.random.manual_seed(12345)
    torch.random.manual_seed(12345)
    torch.manual_seed(12345)
    
    dataloaders,dataset_sizes =data_process(batch_size =64)
    device = torch.device("cuda")
    model_ft = VGG_16()
    #model_ft.load_state_dict(torch.load('../donemodel/neo_original_model2.pt'))
    #model_ft.load_state_dict(torch.load('../donemodel/neo_final_model.pt'))
    model_ft.load_state_dict(torch.load('../donemodel/new_sticker_model010015.pt'))
    #model_ft.load_weights()
    #model_ft=nn.DataParallel(model_ft,device_ids=[0,1]) 
    #if you want to use more gpus, other files may change if you use mutliple gpus
    with open("perm4",'rb') as f:
        perm=pickle.load(f)
        print (perm)

 


    print (len(dataloaders['train']))
    total=0
    x_train=torch.zeros((5437,3,224,224))
    y_train=torch.zeros(5437)
    for data in dataloaders['train']:
            images, labels = data
            images = images[:,[2,1,0],:,:] #rgb to bgr
            bs=labels.size(0)
            x_train[total:total+bs]=images
            y_train[total:total+bs]=labels
            total+=bs
            print (total)
    total=0
    x_test=torch.zeros((744,3,224,224))
    y_test=torch.zeros(744)
    for data in dataloaders['test']:
            images, labels = data
            images = images[:,[2,1,0],:,:] #rgb to bgr
            bs=labels.size(0)
            x_test[total:total+bs]=images
            y_test[total:total+bs]=labels
            total+=bs
            print (total)
    total=0
    x_val=torch.zeros((1517,3,224,224))
    y_val=torch.zeros(1517)
    for data in dataloaders['val']:
            images, labels = data
            images = images[:,[2,1,0],:,:] #rgb to bgr
            bs=labels.size(0)
            x_val[total:total+bs]=images
            y_val[total:total+bs]=labels
            total+=bs
            print (total)
    

    #x_train=torch.cat((x_train,x_val))
    #y_train=torch.cat((y_train,y_val))
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    #print(eval_model(model_ft,x_test,y_test,perm))

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.8)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0011, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.003, momentum=0.9)
    # Decay LR by a factor of 0.1 every 10 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.9)
    
    
    model_ft = train_model(model_ft,x_train,y_train,x_test,y_test, perm,optimizer_ft, exp_lr_scheduler,num_epochs=1)
    
    print("..............3...............")
    
    torch.save(model_ft.state_dict(), '../donemodel/new_defense_model-54-4.pt')  
    
    
