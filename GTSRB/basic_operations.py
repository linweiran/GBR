import numpy as np
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import torch.optim as optim
def train_epoch(model,optimizer,x,y,batch_size,device='cpu',dataset="mnist",transfer=False):
    if (not transfer):
        model.train()
    
    model.to(device)
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    loss_sum=0 
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=x[start_index:end_index]
        y_batch=y[start_index:end_index]
        optimizer.zero_grad()
        output = model(x_batch)
        loss=F.cross_entropy(output,y_batch)
        loss_sum+=loss.item()
        #loss=F.cross_entropy(output[0],y_batch)
        loss.backward()
        optimizer.step()
        #print (str(i+1)+'/'+str(batches)+" batches complete         \r")
        sys.stdout.write(str(i+1)+'/'+str(batches)+" batches complete       \r")
        sys.stdout.flush()
    #torch.save(model.state_dict(), 'models/'+dataset+'-model.pth')
    #torch.save(optimizer.state_dict(), 'models/'+dataset+'-optimizer.pth')
    print ("training loss: "+str(loss_sum))
    return model

def train(model,optimizer,x,y,batch_size,iters,valx=None,valy=None,device='cpu',dataset="mnist"):
    maxacc=0
    if valx is None:
        valx=x
        valy=y
    
    for i in range(iters):
        model=train_epoch(model,optimizer,torch.tensor(x).float().to(device),torch.tensor(y).long().to(device),batch_size,device=device,dataset=dataset)
        print ('training complete for '+str(i+1)+' iterations------------------------------')
        #acc=test(model, valx,valy,batch_size,device=device)
        #print ('test accuracy: '+str(acc))
        #if acc>maxacc:
        #    maxacc=acc
        #    torch.save(model.state_dict(), 'models/'+dataset+'-model.pth')
        #    torch.save(optimizer.state_dict(), 'models/'+dataset+'-optimizer.pth')
    return model

def test(model, x,y,batch_size,device='cpu'):
    model.eval()
    model.to(device)
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    success=0
    loss=0
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=torch.tensor(x[start_index:end_index]).float().to(device)
        y_batch=torch.tensor(y[start_index:end_index]).long().to(device)
        output=model(x_batch)
        pred=torch.argmax(output,dim=1)
        loss+=F.cross_entropy(output,y_batch).item()
        #print (pred.cpu().numpy()[:20])
        #print (y_batch.cpu().numpy()[:20])
        success+=(pred==y_batch).sum().item()
    print ("accuracy: "+str(success/total))
    #print ("loss: "+str(loss))
    return success/total
    #print ("accuracy: "+str(success/total))

def load_model(model,optimizer,dataset="mnist"):
    model.load_state_dict(torch.load('models/'+dataset+'-model2.pth'))
    optimizer.load_state_dict(torch.load('models/'+dataset+'-optimizer2.pth'))
    model.eval()
    return model,optimizer

def load_adv_model(model,optimizer,dataset="mnist"):
    model.load_state_dict(torch.load('models/'+dataset+'-ADVmodel2L_inf.pth'))
    optimizer.load_state_dict(torch.load('models/'+dataset+'-ADVoptimizer2L_inf.pth'))
    model.eval()
    return model,optimizer


