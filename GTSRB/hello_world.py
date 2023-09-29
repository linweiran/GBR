import torch
import torchvision
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_dataset import load_gtsrb
import pickle
import copy
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str, default='/data/GTSRB')

args = parser.parse_args()
print("ARGS: ", args)
MAIN_DIR=args.data_path

(x_train, y_train), (x_test, y_test), (x_val, y_val)=load_gtsrb(data_path=MAIN_DIR)
print ("data loaded sucessfullly!")


x_train=np.swapaxes(x_train, 1, 3)
x_test=np.swapaxes(x_test, 1, 3)
x_val=np.swapaxes(x_val, 1, 3)


random_seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)
np.random.seed(random_seed)

from basic_net import GTSRBNet2,GTSRBNet3,GTSRBNet4,GTSRBNet6,GTSRBNet7
from basic_operations import train,test
from attacks import pgd_attack,autopgd, APGD_caller,PGD,APGD_targeted,PGD
from defenses import FreeAdvTrain,GroupTrain

origins=[0,1,2,3,4,5,6,7,8]
pairs=[(0,14),(0,15),(0,17),(1,14),(1,15),(1,17),(2,0),(2,14),(2,15),(2,17),(3,0),(3,1),(3,14),(3,15),(3,17),(4,0),(4,1),(4,14),(4,15),(4,17),(5,0),(5,1),(5,14),(5,15),(5,17),(6,0),(6,1),(6,14),(6,15),(6,17),(7,0),(7,1),(7,2),(7,14),(7,15),(7,17),(8,0),(8,1),(8,2),(8,3),(8,14),(8,15),(8,17)]

select=y_test<9
x_select=x_test[select]
y_select=y_test[select]


#adv training
ADVnetwork = GTSRBNet4()
ADVoptimizer=optim.Adam(ADVnetwork.parameters())
print ("start adversarial training...")
FreeAdvTrain(ADVnetwork,ADVoptimizer, x_train, y_train,x_val,y_val,512,max_iters=7,eps=8/255,m=2,device='cpu',dataset='GTSRB')
ADVnetwork.eval()

x_adv=pgd_attack(ADVnetwork, x_test,y_test, batch_size=1024,eps=8/255, alpha=8/255, iters=1,device='cpu',L_dist='L_inf')
lresults=[]

print ("Reproducing results in Sec 2 (Metrics) ...")
acc=test(ADVnetwork,x_test,y_test,512,device='cpu')
print ("Benign accuracy of the model is: "+format(acc, ".4f"))
lresults.append(acc)
acc=test(ADVnetwork,x_adv,y_test,512,device='cpu')
print ("Untargeted robustness  of the model is: "+format(acc, ".4f"))


acc=APGD_targeted(ADVnetwork,x_test,y_test,0,43,8/255,"Linf",device="cpu",n_iter=1)
print ("Average targeted robustness  of the model is: "+format(1-acc, ".4f"))
acc=test(ADVnetwork,x_select,y_select,512,device='cpu')
lresults.append(acc)


apgd=PGD(ADVnetwork,nclass=43,eps=8/255,loss='md_single',norm="Linf",device="cpu",n_iter=1)
succ,best,worst=APGD_caller(apgd,pairs,x_test,y_test,bs=512,auto=True)
print ("Group-based robustness of the model is: "+format(1-best, ".4f"))
lresults.append(1-best)
print ("We expect Group-based robustness to be orthogonal to untargeted robustness, average targeted robustness, or benign accuracy.")
print ("Here the Group-based robustness is different from the other three metrics.")


print ("Reproducing results in Sec 3.1 (Loss functions) ...")
print ("Best guess attacks success rate: "+format(best, ".4f"))
print ("Average guess attacks success rate: "+format(succ, ".4f"))
print ("Worst guess attacks success rate: "+format(worst, ".4f"))
apgd=PGD(ADVnetwork,nclass=43,eps=8/255,loss='md_max',norm="Linf",device="cpu",n_iter=1)
succ=APGD_caller(apgd,pairs,x_test,y_test,bs=512,auto=True)
print ("Attacks with MDMAX loss success rate: "+format(succ, ".4f"))
apgd=PGD(ADVnetwork,nclass=43,eps=8/255,loss='md_group',norm="Linf",device="cpu",n_iter=1)
succ=APGD_caller(apgd,pairs,x_test,y_test,bs=512,auto=True)
print ("Attacks with MDMUL loss success rate: "+format(succ, ".4f"))
print ("Attacks with MDMAX or MDMUL loss may find fewer numbers of evasive examples as the best guess attacks, but we expect them to always find more evasive examples than the average or worst guess attacks.")

print ("Reproducing results in Sec 3.2 (Attack strategies) ...")
print ("We did not evaluate strategies on the GTSRB dataset in the main paper.")
print ("Here the setup is that attackers are perturbing speed limits no less than 70 (five classes) as speed limits no higher than 60 (four classes).")
print ("Attackers sample one image from each of the five higher-speed classes, in total five images as a set.")
print ("For each set of five images, attackers can only claim success if they can manipulate these images as all of the lower-speed four classes. They may manipulate the same image as different signs.")
print ("We use with the worst-performing strategy that we proposed (Estimate by Computing a Prior from a Validation Set).")
select=np.logical_and(y_val<9,y_val>3)
x_val_select=x_val[select]
y_val_select=y_val[select]
select=np.logical_and(y_test<9,y_test>3)
x_test_select=x_test[select]
y_test_select=y_test[select]

val_attempts=np.zeros((4,x_val_select.shape[0]))
test_attempts=np.zeros((4,x_test_select.shape[0]))

for i in range(4):  
	val_attempts[i]=APGD_targeted(ADVnetwork,x_val_select,y_val_select,0,43,8/255,"Linf",device="cpu",n_iter=1,array_flag=True,target=i)
	test_attempts[i]=APGD_targeted(ADVnetwork,x_test_select,y_test_select,0,43,8/255,"Linf",device="cpu",n_iter=1,array_flag=True,target=i)

val_dict=np.zeros((4,5))
for i in range(4):
	for j in range(5):
		select=(y_val_select==j+3)
		val_dict[i][j]=val_attempts[i][select].mean()

counts_without_strategy=0
counts_with_strategy=0
total_attempts=0
ATTEMPTS=1000
ids=np.random.rand(ATTEMPTS,5)
for i in range(ATTEMPTS):
	indices=np.zeros(5,dtype=int)
	
	for j in range(5):
		indices[j]=np.floor(ids[i][j]*np.sum(y_test_select==j+3))
	test_attempt_select=test_attempts[:,indices]
	if (np.sum(np.sum(test_attempt_select, axis=1)>0)==4):
		test_attempt_select=test_attempt_select.flatten()
		strat_select=val_dict.flatten().argsort()
		no_strat_select=np.random.permutation(20)
		succeed=np.zeros(4)
		counter=0
		while (np.sum(succeed)<4):
			if succeed[strat_select[counter]//5]==0:
				succeed[strat_select[counter]//5]=test_attempt_select[strat_select[counter]]
				counts_with_strategy+=1
			counter+=1
		succeed=np.zeros(4)
		counter=0
		while (np.sum(succeed)<4):
			if succeed[no_strat_select[counter]//5]==0:
				succeed[no_strat_select[counter]//5]=test_attempt_select[no_strat_select[counter]]
				counts_without_strategy+=1
			counter+=1
		total_attempts+=1
print ("Among " +str(ATTEMPTS) +" that we sampled, attackers may ultimately succeed on "+str(total_attempts)+" sets.")
print ("Without using strategies, attackers need to attempt "+ str(counts_without_strategy) +" attacks.")
print ("Using strategies, attackers need to attempt "+ str(counts_with_strategy) +" attacks.")
print ("We expect attackers need to attempt fewer attacks using the strategies.")


print ("Reproducing results in Sec 4 (Defense) ...")

print ("As we did earlier, without using our defense,")
print ("Benign accuracy of the model is: "+format(lresults[0], ".4f"))
print ("Benign accuracy on the targeted classes is: "+format(lresults[1], ".4f"))
print ("Group-based robustness of the model is: "+format(lresults[2], ".4f"))

GADVnetwork = GTSRBNet4()
GADVnetwork=copy.deepcopy(ADVnetwork)
GADVnetwork.eval()
GADVoptimizer=optim.Adam(GADVnetwork.parameters())
print ("Training with our approach for 1 more iteration")
GADVnetwork=GroupTrain(GADVnetwork,GADVoptimizer, x_train, y_train,x_test,y_test,384,max_iters=1,eps=8/255,m=2,device='cpu',dataset='GTSRB',nclass=43,x_val=x_val,y_val=y_val,origins=origins,pairs=pairs,v=0.05)

print ("With our defense, ")
acc=test(GADVnetwork,x_test,y_test,512,device='cpu')
print ("Benign accuracy of the model is :"+format(acc, ".4f"))
acc=test(GADVnetwork,x_select,y_select,512,device='cpu')
print ("Benign accuracy on the targeted classes is :"+format(acc, ".4f"))
apgd=PGD(GADVnetwork,nclass=43,eps=8/255,loss='md_single',norm="Linf",device="cpu",n_iter=1)
succ,best,worst=APGD_caller(apgd,pairs,x_test,y_test,bs=512,auto=True)
print ("Group-based robustness of the model is: "+format(1-best, ".4f"))
print ("We expect models with our defense to outperform models without in all three metrics.")

print ("Instead, without our defense,")
FADVnetwork = GTSRBNet4()
FADVnetwork=copy.deepcopy(ADVnetwork)
FADVnetwork.eval()
FADVoptimizer=optim.Adam(FADVnetwork.parameters())
print ("Existing adversarial training for 1 more iteration")
FADVnetwork=FreeAdvTrain(FADVnetwork,FADVoptimizer, x_train, y_train,x_val,y_val,512,max_iters=1,eps=8/255,m=2,device='cpu',dataset='GTSRB')

acc=test(FADVnetwork,x_test,y_test,512,device='cpu')
print ("Benign accuracy of the model is :"+format(acc, ".4f"))
acc=test(FADVnetwork,x_select,y_select,512,device='cpu')
print ("Benign accuracy on the targeted classes is :"+format(acc, ".4f"))
apgd=PGD(FADVnetwork,nclass=43,eps=8/255,loss='md_single',norm="Linf",device="cpu",n_iter=1)
succ,best,worst=APGD_caller(apgd,pairs,x_test,y_test,bs=512,auto=True)
print ("Group-based robustness of the model is: "+format(1-best, ".4f"))









			
	
	
	
	
    


