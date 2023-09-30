# Usage
### Install Required Packages
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements.txt
```
### Download dataset
Download dataset from https://image-net.org/index.php
Specifically, we used the validation set of ILSVRC2012 (6.3GB, MD5: 29b22e2961454d5413ddabcf34fc5622)

### install robustness package
Then download the robustness package. You might do the following:
```
git clone https://github.com/MadryLab/robustness.git
```
You might also need to download pretrained models listed at https://github.com/MadryLab/robustness as needed.
In our experiments, we used the defense against L2 attacks at distance 3.0 and the defense against Linf attacks at distance 8/255.

### Setup for experiments
First, let's sample 60 classes out of 1000. You might run
```
python3 resampling.py --data_path DATA_PATH
```
where DATA_PATH is the absolute path of the ILSVRC directory downloaded.

Then we run 
```
python3 reperm.py
```
This will create five permutations of the 60 classes we sampled, so that we can have different split of roles (e.g. instructors vs students) among the 60 classes.

For sake of time let's perform all targeted attacks first. Run
```
CUDA_VISIBLE_DEVICES=0 python3 check.py --data_path DATA_PATH --distance DISTANCE
```
where DISTANCE is either "linf" or "l2".
The result will be stored as a "dictionary" where we can look up whether it is achievable to perturb any instance as any class.
To summarize the "dictionary" with respect to specific impersonation goals, run 
```
python3 get.py --distance DISTANCE --permutation PERMUTATION
```
where PERMUTATION is an integer in [0,4].

### Run new loss function
You may do 
```
CUDA_VISIBLE_DEVICES=0 python3 check_GROUP.py --data_path DATA_PATH --distance DISTANCE --permutation PERMUTATION
```
for attacks with the MDMUL loss.  
Similarly, you may do
```
CUDA_VISIBLE_DEVICES=0 python3 check_MAX.py --data_path DATA_PATH --distance DISTANCE --permutation PERMUTATION
```
for attacks with the MDMAX loss.
To reproduce the information we used to make plots, you may do
```
python3 3drange.py --distance DISTANCE
```

### prepare for attack strategies
You may do 
```
python3 matrix.py --distance DISTANCE
```
to partition a validation set and obtains a prior that will used in strategies.
Alternatively, you might do
"""
CUDA_VISIBLE_DEVICES=0 python3 check_ORDER.py --data_path DATA_PATH --distance DISTANCE"
to calculate the MD loss before and after one iteration of attack that will later used in strategies.
We also want to select set of images X where successful impersonation could be found on.
You may run 
```
python3 batch_sampling.py --distance DISTANCE
```

### execute the strategies
You may run 
```
python3 timer.py --method METHOD distance DISTANCE
```
where METHOD is one of the following strategies:
"baseline": no strategies deployed.
"matrix": Estimate by Computing a Prior from Validation Set
"prev" : Estimate by MD Loss Without Perturbation
"after" Estimate by MD Loss After One Attack Iteration
"matrix-prev": Estimate by Computing a Prior from Validation Set and Estimate by MD Loss Without Perturbation
"matrix-after": Estimate by Computing a Prior from Validation Set and Estimate by MD Loss After One Attack Iteration

There are two variations of experiments (1. Burglars has to impersonate at least one burglar 2. The same burglar cannot impersonate two staff members).
Use batch_samplingM.py/timerM.py for variarion 1 and batch_samplingK.py/timerK.py for variarion 2.
