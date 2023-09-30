# Usage
### Install Required Packages
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements.txt
```
### Download datasets
Download dataset from https://www.cs.columbia.edu/CAVE/databases/pubfig/
### Preprocess datasets
You may run
```
python3 process.py --in_dir INPUT_DIR --out_dir OUTPUT_DIR
```


### Experiment setup
Follow instructions at https://github.com/tongwu2020/phattacks/tree/master/glass, download VGGpretrained model and go through the DOA adversarial training process.
In addition, generate random permutation of classes by
```
python3 randomizer.py
```


### Experiments for loss functions
```
CUDA_VISIBLE_DEVICES=0 python3 glass_attack.py
CUDA_VISIBLE_DEVICES=0 python3 glass_attackMAX.py
CUDA_VISIBLE_DEVICES=0 python3 glass_attackGROUP.py
```

and then
```
python3 get.py
python3 3drange.py
```

### Experiments for attack strategies
You may run 
```
python3 matrix.py
python3 batch_sampling.py
python3 time.py
```

### Experiments for defenses
You may run 
```
CUDA_VISIBLE_DEVICES=0 python3 defense_train.py
```
