# Usage

It's highly recommended to run our implmentation with GPUs. However, we designed a scale-down mini-experiment for NDSS artifact review (https://secartifacts.github.io/ndss2024/), which will be denoted as "(without GPU)" below

### Install Required Packages (with GPU)
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements.txt
```
We highly encourage the use of virtual environments as these packages may conflict with your existing ones.

### Install Required Packages (without GPU)
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements-cpu.txt
```
We highly encourage the use of virtual environments as these packages may conflict with your existing ones.


### Download datasets (with or without GPU)
Please download the dataset from the downloads section of https://benchmark.ini.rub.de/gtsrb_dataset.html#Downloads where a link can be found. Specifically, please download files named "GTSRB_Final_Training_Images.zip", "GTSRB_Final_Test_Images.zip", and "GTSRB_Final_Test_GT.zip". After extracting these zip files, please move the directories named "Final_Test" and "Final_Training", along with the file "GT-final_test.csv" to the same directory.

### Preprocess datasets (with or without GPU)
You may run 
```
python3 preprocess.py --data_path DATA_PATH
```
 where "DATA_PATH" is the path to the directory where the extracted files are stored (mentioned above)

### Run experiments (without GPU)
You may run 
```
python3 hello_world.py --data_path DATA_PATH
```
 where "DATA_PATH" is the path to the directory where the extracted files are stored (mentioned above)

### Run experiments (with GPU)
To reproduce experiments on the new loss functions (MDMAX, MDMUL), you may run
```
CUDA_VISIBLE_DEVICES=0 python3 Freetrain_GTSRB_gpu_indv.py --data_path DATA_PATH
```
 where "DATA_PATH" is the path to the directory where the extracted files are stored (mentioned above).

 To reproduce experiments on the new defense, you may run
```
CUDA_VISIBLE_DEVICES=0 python3 Grouptrain_GTSRB_gpu_indv.py --data_path DATA_PATH
```
 where "DATA_PATH" is the path to the directory where the extracted files are stored (mentioned above).
