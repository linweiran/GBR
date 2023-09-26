# Usage
### Install Required Packages
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements.txt
```
Then download the USE-4 model from https://tfhub.dev/google/universal-sentence-encoder/4 

### Untargeted Attacks
```
CUDA_VISIBLE_DEVICES=0 python3 RunTPGD.py --task SetFit/sst5 --tokenizer_checkpoint roberta-large --decode_mode Roberta --model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_tokenizer_checkpoint Unso/roberta-large-finetuned-sst5 --start 0  --end 2210 --cuda_device 0 --victim_device -1 --perturb_layer 0 --decode_layer 16 --num_seg_steps 100 --num_adv_steps 20 --adv_lr 10 --init_mag 3 --decode_weight -0.5 --bs_lower_limit 0.0 --bs_upper_limit 0.95 --target_metric use --use_path universal-sentence-encoder_4
```

### Targeted Attacks
```
CUDA_VISIBLE_DEVICES=0 python3 RunTPGDT.py --task SetFit/sst5 --tokenizer_checkpoint roberta-large --decode_mode Roberta --model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_tokenizer_checkpoint Unso/roberta-large-finetuned-sst5 --start 0  --end 2210 --cuda_device 0 --victim_device -1 --perturb_layer 0 --decode_layer 16 --num_seg_steps 100 --num_adv_steps 20 --adv_lr 10 --init_mag 3 --decode_weight -0.5 --bs_lower_limit 0.0 --bs_upper_limit 0.95 --target_metric use --target_class 4 --use_path universal-sentence-encoder_4
```

### Group-based Attacks (with MDMAX loss function)
```
CUDA_VISIBLE_DEVICES=0 python3 RunTPGDMAX.py --task SetFit/sst5 --tokenizer_checkpoint roberta-large --decode_mode Roberta --model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_model_checkpoint Unso/roberta-large-finetuned-sst5 --victim_tokenizer_checkpoint Unso/roberta-large-finetuned-sst5 --start 0  --end 2210 --cuda_device 0 --victim_device -1 --perturb_layer 0 --decode_layer 16 --num_seg_steps 100 --num_adv_steps 20 --adv_lr 10 --init_mag 3 --decode_weight -0.5 --bs_lower_limit 0.0 --bs_upper_limit 0.95 --target_metric use --impersonation A --use_path universal-sentence-encoder_4
```

