# Group-based Robustness
This is the official code repository for the NDSS'24 paper: "Group-based Robustness: A General Framework for Customized Robustness in the Real World" (https://arxiv.org/pdf/2306.16614.pdf).

We propose the following to better evaluate the threat in realistic attack scenarios:
1. A new metric, group-based robustness, that empirically evaluates the threat more accurately
2. Two new loss functions, MDMAX and MDMUL, that more efficiently estimate the threat
3. Several attack strategies that more efficiently estimate the threat
4. A defense approach that mitigates the threat

# Usage
You may find the instructions to reproduce our results in each directory corresponding to each dataset. GPUs are highly recommended for all experiments.

## Citation
```
@inproceedings{weiran2024GBR,
    title = {Group-based Robustness: A General Framework for Customized Robustness in the Real World},
    author = {Weiran Lin and Keane Lucas and Neo Eyal and Lujo Bauer and Michael K. Reiter and Mahmood Sharif},
    booktitle = {NDSS},
    year = {2024}
}
```

# Acknowledgement
Our code is heavily based on the following implementations:
- Autoattack (https://github.com/fra31/auto-attack)
- robustness package (https://github.com/MadryLab/robustness)
- T-PGD (https://github.com/Phantivia/T-PGD)

We greatly appreciate authors of the above for their fantastic and innovative work.

We also reused some of the code from our previous CGD attack implementation: (https://github.com/linweiran/CGD),
which you are very welcome to check out!
