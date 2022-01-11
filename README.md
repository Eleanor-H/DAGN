# DAGN: Discourse-Aware Graph Network for Logical Reasoning (NAACL'21)

Official implementation for the NAACL'21 short paper [DAGN: Discourse-Aware Graph Network for Logical Reasoning](https://arxiv.org/abs/2103.14349). 


## Dependencies
This code has been tested with the following dependencies and versions:
```
python==3.7.9
torch==1.5.0
transformers==3.1.0
numpy==1.19.2
gensim==3.8.3
```

## Preparation
The [ReClor data](https://whyu.me/reclor/#download) is ready in the `./reclor_data`.
To run the [LogiQA data](https://github.com/lgw863/LogiQA-dataset), create `./logiqa_data` where you put the downloaded data.


## Run pre-trained LM baseline
```
sh run_roberta_large.sh
sh logiqa_run_roberta_large.sh
```

## Run DAGN
```
sh run_dagn.sh
sh run_dagn_aug.sh

sh logiqa_run_dagn.sh
sh logiqa_run_dagn_aug.sh
```

## Citation
If any part of our paper or code is helpful, please generously cite with:
```
@InProceedings{zhang2021video,
author = {Huang, Yinya and Fang, Meng and Cao, Yu and Wang, Liwei and Liang, Xiaodan},
title = {DAGN: Discourse-Aware Graph Network for Logical Reasoning},
booktitle = {NAACL},
year = {2021}
} 
```
