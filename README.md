<!-- # DAGN: Discourse-Aware Graph Network for Logical Reasoning (NAACL'21) -->
# Discourse-Aware Graph Networks (DAGNs)

<!-- Official implementations for the NAACL'21 short paper [DAGN: Discourse-Aware Graph Network for Logical Reasoning](https://arxiv.org/abs/2103.14349).  -->

## News
* [Discourse-Aware Graph Networks for Textual Logical Reasoning](https://ieeexplore.ieee.org/document/10136812) is accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)!
* [DAGN: Discourse-Aware Graph Network for Logical Reasoning](https://aclanthology.org/2021.naacl-main.467/) is accepted to NAACL 2021!


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

## Run the Extended DAGNs
```
sh run_extended_dagns.sh
```

## Run DAGN
```
sh run_dagn.sh
sh run_dagn_aug.sh

sh logiqa_run_dagn.sh
sh logiqa_run_dagn_aug.sh
```

## Run pre-trained LM baseline
```
sh run_roberta_large.sh
sh logiqa_run_roberta_large.sh
```


## Citations
If you find any part of our papers or code is helpful, please generously cite with:
```
@ARTICLE{10136812,
    author={Huang, Yinya and Liu, Lemao and Xu, Kun and Fang, Meng and Lin, Liang and Liang, Xiaodan},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Discourse-Aware Graph Networks for Textual Logical Reasoning}, 
    year={2023},
    volume={45},
    number={10},
    pages={11668-11688},
    keywords={Cognition;Task analysis;Representation learning;Transformers;Semantics;Linguistics;Birds;Natural language processing;logical reasoning;question answering;multi-turn dialogue reasoning;graph neural networks;supervised learning;zero-shot learning},
    doi={10.1109/TPAMI.2023.3280178}}

```

```
@InProceedings{zhang2021video,
    author = {Huang, Yinya and Fang, Meng and Cao, Yu and Wang, Liwei and Liang, Xiaodan},
    title = {DAGN: Discourse-Aware Graph Network for Logical Reasoning},
    booktitle = {NAACL},
    year = {2021}
} 
```
