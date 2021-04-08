# DAGN: Discourse-Aware Graph Network for Logical Reasoning (NAACL'21)

Official implementation for the NAACL'21 short paper [DAGN: Discourse-Aware Graph Network for Logical Reasoning](https://arxiv.org/abs/2103.14349). 


<img src="./fig/DAGN.pdf" align=center />


## Requirements
* Python 3.0+
* Pytorch 1.4.0+
* transformers 3.1.0
* gensim
* tqdm


## Results
|    |Dev|Test|Test-E|Test-H|
|----|:----:|:----:|:----:|:----:|
| DAGN       | 65.20 | 58.20 | 76.14 | 44.11 |
| DAGN (Aug) | 65.80 | 58.30 | 75.91 | 44.46 | 


1. ## Citation
1. ```
1. @inproeedings{yinya2021dagn,
1.     title = "DAGN: Discourse-Aware Graph Network for Logical Reasoning",
1.     auther = "Yinya {Huang} and Meng {Fang} and Yu {Cao} and Liwei {Wang} and Xiaodan {Liang}",
1.     booktitle = "NAACL-HLT 2021: Annual Conference of the North American Chapter of the Association for Computational Linguistics",
1.     year = "2012"
1. }
1.    
1. ```
