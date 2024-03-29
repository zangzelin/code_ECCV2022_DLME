
# Deep Local-flatness Manifold Embedding (DLME)

The code includes the following modules:
* Datasets (digits, coil20, coil100, Mnist, EMNIST, KMnsit, Colon, Activity, MCA, Gast10K, Samusik, HCL)
* Training for DLME
* Evaluation metrics 
* Visualisation

## Requirements

* pytorch == 1.11.0
* pytorch-lightning == 1.4.8
* torchvision == 0.12.0
* scipy == 1.8.0
* numpy == 1.18.5
* scikit-learn == 1.0
* matplotlib == 3.4.3
* wandb == 0.12.5

## Description

* ./eval
  * eval/eval_core.py -- The code for evaluate the embedding 
* ./Loss -- Calculate losses
  * ./Loss/dmt_loss_aug.py -- The DLME loss
  * ./Loss/dmt_loss_source.py -- The template of loss function 
* ./eval -- The yaml file for gird search
* ./sweep -- the yaml file for gird search
* ./nuscheduler.py  -- Adjustment Learning Rate
* ./main.py -- End-to-end training of the DLME model
* ./load_data_f -- the dataloader
  * ./load_data_f/source.py -- The template of dataset 
  * ./load_data_f/dataset.py -- The DLME dataset 

## Baseline Methods

The compared methods include two manifold learning methods 
([UMAP](https://github.com/lmcinnes/umap), [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)) 
and three deep manifold learning methods ([PHATE](https://github.com/KrishnaswamyLab/PHATE), [ivis](https://github.com/beringresearch/ivis) and  [parametric UMAP(P-UMAP)](https://github.com/lmcinnes/umap)).

## Dataset

The datasets include six simple image datasets ([Digits](https://scikit-learn.org/stable/auto\_examples/datasets/plot\_digits\_last\_image.html), [Coil20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), [Coil100](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php), [Mnist](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits), [EMnist](https://www.tensorflow.org/datasets/catalog/emnist), [KMnist](https://www.tensorflow.org/datasets/catalog/kmnist)) and six biological datasets ([Colon](https://figshare.com/articles/dataset/The\_microarray\_dataset\_of\_colon\_cancer\_in\_csv\_format\_/13658790/1), [Activity](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones), [MCA](https://figshare.com/articles/dataset/MCA\_DGE\_Data/5435866), [Gast10k](http://biogps.org/dataset/tag/gastric\%20carcinoma/), [SAMUSIK](https://github.com/abbioinfo/CyAnno), and HCL).

## Running the code

1. Install the required dependency packages and Before that, and configure wandb with the [instructions](https://wandb.ai/site).

2. To get the grid search results, run

  ```
  wandb sweep sweep/sweep_base_line.yaml
  ```
  and the terminal will show the id of sweep
  ```
  (torch1.8) root@4fisk2abvqo3c-0:/zangzelin/project/dlme_eccv2022# wandb sweep sweep/sweep_base_line.yaml 
  wandb: Creating sweep from: sweep/sweep_base_line.yaml
  wandb: Created sweep with ID: 1frm0208
  wandb: View sweep at: https://wandb.ai/cairi/DLME_ECCV2022/sweeps/1frm0208
  wandb: Run sweep agent with: wandb agent cairi/DLME_ECCV2022/1frm0208
  ```
  The cairi/DLME_ECCV2022/1frm0208 is the id of the sweep. 
  

3. My Replication Results: 

  https://www.wolai.com/zangzelin/gsKxT6fHMtnuwTrLprJmWb

If you find this file useful in your research, please consider citing:

```
@article{zang2022dlme,
  title={DLME: Deep Local-flatness Manifold Embedding},
  author={Zang, Zelin and Li, Siyuan and Wu, Di and Wang, Ge and Shang, Lei and Sun, Baigui and Li, Hao and Li, Stan Z},
  journal={arXiv preprint arXiv:2207.03160},
  year={2022}
}
@InProceedings{10.1007/978-3-031-19803-8_34,
author="Zang, Zelin
and Li, Siyuan
and Wu, Di
and Wang, Ge
and Wang, Kai
and Shang, Lei
and Sun, Baigui
and Li, Hao
and Li, Stan Z.",
editor="Avidan, Shai
and Brostow, Gabriel
and Ciss{\'e}, Moustapha
and Farinella, Giovanni Maria
and Hassner, Tal",
title="DLME: Deep Local-Flatness Manifold Embedding",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="576--592",
abstract="Manifold learning (ML) aims to seek low-dimensional embedding from high-dimensional data. The problem is challenging on real-world datasets, especially with under-sampling data, and we find that previous methods perform poorly in this case. Generally, ML methods first transform input data into a low-dimensional embedding space to maintain the data's geometric structure and subsequently perform downstream tasks therein. The poor local connectivity of under-sampling data in the former step and inappropriate optimization objectives in the latter step leads to two problems: structural distortion and underconstrained embedding. This paper proposes a novel ML framework named Deep Local-flatness Manifold Embedding (DLME) to solve these problems. The proposed DLME constructs semantic manifolds by data augmentation and overcomes the structural distortion problem using a smoothness constrained based on a local flatness assumption about the manifold. To overcome the underconstrained embedding problem, we design a loss and theoretically demonstrate that it leads to a more suitable embedding based on the local flatness. Experiments on three types of datasets (toy, biological, and image) for various downstream tasks (classification, clustering, and visualization) show that our proposed DLME outperforms state-of-the-art ML and contrastive learning methods.",
isbn="978-3-031-19803-8"
}
```



## License

DLME is released under the MIT license.
