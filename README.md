# BioADAPT-MRC

This repository is the official implementation of the paper [BioADAPT-MRC: adversarial learning-based domain adaptation improves biomedical machine reading comprehension task](https://arxiv.org/abs/2202.13174).

If you use any part of this repository, we shall be obliged if you cite [our paper](https://arxiv.org/abs/2202.13174).

The repo's structure is:

```
BioADAPT-MRC
│     ├─ README.md
│     ├─ LICENSE.md
│     ├─ environment.yml
│     ├─ src
│         ├─ configs.py
│         ├─ merge_test_batches.py
│         ├─ tokenization.py
│         ├─ enc_disc_mrc.py
│         ├─ bioadapt_mrc_model.py
└─        └─ test.py
```

## Preparations

### Data and Model Params

[Trained Models's parameters](https://drive.google.com/drive/folders/17769XOnmhp9H0t_4E0EAUb4Th7F0z6z1?usp=sharing)
 and [processed test data](https://drive.google.com/drive/folders/1YxGEJiURH49Twl_rj6AlJK9zeVWdNfa0?usp=sharing) can be directly downloaded from Google Drive.

The baseline model is saved at `model_baseline_#b.pt`. The models trained on BioASQ-7b, BioASQ-8b, and BioASQ-9b are saved at `model_7b.pt`, `model_8b.pt`, and `model_9b.pt`, respectively.

The data used for testing can be found at:
* BioASQ-7b: `test_bioasq_7B.json`
* BioASQ-8b: `test_bioasq_8B.json`
* BioASQ-9b: `test_bioasq_9B.json`

### Environment Setup

The `environment.yml` file contains all the packages and dependencies needed to re-create the Anaconda
environment called `mrc`. In order to create the environment, run: 

```
conda env create -f environment.yml
```

## Evaluation

To evaluate BioASQ answers, the system should be able to execute java codes, and please refer to [the BioASQ official evaluation tool](https://github.com/BioASQ/Evaluation-Measures), for details.

Besides, we have utilized the transformation script released by [DMIS-LAB](https://github.com/dmis-lab/bioasq-biobert/tree/v1.0/biocodes)

For evaluation, please follow the steps below sequentially:

* Save the [transform_n2b_factoid.py](https://github.com/dmis-lab/bioasq8b/blob/master/factoid/biocodes/transform_n2b_factoid.py
) file to `../src/`
* Clone the repo with [the BioASQ official evaluation tool](https://github.com/BioASQ/Evaluation-Measures) in `../BioADAPT-MRC/`
* Make three directories: `../BioADAPT-MRC/output/`, `../BioADAPT-MRC/data/` and `../BioADAPT-MRC/model/`
* Download the golden-enriched test sets from [the official BioASQ-challenge website](http://participants-area.bioasq.org/datasets/) and save to `../data/`
* Download the pre-processed test sets from the Google drive and save to `../data/`
* Download the models from the Google drive and save to `../model/`
* Set the `root_path` (in the `../src/configs.py` file) to the `root/path/of/the/BioADAPT-MRC/folder/`
* For evaluating the baseline model, set `trained_model_name = 'model_baseline_#b.pt'` in the `../src/configs.py` file.
  
  For evaluating the trained model, set `trained_model_name = f'model_{bioasq_comp_num}b.pt'` in the `../src/configs.py` file.
* For evaluating the baseline or trained model on BioASQ-7b test set, set `bioasq_comp_num = '7'` in the `../src/configs.py` file.
  
  For evaluating the baseline or trained model on BioASQ-8b test set, set `bioasq_comp_num = '8'` in the `../src/configs.py` file.
  
  For evaluating the baseline or trained model on BioASQ-9b test set, set `bioasq_comp_num = '9'` in the `../src/configs.py` file.
* Run the following command to merge all 5 golden-enriched test batches for either BioASQ-7b/BioASQ-8b/BioASQ-9b:
  ```
  python3 merge_test_batches.py
  ```
* Run the tokenization file in the `../src/` folder:
  ```
  python3 tokenization.py
  ```
* Run the evaluation file in the `../src/` folder:
  ```
  python3 test.py
  ```

## Results

The output scores for our trained model should be as follows:

### 7b
```
SAcc: 0.4506
LAcc: 0.6420
MRR : 0.5289
```

### 8b
```
SAcc: 0.3841
LAcc: 0.6225
MRR : 0.4749
```

### 9b
```
SAcc: 0.5399
LAcc: 0.7423
MRR : 0.6187
```

### Please reach out if you run into any issues.

# Citation
Mahbub, M., Srinivasan, S., Begoli, E. and Peterson, G.D., 2022. BioADAPT-MRC: Adversarial Learning-based Domain Adaptation Improves Biomedical Machine Reading Comprehension Task. arXiv preprint arXiv:2202.13174.

## BibTeX:
```
@article{mahbub2022bioadapt,
  title={BioADAPT-MRC: Adversarial Learning-based Domain Adaptation Improves Biomedical Machine Reading Comprehension Task},
  author={Mahbub, Maria and Srinivasan, Sudarshan and Begoli, Edmon and Peterson, Gregory D},
  journal={arXiv preprint arXiv:2202.13174},
  year={2022}
}
```
