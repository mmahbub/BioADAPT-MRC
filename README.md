# BioADAPT-MRC

This repository is the official implementation of paper "BioADAPT-MRC: adversarial learning-based domain adaptation improves biomedical machine reading comprehension task". 

The repo's structure is:

```
BioADAPT-MRC
├─ BioADAPT-MRC
│    ├─ readme.md
│    ├─ environment.yml
│    ├─ src
│        ├─ configs.py
│        ├─ tokenization.py
│        ├─ enc_disc_mrc.py
│        ├─ bioadapt_mrc_model.py
│        └─ test.py
├─ README.md
└─ LICENSE.md
```

## Preparations

### Data and Model Params

[Trained Models's parameters](https://drive.google.com/drive/folders/17769XOnmhp9H0t_4E0EAUb4Th7F0z6z1?usp=sharing)
 and [processed test data](https://drive.google.com/drive/folders/1YxGEJiURH49Twl_rj6AlJK9zeVWdNfa0?usp=sharing) can be directly downloaded from Google Drive.

The baseline model is located at `model_baseline.pt`. The models trained on BioASQ-7b, BioASQ-8b, and BioASQ-9b can be found at `model_7b.pt`, `model_8b.pt`, and `model_9b.pt`, respectively.

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

For evaluation, please follow the steps below:

* save the [transform_n2b_factoid.py](https://github.com/dmis-lab/bioasq8b/blob/master/factoid/biocodes/transform_n2b_factoid.py
) file to `../src/`
* clone the repo with [the BioASQ official evaluation tool](https://github.com/BioASQ/Evaluation-Measures) in `../BioADAPT-MRC/`
* make three directories: `../BioADAPT-MRC/output/`, `../BioADAPT-MRC/data/` and `../BioADAPT-MRC/model/`
* download the golden-enriched test sets from [the official BioASQ-challenge website](http://participants-area.bioasq.org/datasets/) and save to `../data/`
* download the pre-processed test sets from the Google drive and save to `../data/`
* download the models from the Google drive and save to `../model/`
* Set the `root_path` (in the `../src/configs.py` file) to the root path of the BioADAPT-MRC folder
* For evaluating the baseline model, set `trained_model_name = 'model_baseline.pt'` in the `../src/configs.py` file.
  
  For evaluating the trained model, set `trained_model_name = f'model_{bioasq_comp_num}b.pt'` in the `../src/configs.py` file.
* For evaluating the baseline or trained model on BioASQ-7b test set, set `bioasq_comp_num = '7'` in the `../src/configs.py` file.
  
  For evaluating the baseline or trained model on BioASQ-8b test set, set `bioasq_comp_num = '8'` in the `../src/configs.py` file.
  
  For evaluating the baseline or trained model on BioASQ-9b test set, set `bioasq_comp_num = '9'` in the `../src/configs.py` file.
* Run `merge_test_batches.py` to merge all 5 golden-enriched test batches.
* Run the tokenization file in the `../src/` folder:
  ```
  python3 tokenization.py
  ```
* Run the evaluation file in the `../src/` folder:
  ```
  python3 test.py
  ```

## Results

The output for our trained model will be as follows:

### 7b
```
SAcc: 0.4506
LAcc: 0.642
MRR : 0.5286
```

### 8b
```
SAcc: 0.3841
LAcc: 0.6159
MRR : 0.4844
```

### 9b
```
SAcc: 0.5583
LAcc: 0.7423
MRR : 0.6308
```
