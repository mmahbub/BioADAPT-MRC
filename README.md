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
│        ├─ load_dataset.py
│        ├─ data_generator.py
│        ├─ enc_disc_qa.py
│        ├─ model.py
│        ├─ test.py
│        └─ transform_n2b_factoid.py
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
environment called `mrc`. In order to create the environment type: 

```
conda env create -f env.yml
```

## Evaluating

For evaluation, please follow the steps below:
```
* 
* 

### Model Training

If you want to train the model from scratch (which means from original BERT), please download the [BERT param](https://github.com/google-research/bert) 
to a `model` folder, 
and [the feature-enrich squad training data (`data/squad`)](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing)
 to a `data` folder, indicate the variables `MODEL_DIR` and `DATA_DIR`,

```
export MODEL_DIR=/full/path/to/model
export DATA_DIR=/full/path/to/data
export OUTPUT_DIR=/please/set/an/output/dir
```
and run the training file:
```
python run_factoid_pos_ner.py \
     --do_train=True\
     --do_predict=True \
     --vocab_file=$MODEL_DIR/vocab.txt \
     --bert_config_file=$MODEL_DIR/bert_config.json \
     --init_checkpoint=$MODEL_DIR/model.ckpt-1000000 \
     --max_seq_length=384 \
     --train_batch_size=8 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=4.0 \
     --do_lower_case=False \
     --train_file=$DATA_DIR/ner_pos_train-v1.1.json \
     --predict_file=$DATA_DIR/ner_pos_BioASQ-test-factoid-6b-1.json \
     --output_dir=$OUTPUT_DIR
```

If you want to train the model from SQuAD-trained step, please download the [squad param under `model/squad`](https://drive.google.com/drive/folders/1mQ68-CIsz3izoj_yuzVE86o8URN2o4SD?usp=sharing)
 to a `model` folder, 
and [the target BioASQ training data (e.g., `data/6b`)](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing)
 to a `data` folder, similarly indicate the variables `MODEL_DIR` and `DATA_DIR`,

```
export MODEL_DIR=/full/path/to/model
export DATA_DIR=/full/path/to/data
export OUTPUT_DIR=/please/set/an/output/dir
```
and run the training file:
```
python run_factoid_pos_ner.py \
     --do_train=True\
     --do_predict=True \
     --vocab_file=$MODEL_DIR/vocab.txt \
     --bert_config_file=$MODEL_DIR/bert_config.json \
     --init_checkpoint=$MODEL_DIR/model.ckpt-1000000 \
     --max_seq_length=384 \
     --train_batch_size=8 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=4.0 \
     --do_lower_case=False \
     --train_file=$DATA_DIR/ner_pos_6b.json \
     --predict_file=$DATA_DIR/ner_pos_BioASQ-test-factoid-6b-1.json \
     --output_dir=$OUTPUT_DIR
```

### Model Prediction
If you only want to predict the result on a specific BioASQ challenge test set, please download the [the target model param under for example `model/6b`](https://drive.google.com/drive/folders/1mQ68-CIsz3izoj_yuzVE86o8URN2o4SD?usp=sharing)
 to a `model` folder, 
and [the target BioASQ test data (e.g., `data/6b`)](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing)
 to a `data` folder, similarly indicate the variables `MODEL_DIR` and `DATA_DIR`,

```
export MODEL_DIR=/full/path/to/model
export DATA_DIR=/full/path/to/data
export OUTPUT_DIR=/please/set/an/output/dir
```
and run the training file:
```
python run_factoid_pos_ner.py \
     --do_train=False\
     --do_predict=True \
     --vocab_file=$MODEL_DIR/vocab.txt \
     --bert_config_file=$MODEL_DIR/bert_config.json \
     --init_checkpoint=$MODEL_DIR/model.ckpt-1000000 \
     --max_seq_length=384 \
     --train_batch_size=8 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=4.0 \
     --do_lower_case=False \
     --train_file=$DATA_DIR/ner_pos_6b.json \
     --predict_file=$DATA_DIR/ner_pos_BioASQ-test-factoid-6b-1.json \
     --output_dir=$OUTPUT_DIR
```

### Evaluation 

To evaluate BioASQ answers, the system should be able to execute java codes, and please refer https://github.com/BioASQ/Evaluation-Measures, the BioASQ official evaluation tool, for details.

Besides, we have utilized the transformation script released by [DMIS-LAB](https://github.com/dmis-lab/bioasq-biobert/tree/v1.0/biocodes)

To conduct the complete evalution process, it is required the golden answers released by BioASQ Challenge, which could be
 downloaded after registering on http://www.bioasq.org/. 

## Results
### 6b
| Metrics  | SAcc  | LAcc  | MRR  |
| ------------- |-------------:| -----:|-----:|
| Our Model   | 0.4517 | 0.6294 | 0.5197 |

### 7b
| Metrics  | SAcc  | LAcc  | MRR  |
| ------------- |-------------:| -----:|-----:|
| Our Model   | 0.4444 | 0.6419| 0.5165 |

### 8b
| Metrics  | SAcc  | LAcc  | MRR  |
| ------------- |-------------:| -----:|-----:|
| Our Model   | 0.3937 | 0.6098| 0.4688 |

