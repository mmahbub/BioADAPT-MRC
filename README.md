# bioadapt-mrc


# BioQA_externalFeatures

This repository is the official implementation of paper "External Features Enriched Model for Biomedical Question Answering". 

The code is based on the [original BERT Repository](https://github.com/google-research/bert) released by Google Team and the [BioBERT Model](https://github.com/dmis-lab/biobert) released by [DMIS-LAB](https://github.com/dmis-lab).

The repo's structure is:
```
BioQAExternalFeatures
├─ dataProcess
│    ├─ bioner.py
│    ├─ posTreat.py
│    ├─ pos_ner_treat.py
│    └─ readme.md
├─ scripts
│    ├─ evaluation.sh
│    ├─ prediction.sh
│    ├─ readme.md
│    └─ training.sh
├─ LICENSE
├─ README.md
├─ __init__.py
├─ create_pretraining_data.py
├─ extract_features.py
├─ modeling.py
├─ modeling_test.py
├─ optimization.py
├─ optimization_test.py
├─ requirements.txt
├─ run_factoid_baseline.py # Original BioBERT model 
├─ run_factoid_pos_ner.py # Our model (Feature Fusion + NER feature + POS feature)
├─ tokenization.py
└─ tokenization_test.py
```

## Preparations

### Data and Model Params

[Trained Models's parameters](https://drive.google.com/drive/folders/1mQ68-CIsz3izoj_yuzVE86o8URN2o4SD?usp=sharing)
 and [processed data (added POS and NER labels)](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing) can be directly downloaded from Google Drive.

For Chinese users, if the speed of Google Drive is limited, please contact us for the data and models on Baidu Pan.

For the trained models:

* We firstly fine-tuned [BERT (BERT-Base, Multilingual Cased)](https://github.com/google-research/bert) under our enriched external-feature framework on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/), 
of which the param could be found under `model/squad`

* Next, we further fine-tuned the model respectively on Biomedical Training Sets (6b, 7b and 8b), and the well trained models could be 
downloaded from `model/6b`, `model/7b`, `model/8b`.

For the data utilized in our experiments:

* The data mainly come from `SQuAD` and `BioASQ Challenge`. Particularly, for the biomedical QA data, we have utilized the
 enriched data provided by [DMIS-LAB](https://github.com/dmis-lab);

* You can get the feature-enriched training data under two ways:

   * Directly download the processed data from our [google drive link](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing)
   
   * Use the provided tools in `dataProcess` to extract the NER, POS and BioNER features on your own side.  

### Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Utilization

Following are the using guides for training, predicting and evaluating our framework. For more information,
 please refer to the script files in the `scripts` folder.

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

