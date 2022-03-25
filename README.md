# BioADAPT-MRC training

* Download training data for SQuAD-1.1 from [here](https://deepai.org/dataset/squad)
* Dowload pre-processed training set for 7B, 8B from [here](https://drive.google.com/drive/folders/1SlgDQUg2hNMBRDgPZlqo_ucRZpDM3TV6)
  Due to privacy reasons, we can not make the preprocessed training dataset for BioASQ-9B pubicly available. However, the raw dataset can be found [here](http://participants-area.bioasq.org/datasets/) and the pre-processing steps are described in the [paper](https://arxiv.org/abs/2202.13174)
* Make three directories: `../BioADAPT-MRC/output/`, `../BioADAPT-MRC/data/` and `../BioADAPT-MRC/model/`  
* Set the `root_path` (in the `../src/configs.py` file) to the `root/path/of/the/BioADAPT-MRC/folder/`
* For training the BioADAPT-MRC model on BioASQ-7b, set `bioasq_comp_num = '7'` in the `../src/configs.py` file.
* Run the tokenization file in the `../src/` folder:
  ```
  python3 tokenization.py
  ```
* Run the train file in the `../src/` folder:
  ```
  python3 train.py
  ```
