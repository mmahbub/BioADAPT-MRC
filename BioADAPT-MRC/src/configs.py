data_dir                      = '/net/kdinxidk03/opt/NFS/75y/data/qa/output/'
output_dir                    = '/net/kdinxidk03/opt/NFS/75y/data/qa/output/'
output_model_dir              = '/net/kdinxidk03/opt/NFS/75y/data/qa/output/model/'
golden_data_folder            = '/net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/bioasq/'
golden_files                  = [
                                 'Task7BGoldenEnriched/7B_golden.json', 
                                 'Task8BGoldenEnriched/8B_golden.json', 
                                 'Task9BGoldenEnriched/9B_golden.json'
                                ]
java_file_path                = '/home/75y/project_dir/QA-IN-PROGRESS/src_bioasq/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5'
trained_model_name            = 'model_baseline.pt' #model_7b.pt #model_8b.pt #model9b.pt
out_domain_name               = 'bioasq_7B' #'bioasq_8B', 'bioasq_9B'
original_model_name_or_path   = 'bioelectra'
pretrained_model_name_or_path = 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt'
no_cuda                       = False
device                        = "cuda"
which_gpu                     = "1"
n_gpu                         = 1
thread                        = 512
local_rank                    = -1
per_gpu_eval_batch_size       = 8
tokenizer_name                = ''
model_type                    = 'electra'
cache_dir                     = ''
n_best_size                   = 5
max_query_length              = 64
max_answer_length             = 30
verbose_logging               = False
max_seq_length                = 384
doc_stride                    = 128
emb_dim                       = 768
do_lower_case                 = True
do_test                       = True
USE_TRAINED_MODEL             = True
freeze_encoder                = True
freeze_qa_output_generator    = True
freeze_discriminator_encoder  = True
freeze_aux_qa_output_generator= True
seed                          = 42
