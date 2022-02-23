import json
import sys
sys.path.append('../')
import src.configs as configs

num    = configs.bioasq_comp_num
path   = configs.golden_data_folder
path_1 = path+f'Task{num}BGoldenEnriched/{num}B1_golden.json'
path_2 = path+f'Task{num}BGoldenEnriched/{num}B2_golden.json'
path_3 = path+f'Task{num}BGoldenEnriched/{num}B3_golden.json'
path_4 = path+f'Task{num}BGoldenEnriched/{num}B4_golden.json'
path_5 = path+f'Task{num}BGoldenEnriched/{num}B5_golden.json'

b1 = json.load(open(path_1))
b2 = json.load(open(path_2))
b3 = json.load(open(path_3))
b4 = json.load(open(path_4))
b5 = json.load(open(path_5))

b_all = (
    b1['questions']+
    b2['questions']+
    b3['questions']+
    b4['questions']+
    b5['questions']
    )

b_all_dict = {'questions':b_all}

out_path  = path+f'Task{num}BGoldenEnriched/'
with open(out_path+f'{num}B_golden.json', "w") as writer:
    writer.write(json.dumps(b_all_dict, indent=4) + "\n") 