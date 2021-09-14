import json

base_parameters = {}

base_parameters.update({
    'path' : '/home/congnitensor/Python_projects/model_class/poker_hand_training.csv',
    'table_name' : 'None',
    'target' : 'class',
    'split'  : 0.3 
})

save_file = '/home/congnitensor/Python_projects/model_class/json_log/data.txt'

with open(save_file,'w') as outfile:
    json.dump(base_parameters,outfile)

