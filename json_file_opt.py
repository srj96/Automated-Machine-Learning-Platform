import json

optimize_parameters = {}
# optimize_parameters['params'] = []

optimize_parameters.update({ 'params':
    {'n_estimators': [1 ,3 ,5 ,7],
    'max_depth': [3 ,4 ,7 ,8],
    'colsample_bytree': [0.2 ,0.3 ,0.4 ,0.5],
    'gamma': [0.2 ,0.4 ,0.5 ,0.6],
    'learning_rate': [0.01 ,0.02 ,0.03 ,0.4]},
    'model_path': '/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt',
    'num_of_iter': 5, 'initial_points': 5, 'maximum_eval': 20, 'cross_validation': 5
})

save_file = '/home/congnitensor/Python_projects/model_class/json_log/grid_search_xgb.txt'

with open(save_file,'w') as outfile:
    json.dump(optimize_parameters,outfile) 
