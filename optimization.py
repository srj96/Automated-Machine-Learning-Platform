from grid_search import GridSearch
from bayesopt import BayesOptimize
from hoptimize import HyperOptimize 
from xgboost import XGBRegressor
import argparse 


class OptimizeMethod:
    
    def __init__(self,params = None,init_points = None,n_iter = None,max_evals = None,cross_val = None):
        
        self.params = params 
        self.init_points = init_points
        self.n_iter = n_iter
        self.max_evals = max_evals 
        self.cross_val = cross_val
    
    def grid_search(self):
        obj_1 = GridSearch(self.params)
        obj_1.grid_search_cv(self.cross_val)
    
    def bayes_opt(self):
        obj_2 = BayesOptimize(self.params)
        obj_2.objective_bayes()
        obj_2.bayesoptimize(self.init_points,self.n_iter)
    
    def hyper_opt(self):
        obj_3 = HyperOptimize(self.params)
        obj_3.objective_hopt(self.params)
        obj_3.hyperopt(self.max_evals)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n_estimators", nargs = '+', type = int)
    parser.add_argument("-max_depth", nargs = '+', type = int)
    parser.add_argument("-colsample_bytree", nargs = '+', type = float)
    parser.add_argument("-gamma", nargs = '+', type = float)
    parser.add_argument("-learning_rate", nargs = '+' , type = float)
    parser.add_argument("-num_of_iter" , "--number_of_iteration" , nargs = '?',const = 10 , type = int, default = 5)
    parser.add_argument("-init_points", "--number_of_initial_points", nargs = '?', const = 5, type = int, default = 5)
    parser.add_argument("-max_evals", "--maximum_evaluations", nargs = '?', const = 20 , type = int , default = 20)
    parser.add_argument("-cross_val", "--cross_validation", nargs = '?', const = 3, type = int, default = 3)
    parser.add_argument("-o", "--option" , help = "enter the required parameter value for optimization")

    result = vars(parser.parse_args())

    command = result.pop('option')

    print(command)

    num_iter = result.pop('number_of_iteration')

    print(num_iter)

    init_points = result.pop('number_of_initial_points')

    print(init_points)

    max_evals = result.pop('maximum_evaluations')

    cross_val = result.pop('cross_validation')


    param_list = {k : result[k] for k in result if result[k] != None}

    print(param_list)

    obj = OptimizeMethod(params = param_list, init_points = init_points , n_iter = num_iter , max_evals = max_evals, cross_val= cross_val) 

    if command == 'grid_search':
        obj.grid_search() 

    if command == 'bayesian':
        obj.bayes_opt()
    
    if command == 'hyperopt':
        obj.hyper_opt()






