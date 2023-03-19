# Python3 program to print DFS traversal
# from a given given graph
import json
from collections import defaultdict

import numpy as np

from beliefrulebase import RuleBase
from data import Data


# This class represents a directed graph using
# adjacency list representation
class Model:
    Sr = 0.55

    # Constructor
    def __init__(self, file_name, attributes):
        self.file_name = file_name
        # default dictionary to store graph
        self.graph = defaultdict(list)
        # default dictionary to trace visited node
        self.visited = defaultdict()
        self.brb_tree = defaultdict()
        self.create_model()
        self.training_data = defaultdict(list)
        self.test_data = defaultdict(list)
        self.training_data_size = None
        self.rulebase_list = defaultdict()
        self.attributes_names = attributes
        self.best_optimized_parameter = list()


    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def create_model(self):
        with open(self.file_name) as file:
            data = json.load(file)
        for each in data:
            self.visited[each] = False
            node = Data(**data[each])
            self.brb_tree[each] = node
            if each == node.parent:
                self.parent = each
            self.addEdge(node.parent, each)

    def inference(self, v, index=None):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        # object_list.append(parent)
        for i in self.graph[v]:
            node = self.brb_tree[i]

            if self.visited[i] == False:
                object_list.append(node)
                self.inference(i, index)
        if parent.is_input == False:
            if self.rulebase == None:
                self.rulebase = RuleBase(object_list, parent)
                self.rulebase.create_belief_rulebase()
            self.rulebase.activation_weight()
            res = self.rulebase.aggregate_rule()
            # rulebase.generate_extended_belief_rule_base()
            # rulebase.input_transformation()
            print(res)
        elif parent.is_input == True:
            new_input = self.training_data[parent.antecedent_id][index]
            parent.input_val = new_input
            parent.input_transformation()

    def inference_test(self, v, index=None):
        self.visited[v] = True
        object_list = []
        node = self.brb_tree[v]
        # object_list.append(parent)
        for i in self.graph[v]:
            child = self.brb_tree[i]
            if self.visited[i] == False:
                object_list.append(child)
                self.inference_test(i, index)
        if node.is_input == False and node.is_root== True:
            if self.rulebase_list.get(v)== None:
                self.rulebase_list[v]= RuleBase(object_list,node)
                self.rulebase_list[v].create_coperative_brb()
            res = self.rulebase_list[v].rules_aggregation_analytical()
            print(res)
        elif node.is_input == False and node.is_root == False:
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = RuleBase(object_list, node)
                self.rulebase_list[v].create_belief_rulebase()
            self.rulebase_list[v].activation_weight()
            res = self.rulebase_list[v].rules_aggregation_analytical()
            # rulebase.generate_extended_belief_rule_base()
            # rulebase.input_transformation()
            print(res)
        elif node.is_input == True:
            # = self.training_data[parent.antecedent_id][index]
            # parent.input_val =new_input
            node.input_transformation()

    def inference_train(self, v, training_parameter,index=None):
        self.visited[v] = True
        object_list = []
        node = self.brb_tree[v]
        # object_list.append(parent)
        for i in self.graph[v]:
            child = self.brb_tree[i]
            if self.visited[i] == False:
                object_list.append(child)
                training_parameter =self.inference_train(i,training_parameter, index)
        if node.is_input == False and node.is_root== True:
            new_input = self.training_data[node.antecedent_id][index]
            node.input_val = new_input
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = RuleBase(object_list, node)
                training_parameter = training_parameter + self.rulebase_list[v].create_population_rulebase_cooperative_brb_root()
                #print(training_parameter)
            res = self.rulebase_list[v].rules_aggregation_analytical()
            # print(res)
        elif node.is_input == False and node.is_root == False:
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = RuleBase(object_list, node)
                training_parameter = training_parameter+self.rulebase_list[v].create_population_rulebase()
                #print(training_parameter)
            self.rulebase_list[v].activation_weight()
            res = self.rulebase_list[v].rules_aggregation_analytical()
            #print(res)
        elif node.is_input == True:
            new_input = self.training_data[node.antecedent_id][index]
            node.input_val = new_input
            node.input_transformation()
        return training_parameter

    def inference_train_fitness(self, v, training_parameter, count=None, index=None):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        for i in self.graph[v]:
            node = self.brb_tree[i]
            if self.visited[i] == False:
                object_list.append(node)
                count = self.inference_train_fitness(i, training_parameter, count, index)

        if parent.is_input == False and parent.is_root==True:
            new_input = self.training_data[parent.antecedent_id][index]
            parent.input_val = new_input
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = RuleBase(object_list, parent)
                #self.rulebase_list[v].assign_belief_degree_rulebase(training_parameter, count)
                count = self.rulebase_list[v].assign_cbrb_weight_coefficient(training_parameter,count)
            #self.rulebase_list[v].activation_weight()
            res = self.rulebase_list[v].rules_aggregation_analytical()
            # print(res)

        if parent.is_input == False and parent.is_root == False:
            #new_input = self.training_data[parent.antecedent_id][index]
            #parent.input_val = new_input
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = RuleBase(object_list, parent)
                count = self.rulebase_list[v].assign_belief_degree_rulebase(training_parameter, count)
            self.rulebase_list[v].activation_weight()
            res = self.rulebase_list[v].rules_aggregation_analytical()
            # print(res)


        elif parent.is_input == True:
            new_input = self.training_data[parent.antecedent_id][index]
            parent.input_val = new_input
            parent.input_transformation()
        return count

    def predict(self):
        self.inference(self.parent)
        print('faisal')
        # self.DFSUtil(self.parent)

    def test(self, attribute_name, test_data):
        row, column = test_data.shape
        self.training_data_size = row
        for index, name in enumerate(attribute_name):
            self.training_data[name] = test_data[0:row, index]
        result = []

        for i in range(self.training_data_size):
            for name in attribute_name:
                self.visited[name] = False
            self.inference_test(self.parent, i)
            result.append(self.brb_tree[self.parent].transformed_val)
        return result

    def train(self, attribute_name, test_data):
        row, column = test_data.shape
        self.training_data_size = row
        for index, name in enumerate(attribute_name):
            self.training_data[name] = test_data[0:row, index]
        result = []

        for i in range(self.training_data_size):
            for name in attribute_name:
                self.visited[name] = False
            self.inference_train(self.parent, i)
            result.append(self.brb_tree[self.parent].transformed_val)
        return result

    def objective_function_population_fitness(self, count=None, training_data=None):
        '''''
        row,column = training_data.shape
        self.training_data_size = row
        for index,name in enumerate(attribute_name):
            self.training_data[name]=training_data[0:row,index]
        '''
        result = []
        mse = 0
        is_training_param = False
        for index in range(self.training_data_size):
            print('For training data instance ', index, ': ')
            count = 0
            for name in self.visited.keys():
                self.visited[name] = False
            training_parameter =[]
            training_parameter = self.inference_train(self.parent, training_parameter,index)
            if not is_training_param:
                param = training_parameter
                is_training_param = True
            # result.append(self.brb_tree[self.parent].transformed_val)
            predicted_value = 0
            parent = self.brb_tree[self.parent]
            for idx, value in enumerate(parent.ref_val):
                predicted_value = predicted_value + (value * parent.transformed_val[idx])
            result.append((parent.input_val, predicted_value))
            mse += np.square(predicted_value - parent.input_val)
        return param, mse / self.training_data_size

    def objective_function_fitness(self, parameters, count=None, training_data=None):
        '''''
        row,column = training_data.shape
        self.training_data_size = row
        for index,name in enumerate(attribute_name):
            self.training_data[name]=training_data[0:row,index]
        '''
        result = []
        mse = 0

        for index in range(self.training_data_size):
            count = 0
            for name in self.visited.keys():
                self.visited[name] = False
            self.inference_train_fitness(self.parent, parameters, count, index)
            # result.append(self.brb_tree[self.parent].transformed_val)
            predicted_value = 0
            parent = self.brb_tree[self.parent]
            for idx, value in enumerate(parent.ref_val):
                predicted_value = predicted_value + (value * parent.transformed_val[idx])
            result.append((parent.input_val, predicted_value))
            mse += np.square(predicted_value - parent.input_val)
        return mse / self.training_data_size

    def de(self, training_data, mut=0.5, crossp=0.9, popsize=20, its=3000):
        # dimensions = len(bounds)
        # pop = np.random.rand(popsize, dimensions)
        # min_b, max_b = np.asarray(bounds).T
        # diff = np.fabs(min_b - max_b)
        # pop_denorm = min_b + pop * diff
        row, column = training_data.shape
        self.training_data_size = row
        for index, name in enumerate(self.attributes_names):
            self.training_data[name] = training_data[0:row, index]

        # fitness = np.asarray([fobj(ind) for ind in popsize])
        fitness_values = []
        populations = []
        for idx in range(popsize):
            self.rulebase_list = defaultdict()
            pop, fitness = self.objective_function_population_fitness()
            populations.append(pop)
            fitness_values.append(fitness)
            # print(pop)
            # print(fitness)
        # print('---------------------------------------------------------------------------------')
        for i in range(popsize):
            print(populations[i],len(populations[i]))
            print(fitness_values[i])
        populations = np.asarray(populations,dtype= np.float32)
        fitness_values = np.asarray(fitness_values,dtype=np.float32)
        best_idx = np.argmin(fitness)
        best = populations[best_idx]
        dimensions = len(best)
        test_res = []
        for i in range(its):
            print('Generation :', i, end='\n')
            for j in range(popsize):
                trial = self.extended_de(populations, dimensions, j, best)

                trial = np.clip(trial,0,1)
                count = 0
                self.reset_brb_tree_node_visited()
                count =self.check_constrain_cbrb(self.parent, trial)
                print('Index of paramater whose constrain is not resolved')
                individual_cost = self.constrain_violation_value_individual(trial)
                self.reset_brb_tree_node_visited()
                count, combined_const = self.constrain_violation_value_cbrb(self.parent, trial)
                const_val = individual_cost+combined_const
                print('Constrain violation :', const_val)

                # print(trial2)
                best_idx = self.diversity_machanism(i, populations, fitness_values, j, best_idx, trial)

                ###################Original DE algorithm#####################
                count = 0
                self.rulebase_list =defaultdict()
                f= self.objective_function_fitness(trial, count)
                if f < fitness_values[j]:
                    fitness_values[j] = f
                    populations[j] = trial
                    if f < fitness_values[best_idx]:
                        best_idx = j
                        best = trial
                ##########################################################################
            # print(fitness_values[best_idx])
            # yield  fitness_values[best_idx]
            test_res.append(fitness_values[best_idx])
        return test_res
    def mutation_recombination_original(self,populations,crossp,mut,index):
        dimensions = len(populations[0])
        pop_size =len(populations)
        idxs = [idx for idx in range(pop_size) if idx != index]
        a, b, c = populations[np.random.choice(idxs, 3, replace=False)]
        # Mutation
        mutant = a + mut * (b - c)
        # Recombination
        cross_points = np.random.rand(len(populations[0])) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, populations[index])
        return trial

    def modified_mutation_recombination(self, population, index, best_solution):
        mut1 = 0.8
        mut2 = 0.2
        crossp = .9
        popsize = len(population)
        dimension = len(population[0])
        idxs = [idx for idx in range(popsize) if idx != index]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        #crossover
        child = c + mut1 * (best_solution - b) + mut2 * (population[index] - a)
        # Recombination
        cross_points = np.random.rand(dimension) < crossp

        # if there is no point to mutation, then randomly select a point to mutate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimension)] = True
        child = np.where(cross_points, child, population[index])
        return child


    def generate_population(self, popsize=20):
        return

    def pop_generation_util(self, v, population):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        for i in self.graph[v]:
            node = self.brb_tree[i]
            if self.visited[i] == False:
                object_list.append(node)
                self.pop_generation_util(i, population)
        if parent.is_input == 'false':
            rulebase = RuleBase(object_list, parent)
            population = population + rulebase.create_population_rulebase()
            print(len(population))
        return population

    def check_constrain(self, v, parameters, count):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        for child in self.graph[v]:
            node = self.brb_tree[child]
            if self.visited[child] == False:
                object_list.append(node)
                count = self.check_constrain(child, parameters, count)
        if parent.is_input == False:
            # self.rulebase_list[v] = RuleBase(object_list,parent)
            self.rulebase_list[v].check_rulebase_constrain(parameters, count)
        return count

    def check_constrain_cbrb(self, v, parameters, count = 0):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        for child in self.graph[v]:
            node = self.brb_tree[child]
            if self.visited[child] == False:
                object_list.append(node)
                count = self.check_constrain_cbrb(child, parameters, count)
        if parent.is_root == False and parent.is_input== False:
            # self.rulebase_list[v] = RuleBase(object_list,parent)
            count=self.rulebase_list[v].check_rulebase_constrain(parameters, count)
        elif parent.is_root == True and parent.is_input== False:
            count = self.rulebase_list[v].check_cbrb_rulebase_constrain_root(parameters,count)
        return count

    def extended_de(self, population, dimension, index, best_solution):
        mut1 = 0.8
        mut2 = 0.2
        crossp = .9
        trial = None
        fitness_value = 10000
        const_val = 1000
        for k in range(0, 10):
            popsize = len(population)
            idxs = [idx for idx in range(popsize) if idx != index]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            child = c + mut1 * (best_solution - b) + mut2 * (population[index] - a)
            # Recombination
            cross_points = np.random.rand(dimension) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimension)] = True
            child = np.where(cross_points, child, population[index])
            for name in self.visited.keys():
                self.visited[name] = False
            individual_constrain_val =self.constrain_violation_value_individual(child)
            count, combined_const_val = self.constrain_violation_value_cbrb(self.parent, child)
            const_val_child = individual_constrain_val + combined_const_val
            #print('Total paramateter check : ', count, 'Constrain violation found: ', con_val_child)

            if k > 0:
                if const_val_child == const_val:
                    # print('Two feasible solution found')
                    count = 0
                    self.rulebase_list = defaultdict()
                    fit_child = self.objective_function_fitness(child, count)
                    if fit_child < fitness_value:
                        fitness_value = fit_child
                        trial = child
                        const_val = const_val_child

                elif const_val_child < const_val:
                    trial = child
                    const_val = const_val_child
                    self.rulebase_list = defaultdict()
                    fitness_value = self.objective_function_fitness(parameters=child,count = 0)

            else:
                trial = child
                self.rulebase_list = defaultdict()
                const_val=const_val_child
                fitness_value = self.objective_function_fitness(child, count)
        print('Constrain Val = ', const_val)
        return trial

    def constrain_violation_value_individual(self, child):
        result = 0
        for each in child:
            if not 0 <= each <= 1:
                result = result + 1
        '''
        count = 0
        result = result + self.rulebase_list[self.parent].calculate_rulebase_constrain_violation_value(child, count)
        '''
        return result

    def constrain_violation_value_cbrb(self, v, parameters, count =0, constrain_val =0):
        self.visited[v] = True
        child_list = []
        parent = self.brb_tree[v]
        for child in self.graph[v]:
            node = self.brb_tree[child]
            if self.visited[child] == False:
                child_list.append(node)
                count, constrain_val = self.constrain_violation_value_cbrb(child, parameters, count, constrain_val)
        if parent.is_root == False and parent.is_input== False:
            # self.rulebase_list[v] = RuleBase(object_list,parent)
            count,constrain_val = self.rulebase_list[v].calculate_rulebase_constrain_violation_value(parameters, count, constrain_val)
        elif parent.is_root == True and parent.is_input ==False:
            count, constrain_val = self.rulebase_list[v].calculate_constrain_violation_value_cbrb_root(parameters, count, constrain_val)
        return count, constrain_val


    def diversity_machanism(self, gen, populations, fitness_value, parent_index, best_index, child):
        if gen < 1000:
            Model.Sr = Model.Sr - 3 * (Model.Sr - 0.025) / 1000
            count = 0
            self.rulebase_list = defaultdict()
            child_fiteness = self.objective_function_fitness(child, count)
            if child_fiteness < fitness_value[parent_index]:
                fitness_value[parent_index] = child_fiteness
                populations[parent_index] = child
                if child_fiteness < fitness_value[best_index]:
                    best_index = parent_index
        else:
            count,con_val_child = self.constrain_violation_value_cbrb(self.parent,child)
            print('Count Child :',count)
            count,con_val_parent = self.constrain_violation_value_cbrb(self.parent,populations[parent_index])
            print('Count_parent :',count)
            print('Constrain violation value paren', con_val_parent, 'Constrain Violation Value child: ', con_val_child)
            if (con_val_child == 0 and con_val_parent == 0) or (con_val_parent == con_val_child):
                # print('Two feasible solution found')
                fit_parent = fitness_value[parent_index]
                self.rulebase_list = defaultdict()
                count = 0
                fit_child = self.objective_function_fitness(child, count)
                if fit_child < fit_parent:
                    fitness_value[parent_index] = fit_child
                    populations[parent_index] = child
                    if fit_child < fitness_value[best_index]:
                        best_index = parent_index

            elif con_val_child < con_val_parent:
                count = 0
                self.rulebase_list = defaultdict()
                fit_child = self.objective_function_fitness(child, count)
                fitness_value[parent_index] = fit_child
                populations[parent_index] = child
                if fit_child < fitness_value[best_index]:
                    best_index = parent_index
        return best_index
    def reset_brb_tree_node_visited(self):
        for name in self.visited.keys():
            self.visited[name] = False

