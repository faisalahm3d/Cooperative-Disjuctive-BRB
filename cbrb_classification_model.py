from cooperative_brb_model import Model
import numpy as np
from dynamic_belief_rulebase import DynamicRuleBase
from collections import defaultdict


# This class was subclass of DynamicModel in original file
class ClassificationModel(Model):
    def __init__(self, name, attribute):
        self.rubase_dictionary = defaultdict()
        # store best population of each generation
        self.generation_wise_best_population = list()
        # store best fitness of each generation
        self.generation_wise_fitness = list()
        # hold current best population
        self.current_best_population = ()
        # hold fitness of current best population
        self.current_best_population_fitness = 1000
        Model.__init__(self, file_name=name, attributes=attribute)


    # function to measure accuracy on test data
    def test(self, test_data):
        row, column = test_data.shape
        self.test_data_size = row
        for index, name in enumerate(self.attributes_names):
            self.test_data[name] = test_data[0:row, index]
        result = []
        absolute_error = 0
        # self.reset_brb_tree_node_visited()
        self.rulebase_list = defaultdict()
        for i in range(self.test_data_size):
            count = 0
            for name in self.visited.keys():
                self.visited[name] = False
            self.inference_test(self.parent, self.current_best_population_fitness, count=count, index=i)
            # self.inference_test(self.parent,i)
            parent = self.brb_tree[self.parent]
            predicted_class = np.argmax(parent.transformed_val) + 1
            if not parent.input_val == predicted_class:
                absolute_error += 1
            result.append(predicted_class)
        return result, absolute_error / self.test_data_size

    # This function test the trained model on test data
    def inference_test(self, v, opmized_parameter, count=None, index=None, ):
        self.visited[v] = True
        object_list = []
        parent = self.brb_tree[v]
        for i in self.graph[v]:
            node = self.brb_tree[i]
            if self.visited[i] == False:
                object_list.append(node)
                count = self.inference_train_fitness(i, opmized_parameter, count=count, index=index)
        if parent.is_input == False and parent.is_root == True:
            new_input = self.training_data[parent.antecedent_id][index]
            parent.input_val = new_input
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = DynamicRuleBase(object_list, parent)
                # self.rulebase_list[v].assign_belief_degree_rulebase(training_parameter, count)
                count = self.rulebase_list[v].assign_cbrb_weight_coefficient(opmized_parameter, count)
            # self.rulebase_list[v].activation_weight()
            self.rulebase_list[v].assign_root_consequence_belief_distribution()
            # res = self.rulebase_list[v].rules_aggregation_analytical()
            res = self.rulebase_list[v].disjuctive_rules_aggregation_analytical()
            # print(res)
        if parent.is_input == False and parent.is_root == False:
            # new_input = self.training_data[parent.antecedent_id][index]
            # parent.input_val = new_input
            if self.rulebase_list.get(v) == None:
                self.rulebase_list[v] = DynamicRuleBase(object_list, parent)
                count = self.rulebase_list[v].assign_belief_degree_rulebase(opmized_parameter, count)
            self.rulebase_list[v].rule_activation_weight_calculation()
            # res = self.rulebase_list[v].rules_aggregation_analytical()
            res = self.rulebase_list[v].disjuctive_rules_aggregation_analytical()
            # print(res)
        elif parent.is_input == True:
            new_input = self.test_data[parent.antecedent_id][index]
            parent.input_val = new_input
            parent.input_transformation_optimize()
        return count

    def objective_function_ccde(self, root, training_parameter, count, single_brb=False):
        self.visited[root] = True
        object_list = []
        parent = self.brb_tree[root]
        if root in self.graph:
            for child in self.graph[root]:
                node = self.brb_tree[child]
                if not self.visited[child]:
                    object_list.append(node)
                    count = self.objective_function_ccde(child, training_parameter,count,single_brb)

        if not parent.is_input and parent.is_root:
            if root not in self.rulebase_list:
                self.rulebase_list[root] = DynamicRuleBase(object_list, parent)
                # self.rulebase_list[v].assign_belief_degree_rulebase(training_parameter, count)
                # count = self.rulebase_list[root].assign_cbrb_weight_coefficient(training_parameter, count)
            # self.rulebase_list[root].assign_root_consequence_belief_distribution()
            # res = self.rulebase_list[v].rules_aggregation_analytical()
            # res = self.rulebase_list[root].rule_aggregation_for_classification()
            self.rulebase_list[root].evaluate_objective_func_root(training_parameter=training_parameter, count=count)
            # print(res)
        if not parent.is_input and not parent.is_root:
            # new_input = self.training_data[parent.antecedent_id][index]
            # parent.input_val = new_input
            if root not in self.rulebase_list:
                self.rulebase_list[root] = DynamicRuleBase(object_list, parent)
            if single_brb:
                count = self.rulebase_list[root].assign_belief_degree_rulebase(training_parameter, count)
                if self.rubase_dictionary[root]:

                    self.rulebase_list[root].evaluate_objective_func(self.training_data,self.training_data_size)
                    self.rubase_dictionary[root] = False
            else:
                count = self.rulebase_list[root].assign_belief_degree_rulebase(training_parameter, count)
                self.rulebase_list[root].evaluate_objective_func(self.training_data, self.training_data_size)

            # self.rulebase_list[root].rule_activation_weight_calculation()
            # res = self.rulebase_list[v].rules_aggregation_analytical()
            # res = self.rulebase_list[root].rule_aggregation_for_classification()
            # print(res)
        '''
        elif parent.is_input == True:
            parent.input_transformation_optimize()
        '''
        return count

    def cooperative_coevalutionary_optimization(self, n_iteration, maximum_func_eval, n_population, min_fitness):
        test_generation_number = 10
        population_list = []
        #current_best_population = None

        # loop to generate initial n_population and find the best population according to the objective function
        for it in range(n_population):
            self.rulebase_list = defaultdict()
            self.reset_brb_tree_node_visited()
            pop = []
            pop = self.population_generator(v=self.parent, training_parameter=pop)
            # Does it necessary? may be move this to the outer part of the loop
            #self.rulebase_list = defaultdict()
            self.reset_brb_tree_node_visited()
            self.objective_function_ccde(self.parent, training_parameter=pop, count=0)
            fitness = self.brb_tree[self.parent].mean_absolute_error
            print(fitness)
            population: Population = Population(pop=pop, fitness=fitness)
            population_list.append(population)
            if it < 1:
                self.current_best_population = tuple(population.pop)
                self.current_best_population_fitness = population.fitness
            else:
                if population.fitness < self.current_best_population_fitness:
                    # self.current_best_population = population
                    self.current_best_population = tuple(population.pop)
                    self.current_best_population_fitness = population.fitness

        func_eval = n_population
        flag = 0
        self.generation_wise_best_population.append(self.current_best_population)
        self.generation_wise_fitness.append(self.current_best_population_fitness)

        # initially set all BRB to visitable
        for key in self.rulebase_list.keys():
            self.rubase_dictionary[key] = True

        # while func_eval < maximum_func_eval:

        for test_gen in range(test_generation_number):
            parameter_index = 0

            # Loop to optimize individual brb
            for key in self.rulebase_list.keys():
                sub_population_list = []
                print('Optimization_start for rulebase :', key)
                for rul_key in self.rulebase_list.keys():
                    self.rubase_dictionary[rul_key] = True

                # find the start index and stop index of parameter to be optimized
                rulebase = self.rulebase_list[key]
                if not key == self.parent:
                    total_param = len(rulebase.obj_list) + len(rulebase.combinations) * (
                        len(rulebase.parent.ref_val) + 1)
                else:
                    total_param = len(rulebase.obj_list)
                last_index = parameter_index + total_param

                # loop to create sub-population for key brb from initial population
                # cancatinating best population with sub-population of key brb the in the initial population
                for j in range(n_population):
                    # set only the brb to be optimized visitable
                    self.rubase_dictionary[key] = True
                    # rulebase = self.rulebase_list[key] total_param = len(rulebase.obj_list) + len(
                    # rulebase.combinations) * (len(rulebase.parent.ref_val) + 1) last_index =
                    # parameter_index+total_param
                    eval_pos = list(self.current_best_population)
                    # evalPos[parameter_index:total_param] = population_list[j].get_pop()[parameter_index:total_param]
                    sub_pop = population_list[j].get_pop()[parameter_index:last_index]
                    eval_pos[parameter_index:last_index] = sub_pop
                    self.reset_brb_tree_node_visited()
                    self.objective_function_ccde(root=self.parent, training_parameter=eval_pos, count=0)
                    sub_pop_fitness = self.brb_tree[self.parent].mean_absolute_error
                    print(sub_pop_fitness)
                    sub_population_list.append(Population(sub_pop, sub_pop_fitness))

                # function call to optimize individual brb using differential evaluation
                # return: best = the current best population
                #          n_sub_pop: the best population for the single brb
                best, n_sub_pop = self.self_adaptive_differential_evaluation(rulebase, sub_population_list, n_iteration, n_population, parameter_index, last_index)

                # current_best_population = best
                for j in range(n_population):
                    population_list[j].get_pop()[parameter_index:last_index] = n_sub_pop[j].get_pop()
                    population_list[j].set_fitness(n_sub_pop[j].fitness)
                parameter_index = last_index

                print('Optimization finished for rulebase ', key)
                # break
            # break
        for value in self.generation_wise_fitness:
            print(value)
        return self.current_best_population_fitness

    def differential_evaluation(self, rulebase, sub_populations, its, popsize, start_index, last_index):
        best_population_per_generation = []

        for iteration in range(its):
            print('Generation :', iteration, end='\n')
            for j in range(popsize):
                # mutation and recombination original
                pop_list = [each.get_pop() for each in sub_populations]
                pop_list = np.asarray(pop_list)

                #trial = self.mutation_recombination_original(pop_list, 0.9, 0.5, j)
                # mutation and recombination modified
                c_best = self.current_best_population[start_index:last_index]
                c_best = np.asarray(c_best)

                trial = self.modified_mutation_recombination(population=pop_list, index=j, best_solution=c_best)

                '''
                if iteration < 300:
                    trial = self.extended_de(sub_populations, dimensions, j, best)
                    # trial = self.mutation_recombination_original(populations,0.8,0.9,j)
                else:
                    trial = self.modified_mutation_recombination(sub_populations, j, best_solution=best)
                    # print('original mutation')
                    # trial = self.mutation_recombination_original(populations,crossp,mut,j)
                '''
                # handle bounding constrain
                trial = np.clip(trial, 0, 1)
                # handle equality constrain
                if (rulebase.parent.antecedent_id == self.parent):
                    trial = rulebase.check_cbrb_rulebase_constrain_root(trial, count=0)
                else:
                    trial = rulebase.check_rulebase_constrain(parameters=trial, count=0)
                count = 0
                self.reset_brb_tree_node_visited()
                #count = self.check_constrain_cbrb(self.parent, trial)
                # print('Index of paramater whose constrain is not resolved')
                # individual_cost = self.constrain_violation_value_individual(trial)
                # self.reset_brb_tree_node_visited()
                # count, combined_const = self.constrain_violation_value_cbrb(self.parent, trial)
                # const_val = individual_cost+combined_const
                # print('Constrain violation :', const_val)

                # print(trial2)
                # best_idx = self.diversity_machanism(i, populations, fitness_values, j, best_idx, trial)

                # Selection using original differential evalution

                count = 0
                #self.rulebase_list = defaultdict()
                trial_population = self.current_best_population
                trial_population = list(trial_population)
                try:
                    trial_population[start_index:last_index] = trial
                except:
                    print('What')
                self.rubase_dictionary[rulebase.parent.antecedent_id] = True
                self.objective_function_ccde(self.parent, trial_population, count=count, single_brb=True)
                trial_fitness = self.brb_tree[self.parent].mean_absolute_error
                print(trial_fitness)
                if trial_fitness < sub_populations[j].fitness:
                    sub_populations[j].fitness = trial_fitness
                    sub_populations[j].pop = list(trial)
                    if trial_fitness < self.current_best_population_fitness:
                        #best_idx = j
                        self.current_best_population = tuple(trial_population)
                        self.current_best_population_fitness = trial_fitness
            # print(fitness_values[best_idx])
            # yield  fitness_values[best_idx]
            #best_population_per_generation.append(current_best_pop)
            self.generation_wise_best_population.append(self.current_best_population)
            self.generation_wise_fitness.append(self.current_best_population_fitness)
            print('Error for generation ', iteration, ' is ', self.current_best_population_fitness)
        return self.current_best_population, sub_populations

    def fit(self, training_data):
        row, column = training_data.shape
        self.training_data_size = row
        for index, name in enumerate(self.attributes_names):
            self.training_data[name] = training_data[0:row, index]

        best_fitness =self.cooperative_coevalutionary_optimization(n_iteration = 50, maximum_func_eval=30000,
                                                     n_population = 20, min_fitness = 0.001)
        return self.generation_wise_fitness

    def generate_population(self, popsize=20):
        return

    def population_generator(self, v, training_parameter):
        self.visited[v] = True
        object_list = []
        node = self.brb_tree[v]
        # object_list.append(parent)
        if v in self.graph:
            for i in self.graph[v]:
                child = self.brb_tree[i]
                if not self.visited[i]:
                    object_list.append(child)
                    training_parameter = self.population_generator(i, training_parameter=training_parameter)
        if not node.is_input and node.is_root:
            if v not in self.rulebase_list:
                self.rulebase_list[v] = DynamicRuleBase(object_list, node)
                training_parameter = training_parameter + self.rulebase_list[
                    v].create_population_rulebase_cooperative_brb_root()
                # print(training_parameter)
        elif not node.is_input and not node.is_root:
            if v not in self.rulebase_list:
                self.rulebase_list[v] = DynamicRuleBase(object_list, node)
                self.rulebase_list[v].create_disjuctive_brb_antecedent_part()
                training_parameter = training_parameter + self.rulebase_list[v].create_population_rulebase()
                # print(training_parameter)

        return training_parameter

    def self_adaptive_differential_evaluation(self, rulebase, sub_populations, its, popsize, start_index, last_index):
        best_population_per_generation = []

        #parameter for SaDE
        probability_strategy1 = 0.5
        probability_strategy2 = 1-probability_strategy1
        learning_period = 50
        generation_tracker = 0
        ns1 = 0
        ns2 = 0
        nf1 = 0
        nf2 = 0
        random_number_vector = np.random.uniform(0, 1, popsize)
        stategy_tracker = 0

        for iteration in range(its):
            print('Generation :', iteration, end='\n')
            for j in range(popsize):
                # mutation and recombination original
                pop_list = [each.get_pop() for each in sub_populations]
                pop_list = np.asarray(pop_list)
                if random_number_vector[j] <= probability_strategy1:
                    trial = self.mutation_recombination_original(pop_list, 0.9, 0.5, j)
                    stategy_tracker = 1
                else:
                    # mutation and recombination modified
                    c_best = self.current_best_population[start_index:last_index]
                    c_best = np.asarray(c_best)
                    trial = self.modified_mutation_recombination(population=pop_list, index=j, best_solution=c_best)
                    stategy_tracker = 2

                # handle bounding constrain
                trial = np.clip(trial, 0, 1)
                # handle equality constrain
                if rulebase.parent.antecedent_id == self.parent:
                    trial = rulebase.check_cbrb_rulebase_constrain_root(trial, count=0)
                else:
                    trial = rulebase.check_rulebase_constrain(parameters=trial, count=0)
                count = 0
                self.reset_brb_tree_node_visited()
                #count = self.check_constrain_cbrb(self.parent, trial)
                # print('Index of paramater whose constrain is not resolved')
                # individual_cost = self.constrain_violation_value_individual(trial)
                # self.reset_brb_tree_node_visited()
                # count, combined_const = self.constrain_violation_value_cbrb(self.parent, trial)
                # const_val = individual_cost+combined_const
                # print('Constrain violation :', const_val)

                # print(trial2)
                # best_idx = self.diversity_machanism(i, populations, fitness_values, j, best_idx, trial)

                # Selection using original differential evalution

                count = 0
                # self.rulebase_list = defaultdict()
                trial_population = self.current_best_population
                trial_population = list (trial_population)
                try:
                    trial_population[start_index:last_index] = trial
                except:
                    print('What')
                self.rubase_dictionary[rulebase.parent.antecedent_id] = True
                self.objective_function_ccde(self.parent, trial_population, count=count)
                trial_fitness = self.brb_tree[self.parent].mean_absolute_error
                print(trial_fitness)
                if trial_fitness < sub_populations[j].fitness:
                    sub_populations[j].fitness = trial_fitness
                    sub_populations[j].pop = list(trial)
                    if trial_fitness < self.current_best_population_fitness:
                        #best_idx = j
                        self.current_best_population = tuple(trial_population)
                        self.current_best_population_fitness = trial_fitness
                    if stategy_tracker == 1:
                        ns1 += 1
                    else:
                        ns2 += 1
                else:
                    if stategy_tracker == 1:
                        nf1 += 1
                    else:
                        nf2 += 2
            # print(fitness_values[best_idx])
            # yield  fitness_values[best_idx]
            #best_population_per_generation.append(current_best_pop)
            self.generation_wise_best_population.append(self.current_best_population)
            self.generation_wise_fitness.append(self.current_best_population_fitness)
            print('Error for generation ', iteration, ' is ', self.current_best_population_fitness)
            generation_tracker += 1
            if generation_tracker == learning_period:
                try:
                    probability_strategy1 = (ns1*(ns2+nf2))/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
                except:
                    probability_strategy1 = np.random.uniform(0, 1, 1)
                probability_strategy2 = 1-probability_strategy1
                ns1 = ns2 = nf1 = nf2 = 0
                generation_tracker = 0

        return self.current_best_population, sub_populations



class Population(object):
    def __init__(self, pop, fitness):
        self.pop = pop
        self.fitness = fitness

    def get_pop(self):
        return self.pop

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_pop(self, pop):
        self.pop = pop
