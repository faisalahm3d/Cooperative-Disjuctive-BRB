from typing import List

from beliefrulebase import RuleBase
from itertools import product
import pandas
from rules import Rules
import numpy as np
from collections import defaultdict

class DynamicRuleBase(RuleBase):
    def __init__(self, object_list, parent):
        RuleBase.__init__(self, object_list, parent)
        self.rulebase_dictionary = defaultdict()
        #self.activated_rules_key_list = list()
        self.number_of_activated_rule = 0
        self.activated_rules_keys_list = list()

    #Function to create initial rulebase from csv file provided by expert
    def create_initial_rulebase(self):
        file_name = self.parent.rulebase_filename
        rulebase_frame = pandas.read_csv(file_name, sep=";", engine='python', header=None)
        rulebase = rulebase_frame.values
        com = []
        for i in range(len(self.obj_list)):
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        self.combinations = combination

        for count, combination in enumerate(self.combinations):
            rule = Rules()
            rule.rule_weight = rulebase[count, 0]
            rule.consequence_val = rulebase[count, 1:self.num_consequence_term + 1]
            rule.combinations = combination
            #self.rule_row_list.append(rule)
            #dictionary to store each rule in the rulebase. Key is the combination of referential value of antecedent attributes
            self.rulebase_dictionary[combination] = rule

    #function to activation weight calculation only for the active rule
    '''
    def rule_activation_weight_calculation(self):
        max_attribute_weight = 0
        self.number_of_activated_rule = 0
        # find the maximum attribute weight
        list_activated_ref = list()
        for each in self.obj_list:
            if each.attribute_weight > max_attribute_weight:
                max_attribute_weight = each.attribute_weight
            #list_activated_ref.append(each.activated_ref_point)
        #self.activated_rules_keys_list = list(product(*list_activated_ref))

        sum_matching_degree = 0
        for key in self.combinations:
            rule_matching_degree = 1
            for attribute_index, attribute in enumerate(self.obj_list):
                rule_matching_degree *= pow(attribute.transformed_val[key[attribute_index]],
                                            attribute.attribute_weight / max_attribute_weight)
            rule = self.rulebase_dictionary[key]
            rule.matching_degree = rule_matching_degree * rule.rule_weight
            self.rulebase_dictionary[key]= rule
            # rule.matching_degree = rule_matching_degree

            if not rule_matching_degree == 0:
                self.number_of_activated_rule += 1
            sum_matching_degree += rule.matching_degree
        
        for key in self.activated_rules_keys_list:
            rule = self.rulebase_dictionary[key]
            rule.activation_weight = rule.matching_degree / sum_matching_degree
            self.rulebase_dictionary[key]= rule
        #print('Total Activated Rule = ', self.number_of_activated_rule)
        return
'''
    #Evidential Reasoning For Regression Problem conjuctive brb
    def rules_aggregation_analytical(self):
        b = [0 for _ in range(len(self.activated_rules_keys_list))]
        a = [0 for _ in range(len(self.activated_rules_keys_list))]
        c = [0 for _ in range(len(self.activated_rules_keys_list))]
        final_consequence = [0 for _ in range(len(self.con_ref_values))]
        product_b = 1
        product_a = [1 for _ in range(len(self.con_ref_values))]
        product_c = 1
        sum_product_a = 0

        for i, key in enumerate(self.activated_rules_keys_list):
            c[i] = 1 - (float(self.rulebase_dictionary[key].activation_weight))
            product_c *= c[i]

            for j in range(len(self.rulebase_dictionary[key].consequence_val)):
                b[i] += self.rulebase_dictionary[key].consequence_val[j] * self.rulebase_dictionary[key].activation_weight
            b[i] = 1 - b[i]
            product_b *= b[i]

            for j in range(len(self.rulebase_dictionary[key].consequence_val)):
                product_a[j] *= self.rulebase_dictionary[key].consequence_val[j] * self.rulebase_dictionary[key].activation_weight + b[i]

        for j in range(len(product_a)):
            sum_product_a += product_a[j]
        for j in range(len(product_a)):
            try:
                final_consequence[j] = (product_a[j] - product_b) / (
                        sum_product_a - ((len(self.con_ref_values)) - 1) * product_b - product_c)
            except ZeroDivisionError:
                print(product_a,product_b,product_c)
        self.parent.transformed_val = final_consequence

        return final_consequence

    # Evidential Reasoning For Regression Problem for disjuctive brb
    def disjuctive_rules_aggregation_analytical(self):
        b = [0 for _ in range(len(self.rule_row_list))]
        a = [0 for _ in range(len(self.rule_row_list))]
        c = [0 for _ in range(len(self.rule_row_list))]
        final_consequence = [0 for _ in range(len(self.con_ref_values))]
        product_b = 1
        product_a = [1 for _ in range(len(self.con_ref_values))]
        product_c = 1
        sum_product_a = 0

        for i, key in enumerate(self.rule_row_list):
            c[i] = 1 - (float(self.rule_row_list[i].activation_weight))
            product_c *= c[i]

            for j in range(len(self.rule_row_list[i].consequence_val)):
                b[i] += self.rule_row_list[i].consequence_val[j] * self.rule_row_list[
                    i].activation_weight
            b[i] = 1 - b[i]
            product_b *= b[i]

            for j in range(len(self.rule_row_list[i].consequence_val)):
                product_a[j] *= self.rule_row_list[i].consequence_val[j] * self.rule_row_list[
                    i].activation_weight + b[i]

        for j in range(len(product_a)):
            sum_product_a += product_a[j]
        for j in range(len(product_a)):
            try:
                final_consequence[j] = (product_a[j] - product_b) / (
                        sum_product_a - ((len(self.con_ref_values)) - 1) * product_b - product_c)
            except ZeroDivisionError:
                print(product_a, product_b, product_c)
        self.parent.transformed_val = final_consequence

        return final_consequence

    #implementation of ER-C(Evedential Reasoning for Classification)

    def rule_aggregation_for_classification(self):
        b = [0 for _ in range(len(self.activated_rules_keys_list))]
        final_consequence = [0 for _ in range(len(self.con_ref_values))]
        product_b = 1
        product_a = [1 for _ in range(len(self.con_ref_values))]

        for i, key in enumerate(self.activated_rules_keys_list):
            for j in range(len(self.rulebase_dictionary[key].consequence_val)):
                b[i] += self.rulebase_dictionary[key].consequence_val[j] * self.rulebase_dictionary[key].activation_weight
            b[i] = 1 - b[i]
            for j in range(len(self.rulebase_dictionary[key].consequence_val)):
                product_a[j] *= self.rulebase_dictionary[key].consequence_val[j] * self.rulebase_dictionary[key].activation_weight + b[i]
        self.parent.transformed_val = product_a

        return self.parent.transformed_val

    #Function to create rulebase using random number
    def create_belief_rulebase(self):
        com: List[List[int]] = []
        for i in range(len(self.obj_list)):
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        self.combinations = combination
        for count, combination in enumerate(self.combinations):
            rule = Rules()
            rule.rule_weight = np.random.uniform(0, 1)
            temp = np.random.uniform(0, 1, self.num_consequence_term)
            rule.consequence_val = temp / np.sum(temp)
            rule.combinations = combination
            #self.rule_row_list.append(rule)
            self.rulebase_dictionary[combination] = rule


    #Funtion to create rulebase from parameter provided as numpy array
    def assign_belief_degree_rulebase(self, parameters, count):
        for idx, each in enumerate(self.obj_list):
            each.attribute_weight = parameters[count]
            count = count + 1

        for index in range(len(self.obj_list[0].ref_val)):
            com = []
            for i in range(len(self.obj_list)):
                com.append(index)
            self.combinations.append(com)

        for idx, combination in enumerate(self.combinations):
            rule = Rules()
            rule.rule_weight = parameters[count]
            count = count + 1
            rule.consequence_val = parameters[count:count + self.num_consequence_term]
            count = count + self.num_consequence_term
            rule.combinations = combination
            self.rule_row_list.append(rule)
        return count

    #Function to check whether the parameters list satisfied and resolve the problem generating random number

    def check_rulebase_constrain(self, parameters, count):
        for each in enumerate(self.obj_list):
            '''
            if 0<=parameters[count]<=1:
                count = count+1
            else:
                parameters[count]= np.random.uniform(0,1)
                count = count+1
                '''
            count = count + 1
        for i in range(len(self.combinations)):
            '''
            if 0<=parameters[count]<=1:
                count =count+1
            else:
                parameters[count]=np.random.uniform(0,1)
                count = count+1
                '''
            count = count + 1
            if parameters[count:count + self.num_consequence_term].sum() == 1:
                count = count + self.num_consequence_term
            else:
                #print(count)
                temp = np.random.random(self.num_consequence_term)
                temp /= temp.sum()
                try:
                    parameters[count:count + self.num_consequence_term] = temp
                except:
                    print(temp)
                count = count + self.num_consequence_term
        return parameters

    # calculate total number of violated equality constrain
    def calculate_rulebase_constrain_violation_value(self, parameters, count, constrain_violation):
        for each in enumerate(self.obj_list):
            count = count + 1
        for each in range(len(self.combinations)):
            count = count + 1
            if 0 <= parameters[count:count + self.num_consequence_term].sum() <= 1:
                count = count + self.num_consequence_term
            else:
                #print(count)
                constrain_violation = constrain_violation + 1
                count = count + self.num_consequence_term
        return count, constrain_violation

    # Funtion to create rulebase as list from random number
    def create_population_belief_rulebase(self):
        com = []
        training_parameter = []
        for i in range(len(self.obj_list)):
            training_parameter.append(np.random.uniform(0, 1))
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        # self.combinations = combination
        for count, combination in enumerate(combination):
            # rule =Rules()
            training_parameter.append(np.random.uniform(0, 1))
            temp = np.random.random(self.num_consequence_term)
            temp /= temp.sum()
            training_parameter = training_parameter + list(temp)
        return training_parameter

    def create_population_rulebase(self):
        com = []
        training_parameter = []

        for i, node in enumerate(self.obj_list):
            temp = np.random.uniform(0, 1)
            # node.attribute_weight = temp
            training_parameter.append(temp)
            ''''
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        self.combinations = combination
        '''
        for count, combination in enumerate(self.combinations):
            # rule = Rules()
            temp = np.random.uniform(0, 1)
            # rule.rule_weight = temp
            training_parameter.append(temp)
            temp = np.random.random(self.num_consequence_term)
            temp /= temp.sum()
            # rule.consequence_val = temp
            training_parameter = training_parameter + list(temp)
            # rule.combinations = combination
            # self.rule_row_list.append(rule)
            # self.rulebase_dictionary[combination] = rule
        return training_parameter

    def create_population_rulebase_cooperative_brb_root(self):
        temp = np.random.uniform(0, 1, len(self.obj_list))
        rule_weight = temp / np.sum(temp)
        '''
        for idx, value in enumerate(self.obj_list):
            rule = Rules()
            rule.activation_weight = rule_weight[idx]
            #rule.consequence_val = value.transformed_val
            #self.rule_row_list.append(rule)
            self.rulebase_dictionary[(1,idx)]= rule
            self.activated_rules_keys_list.append((1,idx))
        '''
        return list(rule_weight)

    def assign_cbrb_weight_coefficient(self, training_parameter, count):
        for idx, value in enumerate(self.obj_list):
            rule = Rules()
            rule.activation_weight = training_parameter[count]
            count = count + 1
            #rule.consequence_val = value.transformed_val
            self.rule_row_list[idx] = rule
            #self.activated_rules_keys_list.append((1,idx))
        return count

    def assign_root_consequence_belief_distribution(self):
        for idx, value in enumerate(self.obj_list):
            self.rule_row_list[idx].consequence_val = value.transformed_val

    def evaluate_objective_func(self, training_data, n_training_instance):
        absolute_error = 0
        for index in range(n_training_instance):
            for each in self.obj_list:
                each.input_val = training_data[each.antecedent_id][index]
                each.input_transformation_optimize()

            self.rule_activation_weight_calculation()
            #self.rule_aggregation_for_classification()
            self.disjuctive_rules_aggregation_analytical()
            predicted_class = np.argmax(self.parent.transformed_val)
            predicted_class += 1
            actual_class = training_data[self.parent.parent][index]
            if not actual_class == predicted_class:
                absolute_error += 1
        mean_absolute_error = absolute_error/n_training_instance
        print('MAE :',mean_absolute_error)
        self.parent.mean_absolute_error = mean_absolute_error
        return

    def evaluate_objective_func_root(self, training_parameter, count):
        fitness = 0
        for each in self.obj_list:
            fitness += each.mean_absolute_error*training_parameter[count]
            count += 1
        self.parent.mean_absolute_error = fitness

    def create_antecedent_part_rulebase(self):
        com = []
        for i, node in enumerate(self.obj_list):
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        self.combinations = combination


    def create_disjuctive_brb_antecedent_part(self):

        for index in range(len(self.obj_list[0].ref_val)):
            com = []
            for i in range(len(self.obj_list)):
                com.append(index)
            self.combinations.append(com)



    def check_constrain_child(self, parameters, count):
        for each in enumerate(self.obj_list):
            '''
            if 0<=parameters[count]<=1:
                count = count+1
            else:
                parameters[count]= np.random.uniform(0,1)
                count = count+1
                '''
            count = count + 1
        for i in range(len(self.combinations)):
            '''
            if 0<=parameters[count]<=1:
                count =count+1
            else:
                parameters[count]=np.random.uniform(0,1)
                count = count+1
                '''
            count = count + 1
            if 0 <= parameters[count:count + self.num_consequence_term].sum() <= 1:
                count = count + self.num_consequence_term
            else:
                #print(count)
                temp = np.random.random(self.num_consequence_term)
                temp /= temp.sum()
                parameters[count:count + self.num_consequence_term] = temp
                count = count + self.num_consequence_term
        return count

    def rule_activation_weight_calculation(self):
        max_attribute_weight = 0
        numberOfTotalActivatedWeight = 0

        # find the maximum attribute weight
        for each in self.obj_list:
            if each.attribute_weight > max_attribute_weight:
                max_attribute_weight = each.attribute_weight

        sum_matching_degree = 0
        for rule_index, rule in enumerate(self.rule_row_list):
            rule_matching_degree = 0
            for attribute_index, attribute in enumerate(self.obj_list):
                rule_matching_degree += pow(attribute.transformed_val[rule.combinations[attribute_index]],
                                            attribute.attribute_weight / max_attribute_weight)
            rule.matching_degree = rule_matching_degree * rule.rule_weight
            # rule.matching_degree = rule_matching_degree
            if not rule_matching_degree == 0:
                numberOfTotalActivatedWeight += 1
            sum_matching_degree += rule.matching_degree
        if numberOfTotalActivatedWeight > 6:
            print('error')

        for rule_index, rule in enumerate(self.rule_row_list):
            rule.activation_weight = rule.matching_degree / sum_matching_degree
        print('Total Activated Rule DBRB= ', numberOfTotalActivatedWeight)
        return
















