import json
from collections import OrderedDict
import logging
import pandas

import math

from rules import Rules
# from openpyxl import load_workbook
from data import Data
from itertools import product
import numpy as np


# with open('data.json') as file_data:
# with open('single_tree.json') as file_data:
#     data = json.load(file_data, object_pairs_hook=OrderedDict)


class RuleBase(object):
    def __init__(self, object_list, parent):
        self.obj_list = object_list
        self.intermediate_ref_val = 0
        self.rule_row_list = list()
        self.parent = parent
        self.num_consequence_term = len(self.parent.ref_val)
        self.con_ref_values = [0 for _ in range(len(self.parent.ref_val))]
        self.combinations = []
        self.consequence_rating_count = [0 for _ in range(len(self.parent.ref_val))]

    def create_belief_rulebase(self):
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
            rule.rule_weight = np.random.uniform(0, 1)
            temp = np.random.uniform(0, 1, self.num_consequence_term)
            rule.consequence_val = temp / np.sum(temp)
            rule.combinations = combination
            self.rule_row_list.append(rule)

    def assign_belief_degree_rulebase(self, parameters, count):
        com = []
        for idx, each in enumerate(self.obj_list):
            each.attribute_weight = parameters[count]
            count = count + 1
            a = []
            for j in range(len(self.obj_list[idx].ref_val)):
                a.append(j)
            com.append(a)
        combinations = list(product(*com))
        self.combinations = combinations
        for idx, combination in enumerate(combinations):
            rule = Rules()
            rule.rule_weight = parameters[count]
            count = count + 1
            rule.consequence_val = parameters[count:count + self.num_consequence_term]
            count = count + self.num_consequence_term
            rule.combinations = combination
            self.rule_row_list.append(rule)
        return count

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
        for each in self.rule_row_list:
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
                print(count)
                temp = np.random.random(self.num_consequence_term)
                temp /= temp.sum()
                parameters[count:count + self.num_consequence_term] = temp
                count = count + self.num_consequence_term
        return count

    def check_cbrb_rulebase_constrain_root(self, parameters, count):
        if not parameters[count:count + len(self.obj_list)].sum() == 1:
            # print(count)
            temp = np.random.random(len(self.obj_list))
            temp /= temp.sum()
            parameters[count:count + len(self.obj_list)] = temp
        return parameters

    def calculate_rulebase_constrain_violation_value(self, parameters, count, constrain_violation):
        for each in enumerate(self.obj_list):
            count = count + 1

        for each in self.rule_row_list:
            count = count + 1
            if 0 <= parameters[count:count + self.num_consequence_term].sum() <= 1:
                count = count + self.num_consequence_term
            else:
                print(count)
                constrain_violation = constrain_violation + 1
                count = count + self.num_consequence_term
        return count, constrain_violation

    def calculate_constrain_violation_value_cbrb_root(self, parameters, count, constrain_violation):
        attribute_number = len(self.obj_list)
        if not 0 <= parameters[count:count + attribute_number].sum() <= 1:
            constrain_violation = constrain_violation + 1
            #print(count)
        return count + attribute_number, constrain_violation

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
            node.attribute_weight = temp
            training_parameter.append(temp)
            a = []
            for j in range(len(self.obj_list[i].ref_val)):
                a.append(j)
            com.append(a)
        combination = list(product(*com))
        self.combinations = combination
        for count, combination in enumerate(self.combinations):
            rule = Rules()
            temp = np.random.uniform(0, 1)
            rule.rule_weight = temp
            training_parameter.append(temp)
            temp = np.random.random(self.num_consequence_term)
            temp /= temp.sum()
            rule.consequence_val = temp
            training_parameter = training_parameter + list(temp)
            rule.combinations = combination
            self.rule_row_list.append(rule)
        return training_parameter

    def generate_extended_belief_rule_base(self):
        wb = load_workbook('DataCenter.xlsx')
        ws = wb.active
        n = ws.max_row
        col = []

        for i in range(len(self.obj_list)):
            col.append(self.obj_list[i].name)
        for row in range(1, n + 1):
            rule = Rules()
            # antecedent_dist = list()
            for idx, column in enumerate(col):
                cell_name = "{}{}".format(column, row)
                input_value = ws[cell_name].value
                self.user_input_transformation(idx, input_value)
                # antecedent_dist.append(self.obj_list[idx].transformed_val)
                rule.antecedents_belief_dist.append(self.obj_list[idx].transformed_val)
            # rule.antecedents_belief_dist.append(antecedent_dist)
            parent_cell_name = "{}{}".format(self.parent.name, row)
            consequent_input_value = ws[parent_cell_name].value
            self.user_input_transformation(None, consequent_input_value)
            rule.consequence_belief_dist = self.parent.transformed_val
            self.rule_row_list.append(rule)
        return self.rule_row_list

    def user_input_transformation(self, attribute_index, attribute_input_value):
        if attribute_index is None:
            attribute = self.parent
        else:
            attribute = self.obj_list[attribute_index]
        attribute.transformed_val = [0 for _ in range(len(attribute.ref_val))]
        for i in range(len(attribute.ref_val) - 1):
            if (float(attribute.ref_val[i]) > attribute_input_value) and (
                    attribute_input_value > float(attribute.ref_val[i + 1])):
                val_1 = (
                        (float(attribute.ref_val[i]) - attribute_input_value) / (
                        float(attribute.ref_val[i]) - float(attribute.ref_val[i + 1]))
                )
                attribute.transformed_val[i + 1] = val_1
                val_2 = 1 - val_1
                attribute.transformed_val[i] = val_2

    def individual_matching_degree(self):
        for rule_index, rule in enumerate(self.rule_row_list):
            # attribute_individual_matching = list()
            # antecedent_belief_dist = current_rule.antecedents_belief_dist
            for attribute_index, belief_dist in enumerate(rule.antecedents_belief_dist):
                # antecedent_attribute = self.obj_list[attribute_index]
                temp = 0
                for index, belief in enumerate(belief_dist):
                    temp += pow((self.obj_list[attribute_index].transformed_val[index] - belief), 2)
                individual_matching = 1 - math.sqrt(temp / 2)
                rule.attributes_individual_matching.append(individual_matching)
        # attribute_individual_matching.append(individual_matching)

    # rule.attribute_individual_matching.append(attribute_individual_matching)

    '''
    Transform input value in the range of consequent values
                each.transformed_val[j + 1] = str(val_1)
    '''

    def input_transformation(self):
        for each in self.obj_list:
            # print("Input value for {} is {}".format(each.name, each.input_val))
            # print "Value before input transformation: {}".format(each.transformed_val)
            try:
                user_input = float(each.input_val)
            # user_input = float(each.crisp_val)
            except:
                user_input = 0

            if user_input > float(each.ref_val[0]):
                user_input = float(each.ref_val[0])
            elif user_input < float(each.ref_val[len(each.ref_val) - 1]):
                user_input = float(each.ref_val[len(each.ref_val) - 1])
            flag = False
            for i in range(len(each.ref_val)):
                if user_input == float(each.ref_val[i]):
                    each.transformed_val[i] = 1
                    flag = True
                    break
            if not flag:
                for j in range(len(each.ref_val) - 1):
                    if (float(each.ref_val[j]) > user_input) and (user_input > float(each.ref_val[j + 1])):
                        val_1 = (
                                (float(each.ref_val[j]) - user_input) / (
                                float(each.ref_val[j]) - float(each.ref_val[j + 1]))
                        )
                        each.transformed_val[j + 1] = val_1
                        val_2 = 1 - val_1
                        each.transformed_val[j] = val_2
        # print("Value after input transformation: {}".format(each.transformed_val))

    def weight_calculation(self, power_factor):
        max_attribute_weight = 0
        for each in range(len(self.obj_list)):
            if self.obj_list[each].attribute_weight > max_attribute_weight:
                max_attribute_weight = self.obj_list[each].attribute_weight

        sum_matching_degree = 0
        matching_degree = list()
        activation_weight = list()
        for rule_index, rule in enumerate(self.rule_row_list):
            rule_matching_degree = 1
            for attribute_index, attribute in enumerate(self.obj_list):
                rule_matching_degree *= pow(pow(rule.attributes_individual_matching[attribute_index], power_factor),
                                            attribute.attribute_weight / max_attribute_weight)
            # rule.matching_degree = rule_matching_degree * rule.rule_weight
            matching_degree.append(rule_matching_degree * rule.rule_weight)
            # rule.matching_degree = rule_matching_degree
            sum_matching_degree += rule_matching_degree

        for matching_index, degree in enumerate(matching_degree):
            activation_weight.append(degree / sum_matching_degree)
        return activation_weight

    #this function is validated with original paper data
    def rule_activation_weight_calculation(self):
        max_attribute_weight = 0
        numberOfTotalActivatedWeight = 0

        # find the maximum attribute weight
        for each in self.obj_list:
            if each.attribute_weight > max_attribute_weight:
                max_attribute_weight = each.attribute_weight

        sum_matching_degree = 0
        for rule_index, rule in enumerate(self.rule_row_list):
            rule_matching_degree = 1
            for attribute_index, attribute in enumerate(self.obj_list):
                rule_matching_degree *= pow(attribute.transformed_val[rule.combinations[attribute_index]],
                                            attribute.attribute_weight / max_attribute_weight)
            rule.matching_degree = rule_matching_degree * rule.rule_weight
            # rule.matching_degree = rule_matching_degree
            if not rule_matching_degree == 0:
                numberOfTotalActivatedWeight += 1
            sum_matching_degree += rule.matching_degree

        for rule_index, rule in enumerate(self.rule_row_list):
            rule.activation_weight = rule.matching_degree / sum_matching_degree
        print('Total Activated Rule = ', numberOfTotalActivatedWeight)
        return

    '''
    Calculate activation weight
    '''

    def activation_weight(self):
        # matching_degree = list()
        total_active_rule = 0

        for i, row in enumerate(self.combinations):
            current_rule = self.rule_row_list[i]
            degree = 1.0
            for idx, val in enumerate(row):
                degree *= float(
                    pow(float(self.obj_list[idx].transformed_val[val]), float(self.obj_list[idx].attribute_weight))

                )
            # matching_degree.insert(i, degree)
            current_rule.matching_degree = degree

        sum = 0.0
        for k in range(len(self.rule_row_list)):
            current_rule = self.rule_row_list[k]
            # current_rule.matching_degree = matching_degree[k]
            sum += float(current_rule.rule_weight) * float(current_rule.matching_degree)

        for p in range(len(self.rule_row_list)):
            current_rule = self.rule_row_list[p]
            activation_weight = float(
                (float(current_rule.rule_weight) * float(current_rule.matching_degree)) /
                sum
            )
            current_rule.activation_weight = activation_weight
            if not activation_weight == 0:
                total_active_rule = total_active_rule + 1
        print('Total Active Rule for ', self.parent.antecedent_id, 'is ', total_active_rule)
        return


    def rules_aggregation_analytical(self):
        b = [0 for _ in range(len(self.rule_row_list))]
        a = [0 for _ in range(len(self.rule_row_list))]
        c = [0 for _ in range(len(self.rule_row_list))]
        final_consequence = [0 for _ in range(len(self.con_ref_values))]
        product_b = 1
        product_a = [1 for _ in range(len(self.con_ref_values))]
        product_c = 1
        sum_product_a = 0

        for i in range(len(self.rule_row_list)):
            c[i] = 1 - (float(self.rule_row_list[i].activation_weight))
            product_c *= c[i]

            for j in range(len(self.rule_row_list[i].consequence_val)):
                b[i] += self.rule_row_list[i].consequence_val[j] * self.rule_row_list[i].activation_weight
            b[i] = 1 - b[i]
            product_b *= b[i]

            for j in range(len(self.rule_row_list[i].consequence_val)):
                product_a[j] *= self.rule_row_list[i].consequence_val[j] * self.rule_row_list[i].activation_weight + b[
                    i]

        for j in range(len(product_a)):
            sum_product_a += product_a[j]
        for j in range(len(product_a)):
            final_consequence[j] = (product_a[j] - product_b) / (
                        sum_product_a - ((len(self.con_ref_values)) - 1) * product_b - product_c)
        self.parent.transformed_val = final_consequence

        return final_consequence


    def dynamic_rule_activation(self):
        power_factor = 4
        best_performing_power_factor = 1
        hyper_tuple_size = len(self.rule_row_list)
        best_performing_factor_consistency = self.hypertuple_consistency_measure()
        factoring_consistency = self.hypertuple_consistency_measure()

        while power_factor > 0 and (
                hyper_tuple_size <= 1 or factoring_consistency < best_performing_factor_consistency):
            temp_rule_list = list()
            weights = self.weight_calculation(power_factor)
            for weight_index, weight in enumerate(weight):
                if weight > 0:
                    temp_rule_list.append(self.rule_row_list[weight_index])
            factoring_consistency = self.hypertuple_consistency_measure()
            if len(temp_rule_list) == 0:
                power_factor = power_factor - 1
            elif factoring_consistency > best_performing_factor_consistency:
                best_performing_power_factor = power_factor
            power_factor = power_factor + 1
            hyper_tuple_size = len(self.rule_row_list)
            factoring_consistency = self.hypertuple_consistency_measure()

        return

    def hypertuple_consistency_measure(self):

        for rule_index, rule in enumerate(self.rule_row_list):
            max_dist = 0
            max_rating = None

            for rating_point, belief_degree in enumerate(rule.consequence_belief_dist):
                if belief_degree >= max_dist:
                    max_dist = belief_degree
                    max_rating = rating_point
            self.consequence_rating_count[max_rating] = self.consequence_rating_count[max_rating] + 1

        overall_max_count = 0
        for index, count in enumerate(self.consequence_rating_count):
            if count > overall_max_count:
                overall_max_count = count

        consistency_measure = overall_max_count / len(self.rule_row_list)
        return consistency_measure

    def create_coperative_brb(self):
        temp = np.random.uniform(0, 1, len(self.obj_list))
        rule_weight = temp / np.sum(temp)

        for idx, value in enumerate(self.obj_list):
            rule = Rules()
            rule.activation_weight = rule_weight[idx]
            rule.consequence_val = value.transformed_val
            self.rule_row_list.append(rule)

    def create_population_rulebase_cooperative_brb_root(self):
        temp = np.random.uniform(0, 1, len(self.obj_list))
        rule_weight = temp / np.sum(temp)

        for idx, value in enumerate(self.obj_list):
            rule = Rules()
            rule.activation_weight = rule_weight[idx]
            rule.consequence_val = value.transformed_val
            self.rule_row_list.append(rule)
        return list(rule_weight)

    def assign_cbrb_weight_coefficient(self, training_parameter, count):
        for idx, value in enumerate(self.obj_list):
            rule = Rules()
            rule.activation_weight = training_parameter[count]
            count = count + 1
            rule.consequence_val = value.transformed_val
            self.rule_row_list.append(rule)
        return count

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
            self.rule_row_list.append(rule)
