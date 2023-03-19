import math
class Data(object):

    def __init__(self, antecedent_id, antecedent_name,
                 attribute_weight, ref_val, ref_title,
                 consequent_values, crisp_val, parent,rulebase_filename,
                 is_input, is_root=False,input_val="0"):
        self.name = ""
        self.antecedent_id = antecedent_id
        self.antecedent_name = antecedent_name
        self.attribute_weight = attribute_weight
        self.ref_title = ref_title
        self.ref_val = ref_val
        self.consequent_values = consequent_values
        self.crisp_val = crisp_val
        self.parent = parent
        self.rulebase_filename=rulebase_filename
        self.is_root = is_root
        self.input_val = input_val
        self.activated_ref_point = None
        self.transformed_val = [0 for _ in range(len(self.ref_val))]
        self.is_input = is_input
        self.mean_absolute_error = 0

    def input_transformation_optimize(self):
        for j, value in enumerate(self.ref_val):
            self.transformed_val[j]=0
        for j in range(len(self.ref_val)-1):
            if self.ref_val[j+1]<=self.input_val<=self.ref_val[j]:
                self.transformed_val[j+1] =(self.ref_val[j]-self.input_val)/(self.ref_val[j]-self.ref_val[j+1])
                self.transformed_val[j]=1-self.transformed_val[j+1]
                self.activated_ref_point = (j,j+1)
                break


