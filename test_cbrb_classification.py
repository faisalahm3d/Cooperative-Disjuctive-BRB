import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#for showing precision, recall and f-measure
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from cbrb_classification_model import ClassificationModel

train_dataframe = pd.read_csv('data\processed_cancer_data.csv')
t =train_dataframe[['Age','BMI','sympthom_score','WBC','RBC','Plat','HGB','Baselinehistological staging']]
#print(t.iloc[:5,:])
#print(t.max())
#print(t.min())
#print(len(t))

'''
#K-fold cross validation
kf = KFold(n_splits=4)
for train, test in kf.split(t.iloc[:5,]):
    print("%s %s" % (train, test))
'''
# spliting dataset 80% for training and 20% for test
train, test = train_test_split(t, test_size=0.2, random_state=True)

attributes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'K')
model = ClassificationModel('data\liver-cancer-data.json',attributes)
fitness_vector = model.fit(train.values)
#print(param)
print('Best fitness ', fitness_vector)
#model.objective_function_ccde(model.parent, count=0)

print('Training Error : ')
# Save the error vector to the csv file
# plot the error vs generation graph
plt.plot(fitness_vector)
plt.xlabel('Generation')
plt.ylabel('Error')
plt.show()
plt.savefig
print('Actual Class')
actual_class = list(test['Baselinehistological staging'])
print(actual_class)
actual_class_dataframe = pd.Series(actual_class)
actual_class_dataframe.to_csv('actual_class.csv')

predicted_class, test_error = model.test(test.values)
print('Predicted Class')
print(predicted_class)
predicted_class_dataframe = pd.Series(predicted_class)
predicted_class_dataframe.to_csv('predicted_class')

# Python script for confusion matrix creation.

results = confusion_matrix(actual_class, predicted_class)

print ('Confusion Matrix :')
print(results)
print ('Accuracy Score :',accuracy_score(actual_class, predicted_class) )
print ('Report : ')
report = classification_report(actual_class, predicted_class, output_dict=True)

#print(report['0']['precision'])


#line chart showing accuracy, f-measure of each stage
stages =['N1','N2','N3','N4']
accuracy_brb = [90,80,77,89]
accuracy_nn = [91,75,76,85]
#plt.figure(num = 3, figsize=(8, 5))
plt.xlabel('Precision(%)')
plt.ylabel('Stage')
plt.plot(stages, accuracy_brb, label = 'BRB')
plt.plot(stages, accuracy_nn,
         color='red',
         label='ANN',
         linewidth=1.0,
         linestyle='-')
plt.legend()
plt.show()