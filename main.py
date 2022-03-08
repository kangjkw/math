import numpy as np
import scipy.special
import matplotlib.pyplot
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,lr):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        self.lr = lr
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets= np.array(targets_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)

        self.who +=self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))#疑问1
        self.wih +=self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),np.transpose(inputs))#疑问2
        pass
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

inputs_nodes = 784
hidden_nodes = 200
outputs_nodes = 10
learning_rate = 0.1
n = neuralNetwork(inputs_nodes,hidden_nodes,outputs_nodes,learning_rate)
train_file=open('./data/mnist_train.csv','r')
train_list=train_file.readlines()
train_file.close()
epochs = 5
for e in range(epochs):
    for record in train_list:
        all_values =record.split(',')
        inputs = (np.asfarray(all_values[1:]) /255.0 * 0.99) + 0.01
        targets = np.zeros(outputs_nodes) +0.01
        targets[int(all_values[0])] =0.99
        n.train(inputs,targets)
        pass
    pass
test_file=open('./data/mnist_test.csv','r')
test_list = test_file.readlines()
test_file.close()
score_card = []
for record in test_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) /255.0 * 0.99) + 0.01
    outputs =n.query(inputs)
    label = np.argmax(outputs)
    if (label==correct_label):
        score_card.append(1)
    else:
        score_card.append(0)
        pass
    pass
scorecard_array = np.asarray(score_card)
print("performance = ",scorecard_array.sum() / scorecard_array.size)
