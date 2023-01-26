import random
import torch
import torch.nn as nn
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm
import nltk
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

stem = nltk.stem.PorterStemmer()
SAVE_MODEL_PATH = 'blatt8/model/test1.pth'


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x
class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def build_bag_of_words(path):
    
    tokens = set()

    with open(path) as file:
        for line in file:
            line = line.split(";")

            linearr = nltk.word_tokenize(line[1])
            linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
            
            #linearrtag = nltk.pos_tag(linearr)
            #linearr = [a for (a, b) in linearrtag if b[0] == 'N' or b[0] == 'V' or b[0] == 'W']
            for ele in linearr:
                tokens.add(stem.stem(ele)) 



    return list(tokens)

def stemm_question(question):
    '''returns a stemmed question'''
    linearr = nltk.word_tokenize(question)
    linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
    for i in range(len(linearr)):
        linearr[i] = stem.stem(linearr[i])
    return linearr


def create_matrix(vocab, questions):
    '''creating a matrix with the quesions and the vocab'''
    tokens = vocab
    matrix = np.zeros(shape=(len(questions),len(tokens)))

    for i in tqdm(range(len(questions))):
        for j in range(len(tokens)):
            if tokens[j] in questions[i]:
                matrix[i][j] = 1

    
    return matrix


def read_train_or_test_data(vocab,path):
        '''reads training or test data and returns anwsertyps and matrix'''
        df = pd.read_csv(path, sep = ";")
        df["question"] = df["question"].apply(stemm_question)
        questions = np.array(df["question"])
        answertype = np.array(df["answer_type"])
        matrix = create_matrix(vocab,questions)
        answertypes, answertype = convert_anwsertype_to_number(answertype)
        return matrix, answertype

def convert_anwsertype_to_number(answertype):
    answertype = list(answertype)
    answertypes = list(set(answertype))
    matrix = np.zeros(shape=(len(answertype), len(answertypes)))
    count = 0;
    for ele in answertype:
        matrix[count][answertypes.index(ele)] = 1
        count += 1
    return answertypes, matrix


def minibatch(X, t, batch_size):
    n = len(X)
    all_index = list(range(n))
    some = np.array( random.sample(all_index, batch_size))
    X_Batch = list()
    t_Batch = list()
    for i in some:
        X_Batch.append(X[i])
        t_Batch.append(t[i])


    X_ = torch.tensor(np.array(X_Batch).astype(np.float32))
    t_ = torch.tensor(np.array(t_Batch).astype(np.float32))
    return X_,t_

def to_shuffled_tensor(X, t):
    n = len(X)
    all_index = list(range(n))
    some = np.array( random.sample(all_index, len(X)))
    X_Batch = list()
    t_Batch = list()
    for i in some:
        X_Batch.append(X[i])
        t_Batch.append(t[i])


    X_ = torch.tensor(np.array(X_Batch).astype(np.float32))
    t_ = torch.tensor(np.array(t_Batch).astype(np.float32))
    return X_,t_

def run_training(mlp, X, t, epochs = 20, batch_size = 10, lr = 0.01):
    optimizer = optim.Adam(net.parameters(), lr) 
    criterion = nn.CrossEntropyLoss()
    count = 0
    for i in range(epochs):
        x_, t_ = to_shuffled_tensor(X, t)
        # set gradients to zero
        optimizer.zero_grad()

        #forward pass
        y_ = mlp(x_)
        loss = criterion(y_, t_)
        loss.backward()
        optimizer.step()

        #if i % 2 == 0:
        print(f'Iteration {i} : Loss {loss.item()}')
        
        count += 1
    
    #torch.save(mlp.state_dict(), SAVE_MODEL_PATH)

def testing(mlp:Net, X:np.ndarray, t:np.ndarray):
    mlp.eval()

    total = len(X)
    correct = 0

    with torch.no_grad(): # no need to track gradients
        output = mlp(torch.tensor(np.array(X).astype(np.float32)))
        #print(output)

        for i in range(len(output)):
            correct += list(t[i]).index(max(t[i])) == list(output[i]).index(max(output[i]))
    
    print(f'Accuracy: {correct/total}')

if '__main__' == __name__:
    train_path = "projekt/data/dataframes/train.csv"
    test_path = "projekt/data/dataframes/test.csv"

    vocab = build_bag_of_words(train_path)
    train_matrix, train_answers = read_train_or_test_data(vocab, train_path)
    test_matrix, test_answers = read_train_or_test_data(vocab, test_path)

    print(len(train_matrix))
    net = Net(len(vocab), 500, len(train_answers[0]))
    
    run_training(net, train_matrix, train_answers, epochs=30, batch_size=30, lr=0.01)
    print("train fertig")
    testing(net, test_matrix, test_answers)
  
