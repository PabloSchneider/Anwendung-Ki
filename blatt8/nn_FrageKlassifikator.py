import torch
import torch.nn as nn
import nltk
import re

stem = nltk.stem.PorterStemmer()

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


    print(len(tokens))

    return tokens

def create_tensor(vocab, question, answer):
    '''creating a tensor with the quesions and the vocab'''
    arr = [1 if x in question else 0 for x in vocab]
    print(arr)


vocab = build_bag_of_words("data/dataframes/train.csv")
create_tensor(vocab, "hallo wie geht es dir", "hallo")