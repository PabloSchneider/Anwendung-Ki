import json, nltk, re
from tqdm import tqdm
import numpy as np
from itertools import islice
import multiprocessing

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


############### changeble parameters ####################
                                                        #
wikirange = -1                                          #
n = 2000                                                #
context = 5                                             #
prozesses = 8 # sollte immer kleiner sein als n         #
useMP = True                                            #
maxWords = 1000                                         #
                                                        #
#########################################################


def loadTextFromWiki():
    """ Loads the wikibase and returns an santized array. """
    stringarray = []
    count = 0
    with open("data/wikibase.jsonl") as file:
        for line in tqdm(file):
            line = json.loads(line)
            
            linearr = nltk.word_tokenize(line["text"])
            linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
            stringarray.extend(linearr)
            count += 1
            
            if wikirange >= 0:
                if count == wikirange:
                    break
    
    return stringarray


def distance(wordList, index, w2):
    """ 
    Checkts if the distance between two words in an list is lesser than +- context.
    returns ture or false
    """
    if wordList[index] == w2:
        return False
    
    posW1 = index

    min = (posW1 - context)
    if min < 0:
        min = 0
    max = posW1 + context
    if max > len(wordList):
        max = len(wordList)-1

    for i in range(min, max):
        if wordList[i] == w2:
            return True
    
    return False


def word_index_count_Dict(wordList:list):
    """ Converts the list to an dict, in which the location of each occurance in the list is saved and the number of occurance"""
    wordDict = dict()
    index = 0
    for word in wordList:
        if word in wordDict:
            wordDict[word][0].append(index)
            wordDict[word][1] = wordDict[word][1]+1
        else:
            wordDict[word] = [[index], 1]  #dict hat den aufbau: {word:[[index_in_array_1, index_in_array_2,...], count]}
        index += 1
    return {k: v for k, v in sorted(wordDict.items(), key=lambda item: item[1][1],reverse=True)} # returns an dict sorted by most occurance


def convert_WordDict_to_WordList(wordDict:dict):
    """
    converts the dict to an list (brauche ich für das iterieren das nur das rechte dreick gefüllt wird.)
    """
    wordList = list()
    for key, val in wordDict.items():
        wordList.append([key, val[0], val[1]])
    
    return wordList


def get_n_most_occurance(wordDict:dict):
    """
    returns a dict with most occurance
    """
    return dict(islice(wordDict.items(), n))


def get_count_w(n_wordList:list):
    """
    claculates the total of words in the list
    """
    count_w = 0
    for ele in n_wordList:
        count_w += ele[2]
    return count_w


def create_PPMI_Matrix(n_wordList:list, wordList:list):
    """
    creates the PPMI matrix 
    """
    matrix = np.zeros(shape=(n,n))
    count_w = get_count_w(n_wordList)
    for i in tqdm(range(0,n)):
        for j in range(i+1,n):

            counter = 0
            for index in n_wordList[i][1]:
                if distance(wordList, index, n_wordList[j][0]):
                    counter += 1
            p1 = counter / count_w
            p2 = n_wordList[i][2] / count_w
            p3 = n_wordList[j][2] / count_w
            if (p2*p3) == 0:
                matrix[i][j] = 0
            else:
                matrix[i][j] = max(np.log2(p1 / (p2 * p3)),0)

    return matrix


def calc_min_max_forIter(i):
    """
    gives boundary in the matrix for each thread
    """

    # hab ich jetzt von hand festgelegt, weil ich zu dumm bin dafür ne vernünftige formel aufzustellen
    if n == 2000 and prozesses == 4:

        if i == 1:
            return 0, 25
        elif i == 2:
            return 25, 75
        elif i == 3:
            return 75, 500
        else:
            return 500, 2000
    elif n == 2000 and prozesses == 8:
        if i == 1:
            return 0, 2
        elif i == 2:
            return 2, 5
        elif i == 3:
            return 5, 10
        elif i== 4:
            return 10, 30
        if i == 5:
            return 30, 80
        elif i == 6:
            return 80, 180
        elif i == 7:
            return 180, 500
        else:
            return 500, 2000

    else:
        # formel die threads sehr ungleich auslastet
        max_n = int(n/prozesses) * i #grenzt den breich ein, den ein Thead abarbeiten soll
        if prozesses == i:
            max_n = n
    
        min_n = int(n/prozesses) * (i-1) #grenzt den breich ein, den ein Thead abarbeiten soll
        return min_n, max_n


def create_PPMI_Matrix_MM(manager, n_wordList:list, wordList:list, i):
    """
    creates PPMI Mathrix with multiple Prozesses
    """
    matrix = np.zeros(shape=(n,n))

    count_w = get_count_w(n_wordList)
    
    min_n, max_n = calc_min_max_forIter(i)
    

    for i in tqdm(range(min_n,max_n)):
        for j in range(i+1,n):

            counter = 0
            for index in n_wordList[i][1]:
                if distance(wordList, index, n_wordList[j][0]):
                    counter += 1
            p1 = counter / count_w
            p2 = n_wordList[i][2] / count_w
            p3 = n_wordList[j][2] / count_w
            if (p2*p3) == 0:
                matrix[i][j] = 0
            else:
                matrix[i][j] = max(np.log2(p1 / (p2 * p3)),0)
    
    manager[i] = matrix


def useMultyProcessing(n_wordList:list, wordList:list):
    """
    creatres multiple Prozesses for the Matrix
    """
    
    manager = multiprocessing.Manager().dict() # ist dazu da um die berechneten Matrix aus den Threds zu lesen

    
    jobs = []

    for i in range(1,prozesses+1):
        t = multiprocessing.Process(target=create_PPMI_Matrix_MM, args=(manager, n_wordList, wordList, i))
        jobs.append(t)
        t.start()

    for job in jobs:
        job.join() 

    #jeder thread gibt einen die Mathrix für den jeweiligen part zurück.
    # wird hier zusammenfeführt
    matrix = np.zeros(shape=(n,n))
    for value in manager.values():
        matrix += value

    return matrix


def words_With_Max_PPMI(matrix, n_wordList:list):

    """
    returns a list of word with the geatest ppmi
    """

    wordsWithMaxPPMI = []
    indexFromMostPPMI = np.dstack(np.unravel_index(np.argsort(matrix.ravel()), (n, n)))[0][::-1]
    print(indexFromMostPPMI)
    for a in range(maxWords):
        index0 = indexFromMostPPMI[a][0] # index von X-Achsen Wort in der Matrix
        index1 = indexFromMostPPMI[a][1] # index von Y-Achsen Wort in der Matrix
        wordsWithMaxPPMI.append([(n_wordList[index0][0], n_wordList[index1][0]), matrix[index0][index1]]) # mit vocab[index][0] holt man sich das Wort
    return wordsWithMaxPPMI


def saveResultInFile(wordsWithMaxPPMI:list):

    """
    saves in file
    """

    filename = "PPMI-W_{a}-n_{b}_MP={c}".format(a=wikirange,b=n, c=useMP)
    f = open("./result/{}.txt".format(filename), "w")

    f.write("PPMI: n = {a} Wikibase = {b} --".format(a=wikirange,b=n))
    
    if useMP:
        f.write("processes: {a} \n".format(a = prozesses))
    count = 1
    for ele in wordsWithMaxPPMI:
        f.write("{a} : {b} == {c} \n".format(a = count, b = ele[0], c = ele[1]))
        count = count +1
    
    
def run():
    print("load_Data")
    wordList = loadTextFromWiki()
    
    print("count words")
    wordDict = word_index_count_Dict(wordList)
    
    print("convert_Dict to list")
    n_word_list = convert_WordDict_to_WordList(get_n_most_occurance(wordDict))

    print("calculate Matrix")
    if useMP:
        matrix = useMultyProcessing(n_word_list, wordList)
    else:
        matrix = create_PPMI_Matrix(n_word_list, wordList)

    print(matrix)
    wordsWithMaxPPMI = words_With_Max_PPMI(matrix, n_word_list)
    saveResultInFile(wordsWithMaxPPMI)
    


if __name__ == "__main__":

    run()
