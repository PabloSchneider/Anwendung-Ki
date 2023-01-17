from elasticsearch import Elasticsearch
from tqdm import tqdm

es = Elasticsearch(['http://localhost:9200'])


TEXTBOOSTING = 1
TITLEBOOST = 1.5

liste = []
with open("data/train_data/train_2.csv") as file:
    for line in file:
        line = line.split(";")
        if len(line) > 1:
            line[4] = int(line[4]) # ID zu int
            line[5] = line[5].replace("\n", "") # getting rid of \n
            liste.append(line)

#print(liste)



def resultFunc(ele, k, boost=False):
    if (boost == False):
        query = {
            "match" : {
                "text" : {
                    "query" : ele[1], # query terms 
                    "operator" : "or", # match >= 1 terms
                    "fuzziness" : 0, # tolerance : 1 char
                }  
            }
        }
    else:
        query = { # title mit matchen??
            "multi_match" : {
                "query": ele[1],
                "operator" : "or", # match >= 1 terms
                "fuzziness" : 0, # tolerance : 1 char
                "fields": [
                    "title^{}".format(TITLEBOOST),
                    "text^{}".format(TEXTBOOSTING)
                ]
            }
        }

    result = es.search(index="wikibase", size=k, query=query) # size=1 nur den Top-Treffer
    return result


def q(result, rightID, k):
    match = 0
    len = 0
    for hit in result["hits"]["hits"]:
        if len == k:
            break
        if (int(hit["_id"])) == rightID:
            match = 1
        len += 1
    return match

def qoverallForAll(boosting):
    kdict = {1:0, 5:0, 10:0, 20:0, 50:0, 100:0}

    for i in tqdm(range(len(liste))):
        ele = liste[i]
        result = resultFunc(ele, 100, boosting) # True mitgeben für multimatch
        for key in kdict:
            kdict[key] += q(result, ele[4], key)

    for key in kdict:
        kdict[key] = kdict[key] / len(liste)

    return kdict

def average(kdict):
    avg = 0
    for val in kdict.values():
        avg += val

    avg = avg / len(kdict)
    return avg


def normalerDurchlauf():
    for ele in liste: # ele[1] = Frage, ele[4] = ID
        query = {
            "match" : {
                "text" : {
                    "query" : ele[1], # query terms 
                    "operator" : "or", # match >= 1 terms
                    "fuzziness" : 0, # tolerance : 1 char
                }  
            }
        }

        result = es.search(index="wikibase", size=100, query=query) # size=1 nur den Top-Treffer
        
        for hit in result["hits"]["hits"]:
            score, doc = hit["_score"], hit["_source"] 
            print("Score: ", score)
            print("ID: ", hit["_id"], ", richtige ID wäre: ", ele[4])
            print("Frage war: ", ele[1])
            print("--------------------------------------------------------------")
        
        #print("K = 1: ", q(result, ele[4], 1)) # K für Einzelergebnisse


def saveResultInFile(filename, kdict, avg, boosting):
    
    f = open("./result/{}.txt".format(filename), "w")

    f.write("Results: \n")
    if boosting == True:
        f.write(f"TEXTBOOSTING : {TEXTBOOSTING}\n")
        f.write(f"TITELBOOSTING : {TITLEBOOST}\n")

    else:
        f.write(f"FIELDBOOSTING : Disabled\n")

    
    for key in kdict.keys():
        f.write(f"PREC@K : {key} -- {kdict[key]}\n")

    f.write(f"Average Precision : {avg}\n")



    


saveResultInFile
boosting = True
kdict = qoverallForAll(boosting)
print(average(kdict))
saveResultInFile("boost_Title_1-5_Text_1-0", kdict, average(kdict), boosting)