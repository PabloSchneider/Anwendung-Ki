import spacy


def eingabe():
    print("write a sentence")
    eingabe = input()
    
    return eingabe

def nlp_converter(text:str):
    nlp = spacy.load("en_core_web_sm")

    return nlp(text)

def named_entities(doc):
    entity_dicc = dict()

    for entity in doc.ents:
        if entity.label_ not in entity_dicc.keys():
            entity_dicc[entity.label_] = [entity.text]
        else:
            entity_dicc[entity.label_].append(entity.text)

    return entity_dicc

def read_questions(path):    
    strings = []
    count = 0
    with open(path) as file:
        for line in file:
            
            
            line = line.split(";")

            if count == 21:
                break
            if count > 0:
                strings.append(line[1])         
            count += 1

    return strings


if __name__ == '__main__':
    for str in read_questions("data/dataframes/train.csv"):
        print("-- ",str," --\n",named_entities(nlp_converter(str)))