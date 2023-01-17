import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn



def eingabe():
    print("give me word: ")
    eingabe = input()

    synsets = wn.synsets(eingabe)

    return synsets


def get_hypernyms(synsets):
    
    s = synsets[0]
    hypernyms = []

    hyper = s.hypernyms()[0]
    hypernyms.append(hyper.lemma_names()[0])
    while hyper.lemma_names()[0] != "entity":
        hyper = hyper.hypernyms()[0]
        hypernyms.append(hyper.lemma_names()[0])
    return hypernyms

def get_synonyms(synsets):

    synonyms = []

    for syn in synsets:
        for i in syn.lemmas():
            synonyms.append(i.name())
    return set(synonyms)


def run():
    synset = eingabe()
    hypernyms = get_hypernyms(synset)
    synonyms = get_synonyms(synset)


    print("Hypernyms: ", hypernyms)
    print("Synonyms: ", synonyms)


if __name__== '__main__':
    run()
