import re, json 
from tqdm import tqdm

def loadTextFromWiki():
    string = ""
    count = 0
    with open("data/wikibase.jsonl") as file:
        for line in tqdm(file):
            line = json.loads(line)
            
            string += line["text"]
            count += 1

            if count == 10000:
                break
    return string


def regexNamen(string):
    names = []
    pattern = re.compile(r'[A-Z][a-z]+ (?:[A-Z]\.? )?[A-Z][a-z]+')
    
    names = pattern.findall(string)

    return names

def regexDates(string):
    dates = []
    pattern = re.compile(r'([0-9]{2}\.[0-9]{2}\.[0-9]{4})')
    pattern2 = re.compile(r':?([A-Z][a-z]{2} [0-9]{1,2}(st|nd|rd|th) [0-9]{4})')
    #pattern3 = re.compile()
    dates = pattern2.findall(string)
    return dates


string = loadTextFromWiki()
#names = regexNamen(string)

print(regexDates(string))

#print(names)


