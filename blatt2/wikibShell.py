from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])


eingabe = input("Ihre Eingabe: ")

query = {
    "match" : {
        "text" : {
            "query" : eingabe, # query terms 
            "operator" : "or", # match >= 1 terms
            "fuzziness" : 0, # tolerance : 1 char
        }  
    }
}

result = es.search(index="wikibase", size=20, query=query) # per size=.... Menge an Hits ausgeben

#print(result)

for hit in result["hits"]["hits"]:
    score, doc = hit["_score"], hit["_source"] 
    print(score, doc["title"])

'''
für "Who murdered Abraham Lincoln?"
-- score -- title --
18.84547 The_Papers_of_Abraham_Lincoln
18.78922 Roy_Basler
18.49391 Abraham_Lincoln_Bicentennial_Commission
18.455437 Lincoln's_Lost_Speech
18.138702 William_Herndon_(lawyer)
18.02123 William_H._Mumler
17.855103 Louis_J._Weichmann
17.85219 Charles_Sabin_Taft
17.808214 Robert_Todd_Lincoln
17.778141 David_Herbert_Donald
'''