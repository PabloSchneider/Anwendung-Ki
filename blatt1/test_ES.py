from elasticsearch import Elasticsearch


es = Elasticsearch(['http://localhost:9200'])

print("test")


settings = {
    "number_of_shards" : 1, #sharding 
    "number_of_replicas" : 2
}

mappings = {
    "properties" : {
        "titel" : {"type":"text"}, 
        "text": {
            "type": "text",
            "analyzer": "english", 
            "search_analyzer": "standard" 
        }
    } 
}

es.options(ignore_status=[400,404]).indices.delete(index='my_index')
es.indices.create(index="my_index", settings=settings, mappings=mappings)


doc1 = {
    'titel': 'Kimchi',
    'text': 'Kimchi schmeckt toll.'
}
doc2 = {
    'titel': 'Patients',
    'text': 'Patients sind super.'
}
doc3 = {
    'titel': 'Sushi',
    'text': 'Sushi hat Reis.'
}
doc4 = {
    'titel': 'Apfel',
    'text': 'Der Apfel ist suess.'
}
doc5 = {
    'titel': 'Imposter',
    'text': 'Der Imposter ist sus.'
}

es.index(index="my_index", document=doc1) 
es.index(index="my_index", document=doc2) 
es.index(index="my_index", document=doc3)
es.index(index="my_index", document=doc4) 
es.index(index="my_index", document=doc5)

#helpers.bulk([doc1,doc2,doc3,doc4,doc5])

es.indices.flush() # persist changes (memory -> disk )
es.indices.refresh() # makes changes available for search


query = {
    "match" : {
        "text" : {
            "query" : "patient", # query terms 
            "operator" : "or", # match >= 1 terms
            "fuzziness" : 0, # tolerance : 1 char
        }  
    }
}

result = es.search(index="my_index", size=10, query=query, explain=True)

# Standard Scoring mode is the Okapi-BM25 V2 F28
# 'description': 'tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) -- Okapi-BM25 V2 F28
# idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) -- Gewichtung
print(result)

for hit in result["hits"]["hits"]:
    score, doc = hit["_score"], hit["_source"] 
    print(score, doc["titel"], doc["text"])



