from elasticsearch import Elasticsearch

# run elasicsearch with:
# ES_PATH_CONF=my_config ./bin/elasticsearch
# or:
# ./bin/elasticsearch

class wikibase_manager():
    ''' class for operations on elasicsearch '''
    def __init__(self, question, size):
        self.question = question
        self.es = Elasticsearch(['http://localhost:9200'])
        self.size = size

    def set_question(self, new_question):
        ''' write a new questins '''
        self.question = new_question
    
    def set_size(self, new_size):
        ''' define a new size for result size '''
        self.size = new_size

    def create_query(self):
        ''' creates a query '''
        query = {
            "match" : {
                "text" : {
                    "query" : self.question, # query terms 
                    "operator" : "or", # match >= 1 terms
                    "fuzziness" : 0, # tolerance : 1 char
                }  
            }
        }
        return query

    def search_in_wikibase(self):
        ''' serches in wikibase '''

        result = self.es.search(index="wikibase", size=self.size, query=self.create_query()) # per size=.... Menge an Hits ausgeben

        return result
    
    def results_to_text(self, path):
        ''' writes result in textfile  '''
        result = self.search_in_wikibase()
        f = open(path, 'w')
        for hit in result["hits"]["hits"]:
            score, doc = hit["_score"], hit["_source"] 
            f.write(doc['text'])
            print(score, doc)


if __name__ == '__main__':

    wiki = wikibase_manager("Who murdered Abraham Lincoln", 20)

    wiki.results_to_text('projekt/data/wikidata.txt')