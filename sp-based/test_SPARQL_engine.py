from SPARQL.sparql_engine import *




if __name__ == '__main__':
    with open('','r') as f:
        # def get_sparql_answer(sparql, data):
        for line in f:
            get_sparql_answer(line, data)