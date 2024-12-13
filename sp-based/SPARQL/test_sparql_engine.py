"""
For testing the SPARQL engine
"""
from sparql_engine import *


def execute():
    endpoint = virtuoso_address
    store = sparqlstore.SPARQLUpdateStore(endpoint)
    gs = rdflib.ConjunctiveGraph(store)
    gs.open((endpoint, endpoint))
    gs1 = gs.get_context(rdflib.URIRef(virtuoso_graph_uri))
    res = gs1.query("SELECT * WHERE {?s ?p ?o}")
    for row in res:
        print(row)