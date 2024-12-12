"""
For testing the SPARQL engine
"""
import rdflib
from rdflib import URIRef, BNode, Literal, XSD, Dataset
from rdflib.plugins.stores import sparqlstore


virtuoso_address = "http://127.0.0.1:8890/sparql"
virtuoso_graph_uri = "http://nlp.project.tudelft.nl/kqapro"


endpoint = virtuoso_address
store = sparqlstore.SPARQLUpdateStore(endpoint)
gs = rdflib.ConjunctiveGraph(store)
gs.open((endpoint, endpoint))
gs1 = gs.get_context(rdflib.URIRef(virtuoso_graph_uri))
res = gs1.query("SELECT * WHERE {?s ?p ?o}")
for row in res:
    print(row)