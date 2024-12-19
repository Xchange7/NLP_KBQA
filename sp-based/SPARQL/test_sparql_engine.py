"""
For testing the SPARQL engine - virtuoso connection
"""
import rdflib
from rdflib.plugins.stores import sparqlstore

# Disable DeprecationWarning from rdflib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def execute(disable_output=False):
    virtuoso_address = "http://127.0.0.1:8890/sparql"
    virtuoso_graph_uri = 'http://nlp.project.tudelft.nl/kqapro'
    endpoint = virtuoso_address

    query_results = None

    print("---------------------------------")
    print("Checking connection to Virtuoso...")

    try:
        store = sparqlstore.SPARQLUpdateStore(endpoint)
        gs = rdflib.ConjunctiveGraph(store)
        gs.open((endpoint, endpoint))
        gs1 = gs.get_context(rdflib.URIRef(virtuoso_graph_uri))
        res = gs1.query("SELECT * WHERE {?s ?p ?o}")
        query_results = res
    except Exception as e:
        raise Exception("Virtuoso connection failed!")
    
    print("Connection to Virtuoso successful!")
    print("---------------------------------")
    if not disable_output:
        print("First 10 results:")
        for i, row in enumerate(query_results):
            if i > 10:
                break
            print(row)

        print("---------------------------------")
        output_result = (input("Do you want to see the remaining query results? (True/False): ").lower() == 'true')
        if output_result:
            for row in query_results:
                print(row)


if __name__ == "__main__":
    execute()