# Masterthesis
Repository to the masterthesis - Enhancing Knowledge Graph creation: Leveraging LLMs inference for automatic node and relationship discovery in system engineering documents

# Prequisits 
1. Install requirements.txt using "pip install -r requirements.txt."
2. Install LangChain packages needed for usage with desired LLM or cloud provider.

# Setup
1. Create an object of the class EntityExtractor initialized with the desired LLM using a LangChain.
2. Create an object of the class RelationsExtractor initialized with the desired LLM using a LangChain.
3. Create an object of the class GraphBuilder.

# Pipeline
Illustrated is the minimal requirements for usage
1. Use entities = entity_ext_obj.extract_entities(context=text_chunk)
2. Add entities to the graph graph.addNodes(nodes=entities)
3. Use relations = relation_ext_obj.extract_relations(context=text_chunk, entities=entities)
4. Add relations to the graph graph.addEdges(edges=relations)
5. DONE!

An example nodebook can be found in ./tests/test_run.ipynb