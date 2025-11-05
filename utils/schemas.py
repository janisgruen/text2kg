from typing import List
from pydantic import BaseModel, Field

# ______________________________ Entity & Relationship Extraction ____________________________ #    
class Entity(BaseModel):
    name : str = Field("The specific name of the entity. It is the exact word or phrase in the context that identifies a particular object, concept or individual. Ensure that the entitiy name captures the entity exactly as it appears in the text.")
    type: str = Field("The specific category that best describes the nature or classification of the entity identified in the context. Ensure that the entity type reflects the role or nature of the entity within the context provided. Rely on surrounding context to determine the most accurate entity type.")
    
class EntitiesExtractor(BaseModel):
    nodes : List[Entity] = Field("A list of all entities that are present in the given context.")
    
class Relation(BaseModel):
    source: str = Field("The primary entity or originator in a relationship between two entities. This source represents the starting point or active initiator of the relationship and must be present in the given entity list.")
    relation: str = Field("The specific type of connection, interaction, or association that links the source and the target node. This relation should precisely define how these two entities are related within the given context.")
    target: str = Field("The entity that receives or is affected by the relationship established by the source. This target node is the endpoint or recipient in the relationship and must be present in the given entity list.")

class RelationsExtractor(BaseModel):
    relations: List[Relation] = Field("Based on the provided list of entities and context, identify all specific connections or associations between two entities within given the context. The predicate defining the relationship should clearly represent how the entities interact or relate to each other.")


# ____________________________ CoT Entity & Relationship Extraction __________________________ #  
class CoT_Entity(Entity):
    reason: str = Field("Underlying rationale for identifying the specific entity within the context and assigning it to the specific type.")

class CoTEntitiesExtractor(BaseModel):
    nodes : List[CoT_Entity] = Field("A list of all entities that are present in the given context with a rationale for identifying each specific entity and assigning it to the specific type.")

class CoT_Relation(Relation):
    reason : str = Field("Underlying rational behind identifying and delineating the relationship between entities within the given text.")

class CoTRelationsExtractor(BaseModel):
    relations: List[CoT_Relation] = Field("Based on the provided list of entities and context, identify all specific connections or associations between two entities within given the context. The predicate defining the relationship should clearly represent how the entities interact or relate to each other.")