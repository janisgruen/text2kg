from ..utils.output_parser import OutputParser 
from ..utils.schemas import RelationsExtractor
from ..utils.prompt_templates import relationship_prompt

from typing import List

from pydantic import BaseModel

class RelationsExtractor:
    """ Class to extract relationships between the provided entities and the context.
    """
    def __init__(self, llm_model: any) -> None:
        """ Initializes the RelationsExtractor class the specified LLM.

        Args:
            llm_model (any): LLM model to be used for relationship extraction.
        """
        self.output_parser = OutputParser(llm_model=llm_model)

    def extract_relations(self, context: str, entities: list[str], 
                          examples: List[str] = [], prompt: str=relationship_prompt, 
                          output_structure: BaseModel=RelationsExtractor, 
                          skip: bool=False) -> List:
        """ Extracts relationships between the provided list of entities from a given context. 
            If wanted examples can be added.
            By changing the output_structure, the models reasoning to the relationship type assignment can be included in the output.

        Args:
            context (str): The text from which relationships should be extracted
            entities (list[str]): The entities between which relationships should be identified
            examples (List[str], optional): Examples to guide the LLM to the expected output. Defaults to [].
            prompt (str, optional): The prompt used to infere the model. Defaults to relationship_prompt.
            output_structure (BaseModel, optional): The data structure to which the LLM is forced to output the results of the inference. Defaults to RelationsExtractor.
            skip (bool): Wether or not the in case of errors, the extraction of entities is skipped. Defaults to False.

        Returns:
            List: A list of the extracted entities adhering to the defined output structure.
        """
        # skipped to reduce computations
        if len(entities) > 0:
            formated_context = f"{context}\n\nentities:{entities}"
            relations = self.output_parser.extract_json(prompt=prompt, examples=examples, context=formated_context, output_structure=output_structure, skip=skip)
            if not isinstance(relations, dict):
                return {}
        return relations