from ..utils.output_parser import OutputParser 
from ..utils.schemas import EntitiesExtractor
from ..utils.prompt_templates import entity_prompt

from typing import List

from pydantic import BaseModel

class EntitiesExtractor:
    """ Class to extract entities of a given input text.
    """
    def __init__(self, llm_model: any) -> None:
        """ Initializes the EntitiesExtractor class with a specified LLM as an input

        Args:
            llm_model (any): LLM model to be used for entity extraction
        """
        self.output_parser = OutputParser(llm_model=llm_model)

    def extract_entities(self, context: str, prompt: str=entity_prompt, examples: List[dict]=[], output_structure: BaseModel=EntitiesExtractor, skip: bool=False) -> List:
        """ Extracts entities from a given context. 
        If wanted examples can be added. 
        By changing the output_structure, the models reasoning to the entity type assignment can be included in the output.

        Args:
            context (str): The text from which entities should be extracted.
            prompt (str, optional): The prompt used to infere the model. Defaults to entity_prompt.
            examples (List[dict], optional): Examples to guide the LLM to the expected output. Defaults to [].
            output_structure (BaseModel, optional): The data structure to which the LLM is forced to output the results of the inference. Defaults to EntitiesExtractor.
            skip (bool): Wether or not the in case of errors, the extraction of entities is skipped. Defaults to False.

        Returns:
            List: A list of entities extracted from the provided context.
        """
        # infere LLM
        entities = self.output_parser.extract_json(
            prompt=prompt, 
            examples=examples, 
            output_structure=output_structure, 
            context=context,
            skip=skip
        )

        if not isinstance(entities, dict):
            return {}
        
        return entities