from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from google.api_core.exceptions import BadRequest
import copy
from typing import List
import warnings
from botocore.exceptions import ReadTimeoutError
from pydantic import BaseModel
import time

class OutputParser:
    """ Dynamically constructs the prompt according to the given parameters and inferres the specified LLM. 
    """
    def __init__(self, llm_model: any) -> None:
        """ initializes the OutputParser class

        Args:
            llm_model (any): LLM model that will be used for inference
        """
        self.llm_model = llm_model

    def extract_json(self, prompt: str, output_structure: BaseModel, context: str, examples: List[dict] = [], skip: bool=False) -> List:
        """ Adjust the prompt according to the input parameters and returns the inferred models response adhering to the specified output_structure.

        Args:
            prompt (str): The prompt for the specific task the LLM is inferred with.
            examples (List[dict]): Examples to guide the LLM to the expected output.
            data_structure (BaseModel): The data structure to which the LLM is forced to output the results of the inference.
            context (str): The text on which the described task will be applied.
        
        Returns:
            List: The LLMs response adhering to the given output_structure.
        
        Raises:
            OutputParserException: The output of the LLM was returned in an incorrect json format.
        """
        parser = JsonOutputParser(pydantic_object=output_structure)
        examples = copy.deepcopy(examples)

        # universal prompting template 
        pt = PromptTemplate(
            template="""
            {prompt}

            {examples}

            Output format: 
            {format_instructions}

            This is the context you should apply this to: 
            {context}
            """,
            input_variables=["context", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            optional_variables=["examples"]
            )

        chain = pt | self.llm_model | parser

        # number of retries to infer the LLM in case of errors
        retries = 5
        attempt = 0
        while attempt < retries:
            try:
                # adjust prompt in case examples are passed in the parameters
                if len(examples) > 0:
                    return chain.invoke({"prompt": prompt, "context": context, "examples": "Here are some examples that comply with the required task:" + "\n- ".join([str(item) for item in examples])})
        
                return chain.invoke({"prompt": prompt, "context": context, "examples": ""})
            
            # avoid faulty json outputs from the LLM 
            except OutputParserException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if skip:
                    warnings.warn("Further attempts are skipped as requested due to an encountered error!")
                    break
            
            except BadRequest as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if skip:
                    warnings.warn("Further attempts are skipped as requested due to an encountered error!")
                    break
                
                # reduction of examples if input token size is exceeded
                if len(examples) > 0:
                    examples.pop()
                    print(f"Reducing to {len(examples)} examples...")
            
            except ReadTimeoutError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if skip:
                    warnings.warn("Further attempts are skipped as requested due to an encountered error!")
                    break

            except TimeoutError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if skip:
                    warnings.warn("Further attempts are skipped as requested due to an encountered error!")
                    break

        # retry   
        if attempt < retries:
            time.sleep(1)  
            print("Retrying...")

        # terminated the process
        if attempt == retries:
            raise Exception("Retrying attempts exceeded!")