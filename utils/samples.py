import json
from langchain_google_vertexai import VertexAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Literal
import random
import copy
import warnings

class Selector():
    """ class to facilitate the selection of examples to improve LLM inference performance
    """
    def __init__(self, file_path) -> None:
        """initialization of the Selector class
        """
        self.examples = []
        self.file_path = file_path

    def load(self) -> object:
        """loads a set of samples into the class

        Args:
            file_path (Optional[str], optional): Path to the jsonl file containing the examples. Defaults to None.

        Raises:
            Exception: "No file path is provided"
            e: FileNotFoundError or JSONDecodeError

        Returns:
            object: The instance of the class for method chaining
        """
        # check if file is of type jsonl
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    self.examples.append(json.loads(line))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise e
        return self
    
    def save(self, file_path: Optional[str]) -> None:
        """saves the examples contained in the class

        Args:
            file_path (Optional[str]): path to jsonl file where the data will be stored

        Raises:
            Exception: No file path is provided
            e: FileNotFoundError or JSONDecodeError
        """
        if not file_path:
            file_path = self.file_path

        if file_path is None:
            raise Exception("No file path is provided")
        
        try:
            with open(file_path, 'w') as f:
                for i, entry in enumerate(self.examples):
                    json.dump(entry, f)
                    if i < len(self.examples) - 1:
                        f.write('\n')
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise e
        
    def get_samples(self, parameters: List[Literal["llm_input", "nodes", "relationships", "reason"]]) -> list[dict]:
        """returns samples that compy with the given parameters

        Args:
            parameters (List[Literal[&quot;llm_input&quot;, &quot;nodes&quot;, &quot;relationships&quot;, &quot;reason&quot;]]): parameters (if existing) that should be included for each example in the output 

        Returns:
            list[dict]: set of examples compliant with the specified output parameters
        """
        samples = []

        for sample in self.examples:
            # Create a copy of the entry to avoid modifying the original data
            sample_copy = copy.deepcopy(sample)
            parsed_sample = {}

            if "llm_input" in parameters and "llm_input" in sample.keys():
                parsed_sample["llm_input"] = sample_copy["llm_input"]
            
            if "reason" not in parameters:
                for node in sample_copy["nodes"]:
                    if "reason" in node.keys():
                        del node["reason"]
                for relationship in sample_copy["relationships"]:
                    if "reason" in relationship.keys():
                        del relationship["reason"]
            
            if "nodes" in parameters and "nodes" in sample.keys():
                parsed_sample["nodes"] = sample_copy["nodes"]
            
            if "relationships" in parameters and "relationships" in sample.keys():
                parsed_sample["relationships"] = sample_copy["relationships"]
            
            # Append the modified entry to the output list
            samples.append(parsed_sample)

        return samples
        
    def __embedd_examples(self, examples: list) -> list[dict]:
        """adds embedding of the llm input

        Args:
            examples (list): list of examples

        Returns:
            list[dict]: all input examples expanded by the llm_input embedded if provided
        """
        for sample in examples:
            if "llm_input" in sample.keys():
                sample["embedding"] = self.__embedd(text=sample["llm_input"])
        return examples
    
    def __embedd(self, text: str) -> list[float]:
        """embedds a provided text

        Args:
            text (str): text that is embedded

        Returns:
            list[float]: embedding vector of the provided text
        """
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004", api_transport="rest")
        return embeddings.embed_query(text=text)

    def similarityBased(self, examples: list, context: str, k: int) -> list[dict]:
        """return the specified count of examples that are most similar to the input

        Args:
            examples (list): list of examples that are checked for similarity
            context (str): input text that is checked for similarity to each provided example
            k (int): number of samples to retrieve

        Raises:
            ValueError: "Dictionary at index {index} is missing the 'llm_input' key."

        Returns:
            list[dict]: k examples that are most similar to the input text
        """
        for index, d in enumerate(examples):
            if "llm_input" not in d:
                raise ValueError(f"Dictionary at index {index} is missing the 'llm_input' key.")
            
        rating_list = []

        # vectorize examples and input text
        examples = self.__embedd_examples()
        vectorized_sample = self.__embedd(text=context)

        # calculate cosine similarity between the input and every example 
        for sample in examples:
            vectors = (vectorized_sample, sample["embedding"])
            similarity_score = cosine_similarity(vectors)[0, 1]
            rating_list.append({"data": sample, "similarity_score": similarity_score})

        # sort the examples by similarity
        rating_list = sorted(rating_list, key=lambda x: x["similarity_score"], reverse=True)
        
        if k <= len(rating_list):
            return [item["data"] for item in rating_list[:k]]
        else:
            warnings.warn(f"There are not enough values to return.")
            return [item["data"] for item in rating_list]

    def lengthBased(self, examples: list, context: str, k: int) -> list[dict]:
        """returns examples most similar in length to the provided context

        Args:
            examples (list): list of examples 
            context (str): context that the list of examples will be checked against for similar length
            k (int): amount of examples that are returned

        Returns:
            list[dict]: k examples that are most similar in length to the input text
        """
        return sorted(examples, key=lambda d: abs(len(d.get("llm_input", "")) - len(context)))[:k]
    
    def countBased(self, examples: list, k: int, random: bool=False) -> list[dict]:
        """retuns the specified number of examples

        Args:
            examples (list): list of examples
            k (int): number of examples to return
            random (bool, optional): Wether or not the retrieved examples will be returned at random. Defaults to False.

        Returns:
            list[dict]: list of k examples
        """
        if random:
            return random.sample(examples, k)
        return examples[:k]
    
    def shuffle(self, examples: list) -> list[dict]:
        """shuffles the examples

        Args:
            examples (list): list of examples

        Returns:
            list[dict]: shuffled list of examples
        """
        return random.shuffle(examples)