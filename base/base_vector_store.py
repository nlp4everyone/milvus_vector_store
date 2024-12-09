# Typing
import uuid
from typing import  List, Sequence, Literal, Optional, Union
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from langchain_core.documents.base import Document
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.embeddings import Embeddings
# Client
from pymilvus import MilvusClient
# Milvus components
from pymilvus import CollectionSchema, DataType
# Fields
node_with_score_fields = ['id_', 'embedding', 'metadata', 'excluded_embed_metadata_keys', 'excluded_llm_metadata_keys',
                          'relationships', 'text', 'mimetype', 'start_char_idx', 'end_char_idx', 'text_template',
                          'metadata_template', 'metadata_seperator', 'class_name']


# DataType
Embedding = List[float]

class BaseVectorStore(MilvusClient):
    def __init__(self,
                 collection_name: str = "milvus_vector_store",
                 uri: str = "http://localhost:19530",
                 user: str = "",
                 password: str = "",
                 db_name: str = "",
                 token: str = "",
                 search_metrics: Literal["COSINE", "L2", "IP", "HAMMING", "JACCARD"] = "COSINE",
                 index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT",
                 **kwargs):
        # Inheritance
        super().__init__(uri = uri,
                         user = user,
                         password = password,
                         db_name = db_name,
                         token = token,
                         **kwargs)

        # Define params
        self._collection_name = collection_name
        self._search_metrics = search_metrics
        self._index_algo = index_algo
        self._uri = uri

    @staticmethod
    def _embed_texts(texts: list[str],
                     embedding_model: Union[BaseEmbedding,Embeddings],
                     batch_size: int,
                     num_workers: int,
                     show_progress: bool = True) -> List[Embedding]:
        """
        Return embedding from documents

        Args:
            texts (list[str]): List of input text
            embedding_model (BaseEmbedding/Embeddings): The text embedding model
            batch_size (int): The desired batch size
            num_workers (int): The desired num workers
            show_progress (bool): Indicate show progress or not

        Returns:
             Return list of Embedding
        """
        # Base Embedding encode text
        if isinstance(embedding_model, BaseEmbedding):
            # Set batch size and num workers
            embedding_model.num_workers = num_workers
            embedding_model.embed_batch_size = batch_size
            # Return embedding
            return embedding_model.get_text_embedding_batch(texts = texts,
                                                            show_progress = show_progress)
        # Langchain Embeddings
        embedding_model :Embeddings
        return embedding_model.embed_documents(texts = texts)


    @staticmethod
    def _convert_upsert_data(documents: Sequence[Union[BaseNode,Document]]) -> list[dict]:
        """
        Construct the payload data from LlamaIndex document/node datatype

        Args:
            documents (BaseNode): The list of BaseNode datatype in LlamaIndex

        Returns:
            Payloads (list[dict).
        """

        # Check document type
        if isinstance(documents[0],BaseNode):
            # Clear private data from payload
            for i in range(len(documents)):
                documents[i].embedding = None
                # Pop file path
                if documents[i].metadata.get("file_path"):
                    documents[i].metadata.pop("file_path")
                # documents[i].excluded_embed_metadata_keys = []
                # documents[i].excluded_llm_metadata_keys = []
                # Remove metadata in relationship
                for key in documents[i].relationships.keys():
                    documents[i].relationships[key].metadata = {}

            # Get payloads
            return [document.dict() for document in documents]
        else:
            # Langchain Document
            return [{"id_": str(uuid.uuid4()),
                     "page_content": document.page_content,
                     "metadata": document.metadata,
                     "_node_type": document.type} for document in documents]

    @staticmethod
    def _convert_response_to_node_with_score(responses: List[dict],
                                             remove_embedding: bool = True) -> Sequence[NodeWithScore]:
        """
        Convert response from searching to NodeWithScore Datatype (LlamaIndex)

        Args:
            responses (List[dict]): List of dictionary
        Returns:
            Sequence of NodeWithScore
        """
        # Get node with format
        results = []
        for response in responses:
            # Get the main part
            temp = dict(response['entity'])
            # temp.update({"score": response['distance']})
            # Remove embedding
            if remove_embedding: temp['embedding'] = None
            results.append(temp)

        # Define text nodes
        text_nodes = [TextNode.from_dict(result) for result in results]
        # Return NodeWithScore
        return [NodeWithScore(node=text_nodes[i], score=responses[i]["distance"]) for i in range(len(responses))]

    def _setup_collection_schema(self,
                                 vector_type :DataType = DataType.FLOAT_VECTOR,
                                 vector_dims :int = 768) -> CollectionSchema:
        """
        Create collection schema

        Args:
             vector_type (DataType): Type of vector
             vector_dims (int): Numbers for embedding dimension.
        """
        schema = self.create_schema(
            auto_id = False,
            enable_dynamic_field = True,
        )

        # Add field
        schema.add_field(field_name = node_with_score_fields[0],
                         datatype = DataType.VARCHAR,
                         is_primary = True,
                         max_length = 64)
        schema.add_field(field_name = node_with_score_fields[1],
                         datatype = vector_type,
                         dim = vector_dims)
        return schema

    def _setup_collection_index(self,
                                index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT",
                                search_metric :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                                params :Optional[dict] = None):
        """
        Set index (Index params dictate how Milvus organizes your data)

        Args:
            index_algo : The name of the algorithm used to arrange data in the specific field
            search_metric : The algorithm that is used to measure similarity between vectors. Possible values are
            IP, L2, COSINE, JACCARD, HAMMING.
            params : The fine-tuning parameters for the specified index type.
        """
        # Define index
        index_params = self.prepare_index_params()
        if params == None:
            params = {"nlist": 128}

        # Add indexes
        index_params.add_index(field_name = node_with_score_fields[0])
        index_params.add_index(field_name = node_with_score_fields[1],
                               index_type = index_algo,
                               metric_type = search_metric,
                               params = params)
        return index_params

    def _create_collection(self,
                           dimension_nums :int) -> dict:
        """
        Create collection with default name

        Args:
            dimension_nums: Number of dimension for embedding (int)
        """
        # Define schema
        schema = self._setup_collection_schema(vector_dims = dimension_nums)
        # Define index
        index_params = self._setup_collection_index(index_algo = self._index_algo,
                                                    search_metric = self._search_metrics)
        # Collection for LlamaIndex payloads
        self.create_collection(collection_name = self._collection_name,
                               schema = schema,
                               index_params = index_params)

        # Return state
        res = self.get_load_state(
            collection_name = self._collection_name
        )
        return res

    def _create_partition(self,
                          partition_name :str) -> dict:
        """
        Create partition from predefined name.
        """
        assert partition_name, "Collection name must be a string"
        # Check whether partition is existed or not.
        # Create collection
        self.create_partition(collection_name = self._collection_name,
                              partition_name = partition_name)
        # Return state
        return self.get_load_state(collection_name = self._collection_name)

    def retrieve(self,
                 query: str,
                 partition_names: Union[str, List[str]],
                 limit: int = 3,
                 **kwargs):
        raise NotImplementedError

    def insert_documents(self,
                         documents: Sequence[BaseNode],
                         partition_name: str):
        raise NotImplementedError

    def collection_info(self):
        """
        Return collection info
        """
        raise NotImplementedError

    def list_partition(self):
        # Check collection exist
        raise NotImplementedError
