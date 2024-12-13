# Typing
from typing import  List, Sequence, Literal, Optional, Union
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from langchain_core.documents.base import Document
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.embeddings import Embeddings
from ..types import (FundamentalField,
                     BaseNodeField,
                     DocumentField)
# Milvus components
from pymilvus import (CollectionSchema,
                      DataType,
                      MilvusClient)
# Sparse embedding
from fastembed import SparseTextEmbedding
from milvus_model.sparse.splade import SpladeEmbeddingFunction
# Import
import uuid, scipy

# List all default keys
default_keys = list(FundamentalField.model_fields.keys())
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
                 dense_search_metrics: Literal["COSINE", "L2", "IP", "HAMMING", "JACCARD"] = "COSINE",
                 index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT",
                 dense_datatype :Literal["FLOAT_VECTOR","FLOAT16_VECTOR","BFLOAT16_VECTOR"] = "FLOAT_VECTOR",
                 **kwargs):
        super().__init__(uri = uri,
                         user = user,
                         password = password,
                         db_name = db_name,
                         token = token,
                         **kwargs)

        # Define params
        self._collection_name = collection_name
        self._dense_search_metrics = dense_search_metrics
        self._index_algo = index_algo
        self._dense_datatype = dense_datatype
        self._uri = uri

    @staticmethod
    def _embed_texts(texts: list[str],
                     embedding_model: Union[BaseEmbedding,Embeddings],
                     batch_size: int,
                     num_workers: int,
                     show_progress: bool = True) -> List[Embedding]:
        """
        Get dense representation vector of incoming document contents.
        :param texts: List of input text
        :param embedding_model:The model for dense embedding
        :param batch_size: The desired batch size
        :param num_workers: The desired num workers
        :param show_progress: Indicate show progress or not
        :return: List[Embedding]
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
    def _embed_query(query: str,
                     embedding_model: Union[BaseEmbedding, Embeddings]) -> Embedding:
        """
        Get dense representation vector of input query.
        :param query: The query text input
        :param embedding_model: The dense embedding model
        :return:
        """

        # Get query representation from Llama Index BaseEmbedding model
        if isinstance(embedding_model, BaseEmbedding):
            return embedding_model.get_query_embedding(query = query)
        # Get query representation from Langchain Embeddings model
        return embedding_model.embed_query(text = query)

    @staticmethod
    def _sparse_embed_texts(texts: list[str],
                            sparse_embedding_model: Union[SparseTextEmbedding, SpladeEmbeddingFunction],
                            batch_size :int = 32,
                            parallel :int = 1) -> List[dict]:
        """
        Get sparse representation of incoming document contents.
        :param texts: List of input texts
        :param sparse_embedding_model: The dense embedding model for embed
        :param batch_size: The desired batch size
        :param parallel: The number of parallel processing
        :return: List[dict]
        """
        # Splade Sparse Embedding encode
        if isinstance(sparse_embedding_model, SparseTextEmbedding):
            sparse_embeddings = sparse_embedding_model.embed(documents = texts,
                                                             batch_size = batch_size,
                                                             parallel = parallel)
            # Return as object
            return [embedding.as_dict() for embedding in list(sparse_embeddings)]
        elif isinstance(sparse_embedding_model,SpladeEmbeddingFunction):
            # Splade Embedding
            sparse_embeddings = sparse_embedding_model.encode_documents(documents = texts)
            return [BaseVectorStore._convert_csr_array_to_dict(embedding) for embedding in sparse_embeddings]
        # Doesnt support
        raise NotImplementedError("This version only support FastEmbed SparseTextEmbedding!")

    @staticmethod
    def _sparse_embed_query(query: str,
                            sparse_embedding_model: SparseTextEmbedding) -> List[dict]:
        """
        Get sparse representation of incoming query.
        :param query: The incoming query
        :param sparse_embedding_model: The sparse embedding model
        :return: List of dictionary with indices and values
        :rtype: List[dict]
        """
        # Convert string to list of string
        if isinstance(query,str): query = [query]

        # Fastembed Sparse Embedding encode
        if isinstance(sparse_embedding_model, SparseTextEmbedding):
            # Get embedding
            sparse_embeddings = sparse_embedding_model.query_embed(query = query)
            # Normalize
            return  [embedding.as_dict() for embedding in list(sparse_embeddings)]
        elif isinstance(sparse_embedding_model, SpladeEmbeddingFunction):
            # Milvus Sparse embedding
            sparse_embeddings = sparse_embedding_model.encode_queries(queries = query)
            # Normalize
            return [BaseVectorStore._convert_csr_array_to_dict(embedding) for embedding in sparse_embeddings]
        # Doesnt support
        raise NotImplementedError("Sparse embedding currently support Milvus/Fastembed!")

    @staticmethod
    def _convert_upsert_data(documents: Sequence[Union[BaseNode,Document]]) -> List[dict]:
        """
        Construct the payload data from LlamaIndex document/node datatype
        :param documents: The list of LlamaIndex BaseNode objects
        :return: List[dict]
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

            # Verify object
            return [BaseNodeField.parse_obj(document.dict()).dict() for document in documents]
        else:
            # Langchain Document verify
            documents = [DocumentField.parse_obj(document.dict()).dict() for document in documents]
            # Add id_ value to dict
            for i in range(len(documents)): documents[i].update({default_keys[0]: str(uuid.uuid4())})
            return documents

    @staticmethod
    def _convert_response_to_node_with_score(responses: List[dict],
                                             remove_embedding: bool = True) -> Sequence[NodeWithScore]:
        """
        Convert response from searching to NodeWithScore Datatype (LlamaIndex)
        :param responses: Response for converting
        :param remove_embedding: Specify whether remove embedding from output or not
        :return: Sequence[NodeWithScore]
        """
        # Get node with format
        results = []
        for response in responses:
            # Get the main part
            temp = dict(response['entity'])
            # temp.update({"score": response['distance']})
            # Remove embedding
            if remove_embedding: temp.update({default_keys[1]:None})
            results.append(temp)

        # Define text nodes
        text_nodes = [TextNode.from_dict(result) for result in results]
        # Return NodeWithScore
        return [NodeWithScore(node = text_nodes[i],
                              score = responses[i]["distance"]) for i in range(len(responses))]

    @staticmethod
    def _convert_response_to_document(responses: List[dict],
                                      remove_embedding: bool = True) -> Sequence[Document]:
        """
        Convert response to Langchain Document format
        :param responses: Response for converting
        :param remove_embedding: Specify whether remove embedding from output or not
        :return: Sequence[Document]
        """
        # Get node with format
        results = []
        for response in responses:
            # Get the main part
            temp = dict(response['entity'])
            # temp.update({"score": response['distance']})
            # Remove embedding
            if remove_embedding: temp.update({default_keys[1]:None})
            results.append(temp)

        # Return Document
        return [Document.parse_obj(result) for result in results]

    @staticmethod
    def _convert_csr_array_to_dict(csr_array :scipy.sparse.csr_array) -> dict:
        """
        Convert csr array (From Milvus Sparse Embedding) to base dictionary format
        :param csr_array: Spicy csr array for converting
        :return: dict
        """
        return {indice:value for (indice, value) in zip(csr_array.indices,csr_array.data)}

    def _setup_collection_schema(self,
                                 document_type: str,
                                 dense_datatype :Literal["FLOAT_VECTOR","FLOAT16_VECTOR","BFLOAT16_VECTOR"] = "FLOAT_VECTOR",
                                 vector_dims :int = 768,
                                 enable_sparse :bool = False) -> CollectionSchema:
        """
        Create collection schema
        :param document_type: Type of document for indicating fields (BaseNode/ Document)
        :param dense_datatype: Including following type: FLOAT_VECTOR, FLOAT16_VECTOR , BFLOAT16_VECTOR for storing
        vectors.
        :param vector_dims: The dimension of vector (for dense vector)
        :param enable_sparse: Enable the sparse schema
        :return: CollectionSchema
        """
        # Define dense datatype
        if dense_datatype == "FLOAT_VECTOR":
            dense_datatype = DataType.FLOAT_VECTOR
        elif dense_datatype == "FLOAT16_VECTOR":
            dense_datatype = DataType.FLOAT16_VECTOR
        elif dense_datatype == "BFLOAT16_VECTOR":
            dense_datatype = DataType.BFLOAT16_VECTOR
        else:
            raise ValueError(f"Dense data type: {dense_datatype} is not compatible with dense vector field!")

        # Define schema
        schema = self.create_schema(
            auto_id = False
        )

        # Add default field
        # id_ field
        schema.add_field(field_name = default_keys[0],
                         datatype = DataType.VARCHAR,
                         is_primary = True,
                         max_length = 64)
        # embedding field
        schema.add_field(field_name = default_keys[1],
                         datatype = dense_datatype,
                         dim = vector_dims)

        # Add optional keys
        if document_type == "BaseNode":
            # BaseNode fields
            base_node_fields = list(BaseNodeField.model_fields.keys())

            # Add fields
            # Metadata field
            schema.add_field(field_name = base_node_fields[1],
                             datatype = DataType.JSON)
            # excluded_embed_metadata_keys field
            schema.add_field(field_name = base_node_fields[2],
                             datatype = DataType.ARRAY,
                             element_type = DataType.VARCHAR,
                             max_capacity = 16,
                             max_length = 64)

            # excluded_llm_metadata_keys
            schema.add_field(field_name = base_node_fields[3],
                             datatype = DataType.ARRAY,
                             element_type = DataType.VARCHAR,
                             max_capacity = 16,
                             max_length = 64)

            # relationships field
            schema.add_field(field_name = base_node_fields[4],
                             datatype = DataType.JSON)

            # text field
            schema.add_field(field_name = base_node_fields[5],
                             datatype = DataType.VARCHAR,
                             max_length = 16384)

            # mimetype field
            schema.add_field(field_name = base_node_fields[6],
                             datatype = DataType.VARCHAR,
                             max_length = 16)

            # start_char_idx field
            schema.add_field(field_name = base_node_fields[7],
                             datatype = DataType.INT64,
                             max_length = 8,
                             nullable = True)
            # end_char_idx
            schema.add_field(field_name = base_node_fields[8],
                             datatype = DataType.INT64,
                             max_length = 8,
                             nullable = True)
            # text_template field
            schema.add_field(field_name = base_node_fields[9],
                             datatype = DataType.VARCHAR,
                             max_length = 64)
            # metadata_template field
            schema.add_field(field_name = base_node_fields[10],
                             datatype = DataType.VARCHAR,
                             max_length = 32)
            # metadata_seperator field
            schema.add_field(field_name = base_node_fields[11],
                             datatype = DataType.VARCHAR,
                             max_length = 8)
        else:
            document_fields = list(DocumentField.model_fields.keys())

            # Add fields
            # Metadata field
            schema.add_field(field_name = document_fields[0],
                             datatype = DataType.JSON)
            # page content field
            schema.add_field(field_name = document_fields[1],
                             datatype = DataType.VARCHAR,
                             max_length = 16384)

        if enable_sparse:
            # Add sparse field
            schema.add_field(field_name = default_keys[2], datatype = DataType.SPARSE_FLOAT_VECTOR)
        return schema

    def _setup_collection_index(self,
                                index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT",
                                dense_search_metric :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                                params :Optional[dict] = None,
                                enable_sparse :bool = False,
                                sparse_index_type :Literal["SPARSE_INVERTED_INDEX","SPARSE_WAND"] = "SPARSE_INVERTED_INDEX",
                                sparse_params :Optional[dict] = None):
        """
        Define collection index (Index params dictate how Milvus organizes your data)
        :param index_algo: Name of the algorithm used to arrange data in the specific field ( FLAT,IVF_FLAT,etc).
        :param dense_search_metric: The algorithm that is used to measure similarity between vectors. Possible values are
        IP, L2, COSINE, JACCARD, HAMMING (For dense representation).
        :param params: The fine-tuning parameters for the specified dense index type.
        :param enable_sparse: Enable the sparse schema or not
        :param sparse_index_type: Index type using with sparse (SPARSE_INVERTED_INDEX ,SPARSE_WAND)
        :param sparse_params:  The fine-tuning parameters for the specified sparse index type.
        :return:
        """
        # Define index
        index_params = self.prepare_index_params()
        # Default dense params
        if params is None: params = {"nlist": 128}

        # Default sparse params
        if sparse_params is None: sparse_params = {"drop_ratio_build": 0.2}

        # Add id key
        index_params.add_index(field_name = default_keys[0])
        # Add dense key
        index_params.add_index(field_name = default_keys[1],
                               index_name = "dense_index",
                               index_type = index_algo,
                               metric_type = dense_search_metric,
                               params = params)

        # If enable sparse index
        if enable_sparse:
            index_params.add_index(
                field_name = default_keys[2],
                index_name = "sparse_index",
                index_type = sparse_index_type,
                metric_type = "IP", # Only Inner Product is used to measure the similarity between 2 sparse vectors.
                params = sparse_params,
            )
        return index_params

    def _create_collection(self,
                           document_type: str,
                           dimension_nums :int,
                           enable_sparse :bool = False) -> dict:
        """
        Create Milvus collection with defined setting
        :param document_type: Type of input document (BaseNode,Document)
        :param dimension_nums: Number of dimension for embedding
        :param enable_sparse: Enable the sparse schema or not
        :return: dict
        """
        # Define schema
        schema = self._setup_collection_schema(document_type = document_type,
                                               vector_dims = dimension_nums,
                                               dense_datatype = self._dense_datatype,
                                               enable_sparse = enable_sparse)

        # Define index
        index_params = self._setup_collection_index(index_algo = self._index_algo,
                                                    dense_search_metric = self._dense_search_metrics,
                                                    enable_sparse = enable_sparse)
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
        :param partition_name: Partition name
        :return: dict
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
        """

        :param query: A query for searching
        :param partition_names: Choose partition for retrieving. Default is current partition.
        :param limit:  Number of resulted responses.
        :param kwargs: Additional params
        :return:
        """
        raise NotImplementedError

    def insert_documents(self,
                         documents: Sequence[BaseNode],
                         partition_name: str):
        """
        Insert document to a specified partition of collection.

        :param documents: List of BaseNode.
        :param partition_name: Name of partition for inserting data
        :return:
        """
        raise NotImplementedError

    def collection_info(self):
        """
        Describe collection information
        :return:
        """
        raise NotImplementedError

    def list_partition(self) -> List[str]:
        """
        Return list of partition of defined collection
        :return: List[str]
        """
        # Check collection exist
        raise NotImplementedError
