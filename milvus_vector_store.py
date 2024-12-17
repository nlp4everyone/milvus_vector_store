# Typing
from typing import Optional, Union, List, Sequence, Literal
from llama_index.core.schema import BaseNode, NodeWithScore
from langchain_core.documents.base import Document
# Dense Embedding
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.embeddings import Embeddings
# Sparse Embedding
from fastembed import SparseTextEmbedding
from milvus_model.sparse.splade import SpladeEmbeddingFunction
# Base vector store
from .base import BaseVectorStore, default_keys
# Milvus components
from pymilvus import (AnnSearchRequest,
                      WeightedRanker,
                      RRFRanker)

# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16

class MilvusVectorStore(BaseVectorStore):
    def __init__(self,
                 dense_embedding_model: Union[BaseEmbedding,Embeddings],
                 sparse_embedding_model: Optional[Union[SparseTextEmbedding,SpladeEmbeddingFunction]] = None,
                 collection_name: str = "milvus_vector_store",
                 uri: str = "http://localhost:19530",
                 user :str = "",
                 password :str = "",
                 db_name :str = "",
                 token :str = "",
                 dense_search_metrics :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                 index_algo: Literal["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "SCANN"] = "IVF_FLAT",
                 **kwargs) -> None:

        """
        Init Milvus client service

        :param collection_name: The name of collection (Required).
        :type collection_name: str
        :param uri: The Milvus url string.
        :type uri: str
        :param user: User information
        :type user: str
        :param password: Password
        :type password: str
        :param db_name: Database name
        :type db_name: str
        :param token: Token information
        :type token: str
        :param dense_search_metrics: The algorithm that is used to measure similarity between vectors. Possible values are
        IP, L2, COSINE, JACCARD, HAMMING (For dense representation).
        :type dense_search_metrics: str
        :param token: Name of the algorithm used to arrange data in the specific field ( FLAT,IVF_FLAT,etc).
        :type token: str
        :param kwargs: Additional params
        """
        assert collection_name, "Collection name must be string"
        # Inheritance
        super().__init__(collection_name = collection_name,
                         uri = uri,
                         user = user,
                         password = password,
                         db_name = db_name,
                         token = token,
                         dense_search_metrics = dense_search_metrics,
                         index_algo = index_algo,
                         **kwargs)
        # Dense model
        self._dense_embedding_model = dense_embedding_model
        # Sparse model
        self._sparse_embedding_model = sparse_embedding_model

    def insert_documents(self,
                         documents :List[Union[BaseNode,Document]],
                         partition_name: str,
                         dense_batch_size: int = 16,
                         dense_num_workers: int = 4,
                         sparse_batch_size: int = 16,
                         sparse_parralel: Optional[int] = None,
                         uploading_batch_size:int = 4,
                         **kwargs):
        """
        Insert document to a specified partition of collection.

        :param documents: List of BaseNode.
        :type documents: Sequence[BaseNode]
        :param partition_name: Name of partition for inserting data
        :type partition_name: str
        :param dense_batch_size: Batch size for embedding model. Default is 64.
        :type dense_batch_size: int
        :param dense_num_workers: Batch size for embedding model (Optional). Default is None.
        :type dense_num_workers: int
        """
        # When document is empty
        if len(documents) == 0:
            raise Exception("Document cant be empty!")

        # Check embedding
        if self._dense_embedding_model is None:
            raise ValueError("Please import embedding model!")

        # Get document type
        document_type = "BaseNode" if isinstance(documents[0],BaseNode) else "Document"

        # Convert BaseNode to Dict and remove redundant information
        nodes = self._convert_upsert_data(documents = documents)

        # Get content
        if isinstance(documents[0], BaseNode):
            # LlamaIndex BaseNode case
            contents = [doc.get_content() for doc in documents]
        else:
            # Langchain Document case
            contents = [doc.page_content for doc in documents]

        # Get dense embeddings
        dense_embeddings = self._embed_texts(texts = contents,
                                             embedding_model = self._dense_embedding_model,
                                             batch_size = dense_batch_size,
                                             num_workers = dense_num_workers)
        # Update dense embedding to node
        for i in range(len(nodes)): nodes[i].update({default_keys[1]: dense_embeddings[i]})

        # When enable sparse
        if self._sparse_embedding_model is not None:
            # Embed sparse embedding
            sparse_embeddings = self._sparse_embed_texts(texts = contents,
                                                         sparse_embedding_model = self._sparse_embedding_model,
                                                         batch_size = sparse_batch_size,
                                                         parallel = sparse_parralel)
            # Add sparse embedding to dictionary
            for i in range(len(nodes)): nodes[i].update({default_keys[2]: sparse_embeddings[i]})

        # Get dimension nums
        dimension_nums = len(nodes[0][default_keys[1]])

        # Check collection existence
        if not self.has_collection(collection_name = self._collection_name):
            # When enable sparse
            enable_sparse = True if self._sparse_embedding_model is not None else False
            # Create collection if doesnt exist
            self._create_collection(document_type = document_type,
                                    dimension_nums = dimension_nums,
                                    enable_sparse = enable_sparse)

        # Create partition if doesnt exist
        if not self.has_partition(collection_name = self._collection_name,
                                  partition_name = partition_name):
            # Create partition
            self._create_partition(partition_name = partition_name)
        else:
            # Verify dense dimension of collection
            self.__verify_collection_dimension(collection_name = self._collection_name,
                                               embedding_dimension = dimension_nums)

        # Iterate over the data with batch
        for i in range(0, len(nodes), uploading_batch_size):
            # Insert to partition inside collection
            res = self.insert(collection_name = self._collection_name,
                              partition_name = partition_name,
                              data = nodes[i:i + uploading_batch_size],
                              **kwargs)

    def retrieve(self,
                 query :Union[str,List[str]],
                 partition_names: Union[str, List[str]],
                 similarity_top_k :int = 3,
                 search_filter :str = "",
                 mode: Literal["dense", "sparse"] = "dense",
                 return_type :Literal["auto","BasePoints"] = "auto",
                 **kwargs) -> Union[Sequence[Union[NodeWithScore,Document,dict]]]:
        """
        Finding relevant contexts from initial question.

        :param query: A query or list of query for searching
        :type query: Union[str,List[str]]
        :param partition_names: Choose partition for retrieving. Default is current partition.
        :type partition_names: Optional[list[str]]
        :param similarity_top_k: Number of resulted responses.
        :type similarity_top_k: int
        :param search_filter: Conditional filter apply to search function
        :type search_filter: str
        :param mode: Type of ANN algorithms for searching (dense/sparse)
        :type mode: Literal["dense", "sparse"]
        :param return_type: Desired object for return (BasePoint or auto)
        :param kwargs: Additional params
        :return: Union[Sequence[Union[NodeWithScore,Document,dict]]]
        """
        # Default partition
        if isinstance(partition_names,str) : partition_names = [partition_names]
        # Check embedding
        if self._dense_embedding_model is None:
            raise ValueError("Please import embedding model!")

        # Get partition of collection name
        list_partitions = self.list_partition()
        # Check condition
        is_accepted = set(partition_names).issubset(set(list_partitions))
        if not is_accepted:
            wrong_partition = ",".join(partition_names)
            raise ValueError(f"Partitions: {wrong_partition} not existed!")

        # Get collection info
        collection_info = self.collection_info()
        # Get field name from collection
        collection_fields = [field["name"] for field in collection_info["fields"]]

        # Search metrics
        search_metrics = {"metric_type": self._dense_search_metrics}
        # Search with dense embedding
        if mode == "dense":
            # Get dense query embedding
            query_embedding = self._embed_query(queries = query,
                                                embedding_model = self._dense_embedding_model)
            # Verify embedding size
            self.__verify_collection_dimension(collection_name = self._collection_name,
                                               embedding_dimension = len(query_embedding[0]))
            # Dense anns fields
            anns_field = default_keys[1]
        else:
            # Sparse embedding case
            if default_keys[2] not in collection_fields:
                raise Exception(f"Collection: {self._collection_name} not supports sparse embedding")

            # Get sparse query embedding
            query_embedding = self._sparse_embed_query(query = query,
                                                       sparse_embedding_model = self._sparse_embedding_model)

            # Sparse anns fields
            anns_field = default_keys[2]
            # Change metric type for sparse searching
            search_metrics.update({"metric_type": "IP"})

        # Get the retrieved text
        results = self.search(collection_name = self._collection_name,
                              partition_names = partition_names,
                              anns_field = anns_field,
                              data = query_embedding,
                              filter = search_filter,
                              limit = similarity_top_k,
                              output_fields = collection_fields,
                              search_params = search_metrics,
                              **kwargs)

        # Return
        return self.__convert_retrieval_nodes(results, return_type)

    def hybrid_query(self,
                     query: Union[str,List[str]],
                     ranker :Union[RRFRanker,WeightedRanker],
                     partition_names: Union[str, List[str]],
                     dense_similarity_top_k: int = 3,
                     sparse_similarity_top_k: int = 3,
                     rerank_similarity_top_k :int = 3,
                     dense_params :Optional[dict] = None,
                     sparse_params: Optional[dict] = None,
                     return_type: Literal["auto", "BasePoints"] = "auto",
                     **kwargs) -> Union[Sequence[Union[NodeWithScore, Document, dict]]]:
        """
        Perform hybrid search over collection with partition names
        :param query: A query or list of query for searching
        :param ranker: Reranking strategy options. Only available with Milvus Reranker (WeightedRanker,RRFRanker)
        :param partition_names: Choose partition for retrieving. Default is current partition.
        :param dense_similarity_top_k: Number of dense responses from searching.
        :param sparse_similarity_top_k: Number of sparse responses from searching.
        :param rerank_similarity_top_k: Number of resulted responses after reranking.
        :param dense_params: Optional parameter for dense retrieval
        :param sparse_params: Optional parameter for sparse retrieval
        :param return_type: return_type: Desired object for return (BasePoint or auto)
        :param kwargs: Additional params
        :return: Union[Sequence[Union[NodeWithScore,Document,dict]]]
        """
        # Check embedding model
        if self._dense_embedding_model is None:
            raise Exception("Dense embedding model is empty!")

        # Check sparse embedding model
        if self._sparse_embedding_model is None:
            raise Exception("Sparse embedding model is empty!")

        # Get collection info
        collection_info = self.collection_info()
        # Get field name from collection
        collection_fields = [field["name"] for field in collection_info["fields"]]

        # Check collection config for vector
        if default_keys[1] not in collection_fields:
            raise ValueError("Please config dense embedding first!")
        if default_keys[2] not in collection_fields:
            raise ValueError("Please config sparse embedding first!")

        # Convert string to list string
        if isinstance(partition_names, str): partition_names = [partition_names]

        # Get dense query embedding
        query_dense_vector = self._embed_query(query,
                                               embedding_model = self._dense_embedding_model)
        # Get sparse query embedding
        query_sparse_vector = self._sparse_embed_query(query = query,
                                                       sparse_embedding_model = self._sparse_embedding_model)

        # Define search params
        dense_search_param = {
            "data": query_dense_vector,
            "anns_field": default_keys[1],
            "param": {
                "metric_type": self._dense_search_metrics,
                "params": {"nprobe": 10} if dense_params is None else dense_params
            },
            "limit": dense_similarity_top_k
        }
        sparse_search_param = {
            "data": query_sparse_vector,
            "anns_field": default_keys[2],
            "param": {
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2} if sparse_params is None else sparse_params
            },
            "limit": sparse_similarity_top_k
        }

        # Get result
        results = self.hybrid_search(collection_name = self._collection_name,
                                     partition_names = partition_names,
                                     reqs = [AnnSearchRequest(**dense_search_param),AnnSearchRequest(**sparse_search_param)],
                                     ranker = ranker,
                                     limit = rerank_similarity_top_k,
                                     output_fields = collection_fields,
                                     **kwargs)
        # Return
        return self.__convert_retrieval_nodes(results, return_type)

    def collection_info(self):
        """
        Describe collection information
        :return:
        """

        # Check collection exist
        if not self.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")
        # Return information
        return self.describe_collection(collection_name=self._collection_name)

    def list_partition(self) -> List[str]:
        """
        Return list of partition of defined collection
        :return:
        """
        # Check collection exist
        if not self.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")

        # Return information
        return self.list_partitions(collection_name = self._collection_name)

    def __convert_retrieval_nodes(self,
                                  results,
                                  return_type: Literal["auto", "BasePoints"] = "auto"):
        """
        Normalize node with specific type
        :param results: Result from searching nodes
        :param return_type: Desired result. Default is auto
        :return:
        """
        results = list(results)
        # Check len
        if len(results) == 0:
            # Return empty list
            return []
        # Convert result to list
        # Desired output
        for result in results:
            # Remove embedding
            for i in range(len(result)):
                result[i]["entity"].update({default_keys[1]: None,
                                            default_keys[2]: None})
        # Return BasePoint
        if return_type == "BasePoints":
            return results

        final_output = []
        # Auto mode
        for result in results:
            # Auto mode
            if result[0]["entity"].get("page_content"):
                # Document case
                final_output.append(self._convert_response_to_document(responses = result))
            else:
                final_output.append(self._convert_response_to_node_with_score(responses = result))
        return final_output

    def __verify_collection_dimension(self,
                                      collection_name :str,
                                      embedding_dimension :int) -> None:
        """
        Verify config of dense representation index
        :param collection_name: The collection name
        :param embedding_dimension: Embedding dimension
        :return: None
        """
        # Check partition stats
        stats = self.describe_collection(collection_name = collection_name)

        # Embedding stats
        collection_stats = dict(stats).get("fields")
        if collection_stats is None:
            raise ValueError("Fields not existed in collection")
        # Stats
        collection_stats = [stats for stats in collection_stats if dict(stats).get("name") == default_keys[1]]
        if len(collection_stats) == 0:
            raise ValueError("Empty embedding field!")

        # Params
        collection_params = dict(collection_stats[0]).get("params")
        if collection_params is None:
            raise ValueError("Empty params field!")
        # Dims
        collection_dims = dict(collection_params).get("dim")
        if collection_dims is None:
            raise ValueError("Empty dim field!")

        # Check type
        if not isinstance(collection_dims,int):
            raise ValueError("Collection dims must be integer!")
        # Check dims
        if embedding_dimension != collection_dims:
            raise ValueError(f"Embed dimension ({embedding_dimension}) is differ with default collection dimension ({collection_dims})")







