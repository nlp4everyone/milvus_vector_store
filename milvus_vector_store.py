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
        :param db_name: Database name
        :type db_name: str
        :param token: Token information
        :type token: str
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
                         documents :Sequence[Union[BaseNode,Document]],
                         partition_name: str,
                         embedded_batch_size: int = 16,
                         embedded_num_workers: int = 4,
                         **kwargs) -> int:
        """
        Insert document to collection.

        :param documents: List of BaseNode.
        :type documents: Sequence[BaseNode]
        :param embedded_batch_size: Batch size for embedding model. Default is 64.
        :type embedded_batch_size: int
        :param embedded_num_workers: Batch size for embedding model (Optional). Default is None.
        :type embedded_num_workers: int
        """
        # When document is empty
        if len(documents) == 0:
            raise Exception("Document cant be empty!")

        # Check embedding
        if self._dense_embedding_model is None:
            raise ValueError("Please import embedding model!")

        # Get document type
        if isinstance(documents[0],Document):
            document_type = "Document"
        else:
            document_type = "BaseNode"

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
                                             batch_size = embedded_batch_size,
                                             num_workers = embedded_num_workers)
        # Update dense embedding to node
        for i in range(len(nodes)): nodes[i].update({default_keys[1]: dense_embeddings[i]})

        # When enable sparse
        if self._sparse_embedding_model is not None:
            # Embed sparse embedding
            sparse_embeddings = self._sparse_embed_texts(texts = contents,
                                                         sparse_embedding_model = self._sparse_embedding_model)
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
            self.__verify_collection_dimension(collection_name = self._collection_name,
                                               embedding_dimension = dimension_nums)


        # Insert to partition inside collection
        res = self.insert(collection_name = self._collection_name,
                          partition_name = partition_name,
                          data = nodes,
                          **kwargs)

        # Return number of inserted items
        return res['insert_count']

    def retrieve(self,
                 query :str,
                 partition_names: Union[str, List[str]],
                 limit :int = 3,
                 mode: Literal["dense", "sparse"] = "dense",
                 return_type :Literal["auto","BasePoints"] = "auto",
                 **kwargs) -> Union[Sequence[Union[NodeWithScore,Document]],Sequence[dict]]:
        """
        Finding relevant contexts from initial question.

        :param query: A query for searching
        :type query: str
        :param partition_names: Choose partition for retrieving. Default is current partition.
        :type partition_names: Optional[list[str]]
        :param limit: Number of resulted responses.
        :type limit: int
        :param return_type: Return object
        :return: Union[Sequence[NodeWithScore],Sequence[dict]]
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
            query_embedding = self._embed_query(query = query,
                                                embedding_model = self._dense_embedding_model)
            # Verify embedding size
            self.__verify_collection_dimension(collection_name = self._collection_name,
                                               embedding_dimension = len(query_embedding))
            # Dense anns fields
            anns_field = default_keys[1]
            # Add outside dimension
            query_embedding = [query_embedding]
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
                              limit = limit,
                              output_fields = collection_fields,
                              search_params = search_metrics,
                              **kwargs)

        # Convert to list
        results = results[0]
        # Return
        if return_type == "BasePoints":
            # Default
            return results
        # Auto mode
        # Document case
        if results[0]["entity"].get("page_content"):
            return self._convert_response_to_document(responses = results)
        # NodeWithScore case
        return self._convert_response_to_node_with_score(responses = results)

    def collection_info(self):
        """
        Return collection info
        """

        # Check collection exist
        if not self.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")
        # Return information
        return self.describe_collection(collection_name=self._collection_name)

    def list_partition(self):
        # Check collection exist
        if not self.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")

        # Return information
        return self.list_partitions(collection_name = self._collection_name)

    def __verify_collection_dimension(self,
                                      collection_name :str,
                                      embedding_dimension :int):
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







