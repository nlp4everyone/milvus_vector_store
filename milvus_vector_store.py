from pymilvus import MilvusClient, CollectionSchema, DataType
from typing import Optional, Union, List, Sequence, Literal
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.embeddings import BaseEmbedding
from .base import BaseVectorStore, node_with_score_fields

# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16

class MilvusVectorStore(BaseVectorStore):
    def __init__(self,
                 dense_embedding_model: BaseEmbedding = None,
                 collection_name: str = "milvus_vector_store",
                 uri: str = "http://localhost:19530",
                 user :str = "",
                 password :str = "",
                 db_name :str = "",
                 token :str = "",
                 search_metrics :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                 index_algo: Literal["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "SCANN"] = "IVF_FLAT") -> None:

        """
        Init Milvus client service

        :param collection_name: The name of collection (Required).
        :type collection_name: str
        :param url: The Milvus url string.
        :type url: str
        :param user: User information
        :type user: str
        :param db_name: Database name
        :type db_name: str
        :param token: Token information
        :type token: str
        """
        assert collection_name, "Collection name must be string"
        # Inheritance
        super().__init__(dense_embedding_model = dense_embedding_model,
                         collection_name = collection_name,
                         uri = uri,
                         search_metrics = search_metrics,
                         index_algo = index_algo)

        # Private value
        self.__user = user
        self.__password = password
        self.__db_name = db_name
        self.__token = token

        # init client
        self._client = MilvusClient(uri = self._uri,
                                    user = self.__user,
                                    password = self.__password,
                                    db_name = self.__db_name,
                                    token = self.__token)



    def insert_documents(self,
                         documents :Sequence[BaseNode],
                         partition_name: str,
                         embedded_batch_size: int = 16,
                         embedded_num_workers: int = 4) -> int:
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

        list_partitions = self.list_partition()
        # Check condition
        is_accepted = set(partition_name).issubset(set(list_partitions))
        if not is_accepted:
            raise ValueError(f"Partition: {partition_name} not existed!")

        # Convert BaseNode to Dict and remove redundant information
        nodes = self._convert_upsert_data(documents = documents)
        # Get content (text information) from node
        contents = [node['text'] for node in nodes]

        # Get embeddings
        if isinstance(self._dense_embedding_model,BaseEmbedding):
            # Return vector representation (Embedding arrays)
            embeddings = self._get_embeddings(texts = contents,
                                              embedding_model = self._dense_embedding_model,
                                              batch_size = embedded_batch_size,
                                              num_workers = embedded_num_workers)
            # Add embedding to node
            for i in range(len(nodes)): nodes[i].update({"embedding" : embeddings[i]})

        # Get dimension nums
        dimension_nums = len(nodes[0]['embedding'])

        try:
            # Create collection if doesnt exist
            self._create_collection(dimension_nums = dimension_nums)

            # Create partition if doesnt exist
            self._create_partition(partition_name = partition_name)

            # Insert to partition inside collection
            res = self._client.insert(collection_name = self._collection_name,
                                      partition_name = partition_name,
                                      data = nodes)
        except Exception as e:
            raise Exception(e)

        # Return number of inserted items
        return res['insert_count']

    def retrieve(self,
                 query :str,
                 partition_names: Union[str, List[str]],
                 limit :int = 3,
                 return_type :Literal["auto","Node_With_Score"] = "NodeWithScore",
                 **kwargs) -> Union[Sequence[NodeWithScore],Sequence[dict]]:
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

        # Get query embedding
        query_embedding = None
        # Get query representation from BaseEmbedding model
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            query_embedding = self._dense_embedding_model.get_query_embedding(query = query)

        # Get the retrieved text
        results = self._client.search(
            collection_name = self._collection_name,
            partition_names = partition_names,
            data = [query_embedding],
            limit = limit,
            output_fields = node_with_score_fields,
            search_params = {"metric_type": self._search_metrics},
        )

        # Convert to list
        results = results[0]

        # Return
        if return_type == "auto":
            # Default output
            return results
        else:
            # Convert back to NodeWithScore
            return self._convert_response_to_node_with_score(responses = results)

    def collection_info(self):
        """
        Return collection info
        """

        # Check collection exist
        if not self._client.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")
        # Return information
        return self._client.describe_collection(collection_name=self._collection_name)

    def list_partition(self):
        # Check collection exist
        if not self._client.has_collection(self._collection_name):
            raise Exception(f"Collection {self._collection_name} is not exist!")
        # Return information
        return self._client.list_partitions(collection_name = self._collection_name)



