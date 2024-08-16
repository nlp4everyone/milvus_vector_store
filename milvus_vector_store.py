from pymilvus import MilvusClient, CollectionSchema, DataType
from typing import Optional, Union, List, Sequence, Literal
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.embeddings import BaseEmbedding

# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16

class MilvusVectorStore:
    def __init__(self,
                 collection_name :str,
                 partition_name :str,
                 url :str = "http://localhost:19530",
                 user :str = "",
                 password :str = "",
                 db_name :str = "",
                 token :str = "",
                 search_metrics :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                 embedding_model :Optional[BaseEmbedding] = None) -> None:

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
        self._client = MilvusClient(url = url,
                                    user = user,
                                    password = password,
                                    db_name = db_name,
                                    token = token)
        # Collection name
        self._collection_name = collection_name
        self._partition_name = partition_name
        self._search_metrics = search_metrics

        # Hybrid search
        self.__embedding_model = embedding_model


    # def _set_hybrid_mode(self, enable :bool = False):
    #     """Set hybrid mode if enabled"""
    #     # Enable hybrid search


    # def _set_dense_model(self):
    #     """Set local embedding model (FastEmbed) if enabled"""


    # def _set_cache_collection(self):
    #     """Enable when set semantic cache search"""
    #     # Define cache collection name

    # def _semantic_cache_search(self,
    #             return nodes

    def __setup_collection_schema(self,
                                  vector_type :DataType = DataType.FLOAT_VECTOR,
                                  vector_dims :int = 768) -> CollectionSchema:
        """
        Create collection schema

        Args:
             vector_type (DataType): Type of vector
             vector_dims (int): Numbers for embedding dimension.
        """
        schema = self._client.create_schema(
            auto_id = False,
            enable_dynamic_field = True,
        )
        # Add field
        schema.add_field(field_name = "id_", datatype = DataType.VARCHAR, is_primary = True, max_length = 64)
        schema.add_field(field_name = "embedding", datatype = vector_type, dim = vector_dims)
        return schema

    def __setup_collection_index(self,
                                 index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT" ,
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
        #
        index_params = self._client.prepare_index_params()
        if params == None:
            params = {"nlist": 128}

        # Add indexes
        index_params.add_index(field_name = "id_")
        index_params.add_index(field_name = "embedding", index_type = index_algo, metric_type = search_metric, params = params)
        return index_params

    def __create_collection(self, dimension_nums :int) -> dict:
        """
        Create collection with default name

        Args:
            dimension_nums: Number of dimension for embedding (int)
        """

        # When collection isnt established
        if not self._client.has_collection(collection_name = self._collection_name):
            # Define schema
            schema = self.__setup_collection_schema(vector_dims = dimension_nums)
            # Define index
            index_params = self.__setup_collection_index(index_algo = "IVF_FLAT",
                                                         search_metric = self._search_metrics)
            # Collection for LlamaIndex payloads
            self._client.create_collection(collection_name = self._collection_name,
                                           schema = schema,
                                           index_params = index_params)

            # Return state
            res = self._client.get_load_state(
                collection_name = self._collection_name
            )
            return res

    def __create_partition(self) -> dict:
        """
        Create partition from predefined name.
        """
        assert self._partition_name, "Collection name must be a string"
        # Check whether partition is existed or not.
        if not self._client.has_partition(collection_name = self._collection_name, partition_name = self._partition_name):
            # Create collection
            self._client.create_partition(collection_name = self._collection_name,
                                          partition_name = self._partition_name)
            # Return state
            res = self._client.get_load_state(
                collection_name = self._collection_name
            )
            return res

    def search(self,
               query :str,
               partition_names :Optional[list[str]] = None,
               limit :int = 3,
               return_type :Literal["default","node_with_score"] = "node_with_score") -> Union[Sequence[NodeWithScore],Sequence[dict]]:
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
        if partition_names == None: partition_names = [self._partition_name]
        # Get query embedding
        query_embedding = None
        # Get query representation from BaseEmbedding model
        if isinstance(self.__embedding_model, BaseEmbedding):
            query_embedding = self.__embedding_model.get_query_embedding(query = query)

        # Define output
        # For BaseNode case
        base_node_fields = ['id_', 'embedding', 'metadata', 'excluded_embed_metadata_keys', 'excluded_llm_metadata_keys',
                         'relationships', 'text', 'mimetype', 'start_char_idx', 'end_char_idx', 'text_template',
                         'metadata_template', 'metadata_seperator', 'class_name']

        # Get the retrieved text
        results = self._client.search(
            collection_name = self._collection_name,
            partition_names = partition_names,
            data = [query_embedding],
            limit = limit,
            output_fields = base_node_fields,
            search_params = {"metric_type": self._search_metrics},
        )

        # Convert to list
        results = results[0]

        # Return
        if return_type == "default":
            # Default output
            return results
        else:
            # Convert back to NodeWithScore
            return self._convert_response_to_node_with_score(responses = results)

    def insert_documents(self,
                         documents :Sequence[BaseNode],
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

        # Convert BaseNode to Dict and remove redundant information
        nodes = self._convert_upsert_data(documents = documents)
        # Get content (text information) from node
        contents = [node['text'] for node in nodes]

        # Get embeddings
        if isinstance(self.__embedding_model,BaseEmbedding):
            # Return vector representation (Embedding arrays)
            embeddings = self.__get_embeddings(texts = contents,
                                              embedding_model = self.__embedding_model,
                                              batch_size = embedded_batch_size,
                                              num_workers = embedded_num_workers)
            # Add embedding to node
            for i in range(len(nodes)): nodes[i].update({"embedding" : embeddings[i]})

        # Get dimension nums
        dimension_nums = len(nodes[0]['embedding'])

        # Create collection if doesnt exist
        self.__create_collection(dimension_nums = dimension_nums)

        # Create partition if doesnt exist
        self.__create_partition()

        # Insert to partition inside collection
        res = self._client.insert(collection_name = self._collection_name,
                                  partition_name = self._partition_name,
                                  data = nodes)
        # Return number of inserted items
        return res['insert_count']

    # def update_point(self, id, vector):
    #     """Update value for points"""
    #     result = self._client.update_vectors(
    #         collection_name = self.collection_name,
    #         points = [
    #             models.PointVectors(
    #                 id = id,
    #                 vector = vector
    #             )])
    #     print(result)

    def collection_info(self, collection_name: str):
        """
        Return collection info
        """

        # Check collection exist
        if not self._client.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} is not exist!")
        # Return information
        return self._client.describe_collection(collection_name = collection_name)

    @staticmethod
    def __get_embeddings(texts: list[str],
                         embedding_model: BaseEmbedding,
                         batch_size: int,
                         num_workers: int,
                         show_progress: bool = True) -> List[Embedding]:
        """
        Return embedding from documents

        Args:
            texts (list[str]): List of input text
            embedding_model (BaseEmbedding): The text embedding model
            batch_size (int): The desired batch size
            num_workers (int): The desired num workers
            show_progress (bool): Indicate show progress or not

        Returns:
             Return list of Embedding
        """
        # Set batch size and num workers
        embedding_model.num_workers = num_workers
        embedding_model.embed_batch_size = batch_size
        # Return embedding
        return embedding_model.get_text_embedding_batch(texts = texts,
                                                        show_progress = show_progress)

    @staticmethod
    def _convert_upsert_data(documents: Sequence[BaseNode]) -> list[dict]:
        """
        Construct the payload data from LlamaIndex document/node datatype

        Args:
            documents (BaseNode): The list of BaseNode datatype in LlamaIndex

        Returns:
            Payloads (list[dict).
        """

        # Clear private data from payload
        for i in range(len(documents)):
            documents[i].embedding = None
            # Pop file path
            documents[i].metadata.pop("file_path")
            # documents[i].excluded_embed_metadata_keys = []
            # documents[i].excluded_llm_metadata_keys = []
            # Remove metadata in relationship
            for key in documents[i].relationships.keys():
                documents[i].relationships[key].metadata = {}

        # Get payloads
        payloads = [document.dict() for document in documents]
        return payloads

    @staticmethod
    def _convert_response_to_node_with_score(responses: List[dict],
                                             remove_embedding :bool = True) -> Sequence[NodeWithScore]:
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
        return [NodeWithScore(node = text_nodes[i], score = responses[i]["distance"]) for i in range(len(responses))]