# Typing
from typing import Optional, Union, List, Sequence, Literal
from llama_index.core.schema import ImageNode, NodeWithScore
# Image Embedding
from fastembed import ImageEmbedding
from .multimodal_embedding import SentenceTransformerEmbedding
# Image component
from PIL import Image
# Base vector store
from .base import BaseVectorStore, default_keys
# Component
from .utils import ImageUtils
import uuid
# Pymilvus
from pymilvus import (CollectionSchema,
                      DataType)
# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16
# Additional field for ImageNode
image_node_field = list(ImageNode.model_fields.keys())

class ImageMilvusVectorStore(BaseVectorStore):
    def __init__(self,
                 embedding_model: Union[ImageEmbedding,SentenceTransformerEmbedding],
                 text_captioning_mode = None,
                 collection_name: str = "image_milvus",
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
        # Image model
        self._embedding_model = embedding_model
        # Text model
        # Text captioning mode
        self._text_captioning_model = text_captioning_mode

    def insert_images(self,
                      images :List[Union[ImageNode,str]],
                      partition_name: str,
                      embedded_batch_size: int = 16,
                      embedded_num_workers: int = 4,
                      uploading_batch_size :int = 4,
                      **kwargs):
        """
        Insert document to a specified partition of collection.

        :param images: List of image path or ImageNode
        :type images: Sequence[Union[ImageNode,str]]
        :param partition_name: Name of partition for inserting data
        :type partition_name: str
        :param embedded_batch_size: Batch size for embedding model. Default is 64.
        :type embedded_batch_size: int
        :param embedded_num_workers: Batch size for embedding model (Optional). Default is None.
        :type embedded_num_workers: int
        """
        # When document is empty
        if len(images) == 0:
            raise Exception("Images cant be empty!")

        # Default value
        image_nodes = images

        if isinstance(images[0],str):
            # Path case
            # Filter only images
            images_url, images_path = ImageUtils.get_image_path(file_paths = images)
            # Get mimetype
            images_mimetype = ImageUtils.get_image_mimetype(images = images_url)
            # Convert to PIL Image
            pil_images = [Image.open(url) for url in images_url]

            # Get encodings
            image_embeddings = self._embed_images(images = pil_images,
                                                  batch_size = embedded_batch_size,
                                                  num_workers = embedded_num_workers)

            # Construct nodes
            image_nodes = [ImageNode(id_ = str(uuid.uuid4()),
                                     image_mimetype = images_mimetype[i],
                                     image_path = images_path[i],
                                     image_url = images_url[i],
                                     embedding = image_embeddings[i]).dict() for i in range(len(images_url))]

        # Enable text embedding model
        if self._text_captioning_model is None:
            for i in range(len(image_nodes)):
                # Drop text embedding field
                if image_nodes[i].get(image_node_field[18]) is None:
                    image_nodes[i].pop(image_node_field[18])

        # Get dimension nums
        dimension_nums = len(image_nodes[0]["embedding"])

        # Check collection existence
        if not self.has_collection(collection_name = self._collection_name):
            # Create collection if doesnt exist
            self._create_collection(document_type = "ImageNode",
                                    dimension_nums = dimension_nums)

        # Verify dense dimension of collection
        self._verify_collection_dimension(collection_name = self._collection_name,
                                          embedding_dimension = dimension_nums)

        # Create partition if doesnt exist
        if not self.has_partition(collection_name = self._collection_name,
                                  partition_name = partition_name):
            # Create partition
            self._create_partition(partition_name = partition_name)


        # Iterate over the data with batch
        for i in range(0, len(image_nodes), uploading_batch_size):
            # Insert to partition inside collection
            res = self.insert(collection_name = self._collection_name,
                              partition_name = partition_name,
                              data = image_nodes[i:i + uploading_batch_size],
                              **kwargs)

    def retrieve(self,
                 queries :Union[str,List[str],Image.Image,List[Image.Image]],
                 partition_names: Union[str, List[str]],
                 similarity_top_k :int = 3,
                 search_filter :str = "",
                 mode: Literal["embedding", "text_embedding"] = "embedding",
                 return_type :Literal["auto","BasePoints"] = "auto",
                 **kwargs) -> Union[Sequence[dict]]:
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

        # Get partition of collection name
        list_partitions = self.list_partition()
        # Check condition
        is_accepted = set(partition_names).issubset(set(list_partitions))
        if not is_accepted:
            wrong_partition = ",".join(partition_names)
            raise ValueError(f"Partitions: {wrong_partition} not existed!")

        # Convert string or Image query to list
        if isinstance(queries,str) or isinstance(queries,Image.Image):
            queries = [queries]

        # Search metrics
        search_metrics = {"metric_type": self._dense_search_metrics}
        # Get collection info
        collection_info = self.collection_info()
        # Get field name from collection
        collection_field_keys = [field["name"] for field in collection_info["fields"]]
        # Fields
        collection_field = collection_info.get("fields")

        # Check type
        is_image = isinstance(queries[0],Image.Image)

        # Image case
        if is_image:
            # Image convert
            query_embeddings = self._embed_images(images = queries,
                                                  batch_size = 2)

        else:
            # Text query case
            query_embeddings = self._embed_query(texts = queries)
        # Get query dimension
        query_dims = len(query_embeddings[0])

        # Get vector field
        vector_fields = [field for field in collection_field if dict(field).get("name") == mode]
        # Check field
        if len(vector_fields) == 0:
            raise ValueError(f"Field: {mode} not existed in collection schema")
        # Get unique vector field
        vector_dim :int = dict(dict(vector_fields[0]).get("params")).get("dim")

        # Check dimension between query embedding and schema embedding
        if vector_dim != query_dims:
            raise ValueError(f"Query embedding: {query_dims} is different with schema vector {vector_dim}")

        # # Get the retrieved text
        results = self.search(collection_name = self._collection_name,
                              partition_names = partition_names,
                              anns_field = mode,
                              data = query_embeddings,
                              filter = search_filter,
                              limit = similarity_top_k,
                              output_fields = collection_field_keys,
                              search_params = search_metrics,
                              **kwargs)

        # Return
        return self.__convert_retrieval_nodes(results = results,
                                              return_type = return_type)

    def collection_info(self) -> dict:
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

    def _embed_images(self,
                      images :List[Image.Image],
                      batch_size: int = 8,
                      parralel :Optional[int] = None,
                      **kwargs) -> List[List[float]]:
        # SentenceTransformer
        if isinstance(self._embedding_model,SentenceTransformerEmbedding):
            return self._embedding_model.embed_images(images = images,
                                                      batch_size = batch_size,
                                                      **kwargs)
        elif isinstance(self._embedding_model,ImageEmbedding):
            enbeddings = self._embedding_model.embed(images = images,
                                                     batch_size = batch_size,
                                                     parallel = parralel,
                                                     **kwargs)
            return [list(embedding) for embedding in enbeddings]
        else:
            raise Exception("Wrong embedding model")

    def _embed_query(self,
                     texts :List[str],
                     batch_size: int = 8,
                     parralel :Optional[int] = None,
                     **kwargs) -> List[List[float]]:
        # SentenceTransformer
        if isinstance(self._embedding_model,SentenceTransformerEmbedding):
            return self._embedding_model.embed_documents(texts = texts,
                                                         batch_size = batch_size,
                                                         **kwargs)
        # Other case
        raise Exception("Please use Sentences Embedding model for querying by text!")

    def _setup_collection_schema(self,
                                 document_type: str,
                                 image_vector_dims: int,
                                 text_vector_dims: Optional[int] = None,
                                 image_datatype :Literal["FLOAT_VECTOR","FLOAT16_VECTOR","BFLOAT16_VECTOR"] = "FLOAT_VECTOR",
                                 text_datatype: Literal["FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"] = "FLOAT_VECTOR"
                                 ) -> CollectionSchema:
        # Overide _setup_collection_schema function
        schema = super()._setup_collection_schema(document_type = "ImageNode",
                                                  dense_datatype = image_datatype,
                                                  vector_dims = image_vector_dims)
        # Get text datatype
        text_datatype = self._get_datatype(text_datatype)

        # Image field
        schema.add_field(field_name = image_node_field[14],
                         datatype = DataType.VARCHAR,
                         max_length = 256,
                         nullable = True)
        # Image path field
        schema.add_field(field_name = image_node_field[15],
                         datatype = DataType.VARCHAR,
                         max_length = 256)
        # Image url field
        schema.add_field(field_name = image_node_field[16],
                         datatype = DataType.VARCHAR,
                         max_length = 256)
        # Image mimetype field
        schema.add_field(field_name = image_node_field[17],
                         datatype = DataType.VARCHAR,
                         max_length = 16,
                         nullable = True)
        # Enable text embedding model
        if self._text_captioning_model is not None:
            # Text embedding field
            schema.add_field(field_name = image_node_field[18],
                             datatype = text_datatype,
                             max_length = text_vector_dims)
        return schema

    def _setup_collection_index(self,
                                index_algo :Literal["FLAT","IVF_FLAT","IVF_SQ8","IVF_PQ","HNSW","SCANN"] = "IVF_FLAT",
                                dense_search_metric :Literal["COSINE","L2","IP","HAMMING","JACCARD"] = "COSINE",
                                text_search_metric: Literal["COSINE", "L2", "IP", "HAMMING", "JACCARD"] = "COSINE",
                                params :Optional[dict] = None,
                                text_params :Optional[dict] = None,
                                **kwargs):
        # Define index
        index_params = self.prepare_index_params()
        # Default dense params
        if params is None: params = {"nlist": 128}
        if text_params is None: params = {"nlist": 128}

        image_node_field = list(ImageNode.model_fields.keys())
        # Add id index
        index_params.add_index(field_name = default_keys[0],
                               index_name = "unique_id")
        # Add image index
        index_params.add_index(field_name = default_keys[1],
                               index_name = "image_representation",
                               index_type = index_algo,
                               metric_type = dense_search_metric,
                               params = params)

        # Enable text embedding model
        if self._text_captioning_model is not None:
            # Add text embedding index
            index_params.add_index(field_name = image_node_field[18],
                                   index_name = "text_representation",
                                   index_type = index_algo,
                                   metric_type = text_search_metric,
                                   params = text_params)
        return index_params

    def _create_collection(self,
                           document_type: str,
                           dimension_nums :int,
                           **kwargs) -> dict:
        """
        Create Milvus collection with defined setting
        :param document_type: Type of input document (BaseNode,Document)
        :param dimension_nums: Number of dimension for embedding
        :param enable_sparse: Enable the sparse schema or not
        :return: dict
        """
        # Define schema
        schema = self._setup_collection_schema(document_type = document_type,
                                               image_vector_dims = dimension_nums,
                                               image_datatype = self._dense_datatype)
        # Define index
        index_params = self._setup_collection_index(index_algo = self._index_algo,
                                                    dense_search_metric = self._dense_search_metrics)
        # Collection for LlamaIndex payloads
        self.create_collection(collection_name = self._collection_name,
                               schema = schema,
                               index_params = index_params)

        # Return state
        res = self.get_load_state(
            collection_name = self._collection_name
        )
        return res

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
                                            image_node_field[18]: None})
        # Return BasePoint
        if return_type == "BasePoints":
            return results

        final_output = []
        # Auto mode
        for result in results:
            # Node Result
            node_results = [ImageNode.parse_obj(dict(element).get("entity")) for element in result]
            # NodeWithScore
            node_with_score = [NodeWithScore(node = node_results[i],
                                             score = result[i]["distance"]) for i in range(len(node_results))]
            # Append
            final_output.append(node_with_score)
        return final_output






