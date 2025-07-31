import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
import faiss
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DATA_DIR = "data"


# initialize the project and location.
PROJECT_ID = os.getenv("project_id", None)
LOCATION = os.getenv("project_location", None)


class EmbeddingOperation:
    def __init__(self):
        print("Control into Embedding operation")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.embedding_dimension = 1408
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    def create_embeddings_for_cxr_and_radiologist_report(self,image_path, report_path):
        print("Control inside create embedding for cxr")
        self.image_path = image_path
        self.report_path = report_path
        self.report_text = "There is a small nodule in the right upper lobe, no pleural effusion."

        image = Image.load_from_file(self.image_path)

        # generate embedding for image and text report.
        embeddings = self.model.get_embeddings(
        image=image,
        contextual_text=self.report_text,
        dimension=self.embedding_dimension,
        )

        image_embedding = embeddings.image_embedding
        text_embedding = embeddings.text_embedding
        return image_embedding, text_embedding

    def create_embedding_for_user_cxr_image(self, image_path):
        print("Control inside the create embedding for user cxr")
        image = Image.load_from_file(image_path)
        embed_vector = self.model.get_embeddings(image=image)
        user_cxr_image_embedding = embed_vector.image_embedding
        return user_cxr_image_embedding


class StoreVectorEmbedding:

    def __init__(self):
        print("Contorl inside the store vector embedding")
        self.index_path = "vector_store.faiss"
        self.id_map_path = "metadata.pkl"
        self.embedding_dimension = 1408
        self.top_k = 2

    def store_image_and_text_embedding(self, image_path, report_text, image_embedding, text_embedding):
        print("cpontrol inside the store image and text embedding")
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path):
            index = faiss.read_index(self.index_path)
            with open(self.id_map_path, "rb") as f:
                id_map = pickle.load(f)
        else:
            index = faiss.IndexFlatL2(self.embedding_dimension)
            id_map = {}

        # add embedding to faiss index
        combined_embedding = (self.image_embedding + self.text_embedding) / 2
        index.add(np.expand_dims(combined_embedding, axis=0))
        doc_id = f"doc_{len(id_map)}"
        id_map[len(id_map)] = {"id": doc_id, "report": report_text, "image_path": image_path}

        # save faiss vector embedding
        faiss.write_index(index, self.index_path)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(id_map, f)

        print(f"Added {doc_id} to FAISS index with combined embedding.")

    def perform_similarity_search_from_faiss_vector_store(self, user_embed_query):
        print("control inside the perform similarity search from faiss vector store.")
        index = faiss.read_index(self.index_path)
        with open(self.id_map_path, "rb") as read_metadata_file:
            metadata = pickle.load(read_metadata_file)

        data, id = index.search(np.array([user_embed_query]), self.top_k)

        # return matched result content
        results = []
        for content in id[0]:
            results.append(metadata[content])

        return results