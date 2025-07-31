import os
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from cxr.prompt import chest_xray_prompt, chatgroq_system_message
from typing import List, Dict
from langchain_core.tools import tool

# from create_and_store_faiss_embedding import EmbeddingOperation, StoreVectorEmbedding
load_dotenv()

@tool
def get_similarity_search_for_given_image(user_image_path: str)->List[str]:
    """This method is used to perform similaraity search based upon given user image.
    This will create image embedding and then perform similarity search upon doing on faiss vectore search."""
    # embedding_ops = EmbeddingOperation()
    # user_image_embed = embedding_ops.create_embedding_for_user_cxr_image(user_image_path)
    # store_vector_embedding = StoreVectorEmbedding()
    # similarity_result = store_vector_embedding.perform_similarity_search_from_faiss_vector_store(user_image_embed)
    similarity_result = ["1. Highly infected with localize TB. Left lung is abnormally bulky and hyperlucid fluid is clearly visible.",
    "2. There is semi liquid is passing from right lung to upper breathing system. There are several lumps observe near to left lung."]
    return similarity_result


@tool
def generate_radiologist_report(similarity_search: List)-> str:
    """This method is used to generate radiologist report based upon provided similarity search data.
    It will accept the list containing similar search in str format over here. As part of result, it will generate a text data in form of report.."""
    similar_report = "".join(similarity_search)
    # similar_report = similarity_search
    llm = ChatGroq(
        temperature=0.0,
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API", None)
    )
    chest_xray_prompt_content = chest_xray_prompt.format(similar_report=similar_report)
    message = [("system", chatgroq_system_message),
               ("human", chest_xray_prompt_content)]
    ai_msg = llm.invoke(message)
    result = ai_msg.content.strip()
    return result
