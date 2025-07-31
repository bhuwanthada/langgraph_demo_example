import uvicorn
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from cxr.agent_workflow import compile_agent_workflow, set_current_thread_id
import shutil


# Compile the workflow once when the FastAPI app starts
compiled_graph, memory = compile_agent_workflow()

app = FastAPI(
    title="LangGraph Agent for CXR execution",
    description="A REST endpoint demonstrating CXR early stage detection.",
    version="1.0.0"
)


class CXRAPIResponse(BaseModel):
    status: str
    user_id: str
    conversation_history: List[Dict[str, Any]]
    radiologist_report: str


# --- REST Endpoint ---
@app.post("/generate-cxr-radiologist-report", response_model=CXRAPIResponse)
async def process_csr_image_to_generate_radiologist_report(file: UploadFile = File(...)):
    user_id = str(uuid.uuid4())
    set_current_thread_id(user_id)  # Set global in agent_workflow for print statements

    print(f"\n--- API Call for User: {user_id} (ReAct Agent) ---")

    # Prepare the initial input for the LangGraph workflow.
    # The `messages` are key for the LLM's reasoning.

    # First, try to load the existing state to get the current_number and previous messages.
    # This is how the system 'remembers' the current calculation or conversation.
    try:
        loaded_state_values = memory.get_state({"configurable": {"thread_id": user_id}}).values
        # print(f"DEBUG: Loaded state for {user_id}: {loaded_state_values}")
    except Exception:
        loaded_state_values = {}  # No existing state found for this thread_id

    # The messages list must always start with the previous conversation history
    # and then the new human input.
    messages_from_state = loaded_state_values.get("messages", [])[:]

    # Add the current user's input as a new HumanMessage for the agent to process.
    # here we need to pass the image path to calling method so that.
    file_location = f"uploaded_files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    human_message_content = (f"You have been provided the Chest Xray image path: {file_location}. "
                             f"You have to process it for generate radiologist report.")
    messages_from_state.append(HumanMessage(content=human_message_content))

    # the state loaded from the checkpointer.
    initial_graph_input = {
        "messages": messages_from_state,
        "error_message": loaded_state_values.get("error_message")  # Carry over error if any
    }

    try:
        # Invoke the graph with the current user's thread_id
        final_state = compiled_graph.invoke(initial_graph_input, {"configurable": {"thread_id": user_id}})

        print(f"--- User: {user_id} - ReAct Agent Processed Successfully ---")
        return CXRAPIResponse(
            status="success",
            user_id=user_id,
            radiologist_report=final_state.get("radiologist_report_content", None)
        )

    except Exception as e:
        print(f"\n!!! Exception for User: {user_id} (ReAct Agent) during API call !!!")
        print(f"Error: {e}")
        # The checkpointer will have saved the state before the failure.
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "user_id": user_id,
                "error": str(e),
                "message": "An internal error occurred during processing by the agent. "
                           "The state has been saved for recovery. Please try again."
            }
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)