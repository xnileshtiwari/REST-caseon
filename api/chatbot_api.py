from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from test.main_chat import start_chatting
from typing import Optional

app = FastAPI()

class ChatRequest(BaseModel):
    index_name: str
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.index_name or not request.user_input:
            raise HTTPException(status_code=400, detail="Index name and user input are required")
        
        response = start_chatting(request.index_name, request.user_input)
        return {"status": "success", "response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000) 