from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import chat_router

app = FastAPI(title="Know Your Rights Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://know-your-rights-tau.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
def home():
    return {"message": "Know Your Rights chatbot API is running"}
