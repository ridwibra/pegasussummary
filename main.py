from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    #"http://localhost:3000",
    "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
def summarize(text):
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary = model.generate(**tokens)
    sum = tokenizer.decode(summary[0][1:-1])
    return sum





@app.get('/')
async def root():
    return "Welcome"


@app.get('/generate')
# async def receive(seed: str = Form(...), length: int = Form(...)):
async def receive(text: str):
    summary = summarize(text)
    return summary
