# This file is just to test LangChain QA Chain
from dotenv import load_dotenv
import os
import faiss
import pickle
import argparse
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# load vectorized index and faiss pkl
index = faiss.read_index("docs.index")
with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
store.index = index

chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, openai_api_key=openai_key), vectorstore=store)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
