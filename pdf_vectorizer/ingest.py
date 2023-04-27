# Create vector for text chunks
from dotenv import load_dotenv
import os
import faiss
import pickle
import PyPDF2

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')

# load files
project_folder_path = './pdf_files'
files = os.listdir(project_folder_path)
print(files)
data = []
sources = []
for f in files:
    file_format = files[0][-4:]
    sources.append(f)
    if file_format == '.pdf': # only process pdf files
        pdf_file = open(f'{project_folder_path}/{f}', 'rb')
        print(f)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        f_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            f_text += text
            # all_text += text 
        data.append(f_text)       
        pdf_file.close()

text_splitter = CharacterTextSplitter(chunk_size=300, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    print( len(splits))
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=openai_key), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

