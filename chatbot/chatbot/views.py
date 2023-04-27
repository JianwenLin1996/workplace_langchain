import json
import os
import requests
import faiss
import pickle
from typing import List
from dotenv import load_dotenv

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader

@csrf_exempt
def chat(response):
    load_dotenv()
    if response.method == 'GET':  # for callback
        token = os.getenv('VERIFY_TOKEN')
        try:
            # verify token is set at integration chatbot there, use it as verification if needed
            challenge = response.GET['hub.challenge']
            verify_token = response.GET['hub.verify_token']
            if verify_token == token:
                # must return status 200 to verify
                return HttpResponse(challenge, status=200)
            else:
                return HttpResponse(challenge, status=400)
        except Exception as e:
            return HttpResponse(e, status=200)
    elif response.method == 'POST':  # POST for webhook       
        access_token =  os.getenv('WORKPLACE_ACCESS_TOKEN')  
        openai_key =  os.getenv('OPENAI_KEY')  

        body = json.loads(response.body)
        # print(body)
        messaging = body['entry'][0]['messaging'][0]
        sender_id = messaging['sender']['id']
        message = messaging['message']['text']
        try:
            thread_id = messaging['thread']['id'].replace("t_", "")
            conversation_id = "me" # group chat
        except:
            thread_id = None
            conversation_id = body['entry'][0]['id']

        is_thread = False if thread_id is None else True
        recipient_key = "thread_key" if is_thread is True else "id"
        sender_id = thread_id if is_thread is True else sender_id # sender_id set to None for group chat

        # trigger typing action
        url = "https://graph.facebook.com/v16.0/%s/messages" % (
            conversation_id)
        headers = {
            'Content-Type': 'application/json'
        }
        replying_data = json.dumps({
            "access_token": access_token,
            "messaging_type": "RESPONSE",
            "recipient": {
                recipient_key: sender_id
            },
            "sender_action": "typing_on"
        })
        r = requests.request("POST", url, headers=headers, data=replying_data)
        # print(r.json())

        # load vectorized index and faiss pkl
        base_file = os.getcwd()
        index = faiss.read_index(f"{base_file}/static/docs.index")
        with open(f"{base_file}/static/faiss_store.pkl", "rb") as f:
            store = pickle.load(f)
        store.index = index

        # change prompt to any assistant character you prefer.
        combine_prompt_template = """
        You are an assistant named John from Google. 
        Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
        If it is a greeting, just reply with a polite greeting and stop right there.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.

        SOURCES:

        QUESTION: {question}
        =========
        {summaries}
        =========
        FINAL ANSWER:"""
        c_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])

        # call openai
        # chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(
        #     temperature=0.2, openai_api_key=openai_key), combine_prompt=c_prompt, vectorstore=store)
        # (reduce_k_below_max_tokens=True, max_tokens_limit=1000)
        
        #TODO OpenAI is currently using text-davinci-003, which is 10 times more expensive that gpt-3.5-turbo， however when using ChatOpenAI, which by default use gpt-3.5-turbo, the response is super slow. Try again after fix from LangChain.
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0.25, openai_api_key=openai_key, request_timeout=120), chain_type="stuff", retriever=store.as_retriever())

        c_reply = chain({"question": message})

        # organize reply
        reply_sources = c_reply['sources'] if c_reply['sources'] is List else list(
            c_reply['sources'])
        reply_sources = c_reply['sources'].split(',')

        sources = "None"
        for i in range(len(reply_sources)):
            # print(f'file: {i}')
            tmp_s = reply_sources[i].strip()
            if i != 0:
                sources = sources + ', ' + tmp_s
            else:
                sources = tmp_s
        reply = c_reply['answer'].strip()
        reply = reply + '\n\nReferences: ' + sources
        # print(reply)

        # post to workplace with reply
        data = json.dumps({
            "access_token": access_token,
            "messaging_type": "RESPONSE",
            "recipient": {
                recipient_key: sender_id
            },
            "message": {"text": reply}
        })
        r = requests.request("POST", url, headers=headers, data=data)
        print(url)
        if r.status_code == 200:
            print(r.json())
        else:
            print('Error:', r.status_code)

        return HttpResponse('successful', status=200)


def message(response):
    data = {'key': 'value'}
    return JsonResponse(data)

