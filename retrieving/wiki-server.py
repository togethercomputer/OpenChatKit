# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
from huggingface_hub import snapshot_download
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import os

import faiss
import numpy as np
import pandas as pd

import time
import os
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(x, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)


# XXX provide the path of your trained model
model_name_or_path = '/home/you/chat_model_path'

app = Flask(__name__)

path = snapshot_download('ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index', repo_type='dataset')
indexpath = os.path.join( path, 'knn.index')
wiki_sentence_path = os.path.join( path, 'wikipedia-en-sentences.parquet')
print("WIKI", path, indexpath, wiki_sentence_path)

print("loading model....")
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
contriever = AutoModel.from_pretrained('facebook/contriever-msmarco')
device = 'cuda'
contriever = contriever.to(device)

print("loading wiki data...")
df_sentences = pd.read_parquet(wiki_sentence_path, engine='fastparquet')

print("loading faiss index...")
wiki_index = faiss.read_index(indexpath, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

print("loading chat model....")
# Initialize the chatbot model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().cuda()
tokenizer_chat = AutoTokenizer.from_pretrained(model_name_or_path)

@app.route('/search/<query>',methods=['GET'])
def naive_search(query):
    
    # k = request.args.get('k', default=1, type = int)
    # w = request.args.get('w', default=1, type = int)
    # w_th = request.args.get('w_th', default=0.7, type = float)
    k = 1
    w = 5
    w_th = 0.5
    
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = contriever(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    
    query_vector = embeddings.cpu().detach().numpy().reshape(1, -1)
    
    distances, indices = wiki_index.search(query_vector, k)
    
    texts = []
    for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
        text = df_sentences.iloc[indice]['text_snippet']
        # print(text)

        try:
            
            input_texts = [df_sentences.iloc[indice]['text_snippet']]
            for j in range(1, w+1):
                input_texts = [df_sentences.iloc[indice-j]['text_snippet']] + input_texts
            for j in range(1, w+1):
                input_texts = input_texts + [df_sentences.iloc[indice+j]['text_snippet']]
            
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(device)

            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()

            for j in range(1, w+1):
                if cos_sim_2d(embeddings[w-j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                    text = df_sentences.iloc[indice-j]['text_snippet'] + text
                else:
                    break

            for j in range(1, w+1):
                if cos_sim_2d(embeddings[w+j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                    text += df_sentences.iloc[indice+j]['text_snippet']
                else:
                    break

        except Exception as e:
            print(e)

        texts.append(text)
    
    print(texts)
    
    return jsonify({
        'texts': texts,
    })

def retrieve(query, w=5, w_th=0.7):
    endpoint = 'http://127.0.0.1:7003/search'
    res = requests.post(endpoint, json={
        'query': query,
        'k': 1,
        'w': w,
        'w_th': w_th,
    })
    return res.json()['texts']


# Generate a response using the chatbot model
#prompt = f"{context}\n\n<human>: {input_query}\n<bot>:"
#inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
#outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.6, top_k=40)
#output = tokenizer.batch_decode(outputs)[0]
#print(output)


@app.route('/inference',methods=['POST'])
def inference():
    data = request.get_json(force=True)
    input_prompt = data['prompt']

    start = datetime.datetime.now()
    context = retrieve(input_prompt)[0]
    end = datetime.datetime.now()
    tot = end - start
    tot = tot / datetime.timedelta(milliseconds=1)
    print("Time for http localhost FAISS ", tot )

    # Generate a response using the chatbot model
    start = datetime.datetime.now()
    prompt = f"{context}\n\n<human>: {input_prompt}\n<bot>:"
    inputs = tokenizer_chat(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.6, top_k=40)
    end = datetime.datetime.now()
    tot = end - start
    tot = tot / datetime.timedelta(milliseconds=1)
    print("Time for Model ", tot )

    start = datetime.datetime.now()
    output = tokenizer_chat.batch_decode(outputs)[0]
    end = datetime.datetime.now()
    tot = end - start
    tot = tot / datetime.timedelta(milliseconds=1)
    print("Time for Batch Decode ", tot )


    return jsonify({
        'prompt': input_prompt,
        'output': output
    })
    

@app.route('/search',methods=['POST'])
def search():
   
    start = datetime.datetime.now()
    data = request.get_json(force=True)
    k = data.get('k', 1)
    w = data.get('w', 5)
    w_th = data.get('w_th', 0.5)
    
    query = data['query']
    
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = contriever(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    
    query_vector = embeddings.cpu().detach().numpy().reshape(1, -1)
    
    distances, indices = wiki_index.search(query_vector, k)
    
    texts = []
    for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
        text = df_sentences.iloc[indice]['text_snippet']
        # print(text)

        try:
            
            input_texts = [df_sentences.iloc[indice]['text_snippet']]
            for j in range(1, w+1):
                input_texts = [df_sentences.iloc[indice-j]['text_snippet']] + input_texts
            for j in range(1, w+1):
                input_texts = input_texts + [df_sentences.iloc[indice+j]['text_snippet']]
            
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(device)

            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()

            for j in range(1, w+1):
                if cos_sim_2d(embeddings[w-j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                    text = df_sentences.iloc[indice-j]['text_snippet'] + text
                else:
                    break

            for j in range(1, w+1):
                if cos_sim_2d(embeddings[w+j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                    text += df_sentences.iloc[indice+j]['text_snippet']
                else:
                    break

        except Exception as e:
            print(e)

        texts.append(text)
    
    print(texts)

    end = datetime.datetime.now()
    tot = end - start
    tot = tot / datetime.timedelta(milliseconds=1)
    print("Time for FAISS", tot ) 
    return jsonify({
        'texts': texts,
    })
    

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=7003, debug=False)
