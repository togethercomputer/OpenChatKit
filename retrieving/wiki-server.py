# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

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


app = Flask(__name__)



indexpath = "./wikipedia-3sentence-level-retrieval-index/knn.index"
wiki_sentence_path = "./wikipedia-3sentence-level-retrieval-index/wikipedia-en-sentences.parquet"

print("loading model....")
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
contriever = AutoModel.from_pretrained('facebook/contriever-msmarco')
device = 'cuda'
contriever = contriever.to(device)

print("loading wiki data...")
df_sentences = pd.read_parquet(wiki_sentence_path, engine='fastparquet')

print("loading faiss index...")
wiki_index = faiss.read_index(indexpath, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

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

@app.route('/search',methods=['POST'])
def search():
    
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
    
    return jsonify({
        'texts': texts,
    })
    

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=7003, debug=False)
