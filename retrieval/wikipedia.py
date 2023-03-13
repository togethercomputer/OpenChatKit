# This file was adapted from ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index:
#   https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index/blob/main/wikiindexquery.py
#
# The original file was licensed under the Apache 2.0 license.

import os

from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)


class WikipediaIndex:
    def __init__(self):
        path = os.path.join(DIR, '..', 'data', 'wikipedia-3sentence-level-retrieval-index', 'files')
        indexpath = os.path.join(path, 'knn.index')
        wiki_sentence_path = os.path.join(path, 'wikipedia-en-sentences.parquet')

        self._device = 'cuda'
        self._tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
        self._contriever = AutoModel.from_pretrained('facebook/contriever-msmarco').to(self._device)

        self._df_sentences = pd.read_parquet(wiki_sentence_path, engine='fastparquet')

        self._wiki_index = faiss.read_index(indexpath, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


    def search(self, query, k=1, w=5, w_th=0.5):
        inputs = self._tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(self._device)
        outputs = self._contriever(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        
        query_vector = embeddings.cpu().detach().numpy().reshape(1, -1)
        
        distances, indices = self._wiki_index.search(query_vector, k)
        
        texts = []
        for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
            text = self._df_sentences.iloc[indice]['text_snippet']

            try:
                input_texts = [self._df_sentences.iloc[indice]['text_snippet']]
                for j in range(1, w+1):
                    input_texts = [self._df_sentences.iloc[indice-j]['text_snippet']] + input_texts
                for j in range(1, w+1):
                    input_texts = input_texts + [self._df_sentences.iloc[indice+j]['text_snippet']]
                
                inputs = self._tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(self._device)

                outputs = self._contriever(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()

                for j in range(1, w+1):
                    if cos_sim_2d(embeddings[w-j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                        text = self._df_sentences.iloc[indice-j]['text_snippet'] + text
                    else:
                        break

                for j in range(1, w+1):
                    if cos_sim_2d(embeddings[w+j].reshape(1, -1), embeddings[w].reshape(1, -1)) > w_th:
                        text += self._df_sentences.iloc[indice+j]['text_snippet']
                    else:
                        break

            except Exception as e:
                print(e)

            texts.append(text)
        
        return texts
