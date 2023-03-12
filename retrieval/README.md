# Retrieval-Enhanced Chatbot

This is a demonstration of how to enhance a chatbot using Wikipedia. We'll be using [ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index](https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index). for this demo. Thank Christoph for providing this resource!

In this demo, we'll be extending the approach of comparing and adding the adjacent `w` sentences to the matched sentence if their cosine similarity is larger than `w_th`. By doing so, we can provide the chatbot with a longer context, which may improve its performance.

This demo combines both the above index and the chat model into one system

## Start the combined  server

To get started, we need to install some dependencies and download the Wikipedia index:

0. Install dependencies

Install the necessary dependencies, including `torch`, `transformers`, `flask`, `faiss`, and `fastparquet`.

1. Open up wiki-server.py and set model_name_or_path to point to the path that contains the chat
model


2. Start the retrieval server

```shell
python wiki-server.py
```

The server will listen on port 7003.  It will download the data sets from ChristophSchuhman.  This
may take a few minutes.

3. Test the full retrieval enhanced chatbot

We now demonstrate both the wiki index and the GPT-NeoX-fine-tuned model.

```curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:7003/inference -d '{ "prompt" : "where is zurich located?" }'```

Internally we first query the wiki index and generate a response using the provided model.  To do
this, We concatenate the retrieved information and the users' query into a prompt, 
encode it with a tokenizer, and generate a response using the chatbot model.

The response should indicate the location of Zurich city.


4. To test just the retrieval functionality of the system you can can do the following.  Curl works
as well.

```python
import requests

endpoint = 'http://127.0.0.1:7003/search'
res = requests.post(endpoint, json={
    'query': 'Where is Zurich?',
    'k': 1,
    'w': 5,
    'w_th': 0.7,
})
print(res.json())
```

This should print the most relevant sentences about Zurich from Wikipedia. By increasing w and 
decreasing w_th, we can retrieve a longer context.


