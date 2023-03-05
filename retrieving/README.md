# Retrieval-Enhanced Chatbot

This is a demonstration of how to enhance a chatbot using Wikipedia. We'll be using [ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index](https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index). for this demo. Thank Christoph for providing this resource!

In this demo, we'll be extending the approach of comparing and adding the adjacent `w` sentences to the matched sentence if their cosine similarity is larger than `w_th`. By doing so, we can provide the chatbot with a longer context, which may improve its performance.

## Start index server

To get started, we need to install some dependencies and download the Wikipedia index:

0. Install dependencies

Install the necessary dependencies, including `torch`, `transformers`, `flask`, `faiss`, and `fastparquet`.

1. Clone wiki index

```shell
git lfs install
git clone https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index
```

2. Start the retrieval server

```shell
python wiki-server.py
```

The server will listen on port 7003.

3. Test the server
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

This should print the most relevant sentences about Zurich from Wikipedia. By increasing w and decreasing w_th, we can retrieve a longer context.


## Integrate into the chatbot

Now that we have the retrieval server set up, we can integrate it into our chatbot. Here's an example:

```python
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

def retrieve(query, w=5, w_th=0.7):
    endpoint = 'http://127.0.0.1:7003/search'
    res = requests.post(endpoint, json={
        'query': query,
        'k': 1,
        'w': w,
        'w_th': w_th,
    })
    return res.json()['texts']

# Initialize the chatbot model
model_name_or_path = '...'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Define the user's query
input_query = "Where is Zurich?"

# Retrieve relevant information from Wikipedia
context = retrieve(input_query)[0]

# Generate a response using the chatbot model
prompt = f"<human>: {context}\n\n{input_query}\n<bot>:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.6, top_k=40)
output = tokenizer.batch_decode(outputs)[0]

print(output)
```

In this example, we define a `retrieve()` function to retrieve relevant information from Wikipedia using the retrieval server we set up earlier. We then initialize the chatbot model and define the user's query.

Next, we retrieve relevant information from Wikipedia using the `retrieve()` function and generate a response using the chatbot model. We concatenate the retrieved information and the user's query into a prompt, encode it using the tokenizer, and generate a response using the the chatbot model.
The response should indicate the location of Zurich city.