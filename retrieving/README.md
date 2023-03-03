<<<<<<< HEAD
# Wiki Index

A replication of [ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index](https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index). Thank you, Christoph!

Morever, instead of comparing and adding the adjacent sentence, I extend to `w` sentences, so the context could be much longer.

0. Install dependencies

torch, transformers, flask, faiss, etc.

1. download data
```shell
=======
# simple-wiki-index

a replication of https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index

1. download data
```
>>>>>>> c7fdd1b24673fd6ca2c0ec0873ba7190d8e38cd1
git lfs install
git clone https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index
```

2. start server
<<<<<<< HEAD
```shell
python wiki-server.py
```
=======
```
python wiki-server.py
```
>>>>>>> c7fdd1b24673fd6ca2c0ec0873ba7190d8e38cd1
