# Description
With this repository we created a Retrieval Augmented Generation demo using the duckdb faiss extension for the retrieval part, and ATLAS for creating the embeddings and generating the final output.

# How to install
1. Clone atlas: ```git clone https://github.com/facebookresearch/atlas.git``
2. Create a python environment
3. Install torch: ``pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html``
4. Install FAISS with GPU support: ``pip install faiss-gpu==1.7.2``
5. Install requirements: ``pip install -r requirements.txt``

# Models and indices
1. Download one of the pre-trained atlas models (available on ATLAS page: https://github.com/facebookresearch/atlas?tab=readme-ov-file#installation). For example base:
```
python3 atlas/preprocessing/download_model.py --model models/atlas/base --output_directory {MODEL_DIR}
```
2. Download the indices shard files from ATLAS. For example base:
```
python atlas/preprocessing/download_index.py --index indices/atlas/wiki/base --output_directory {INDEX_DIR} 
```
3. Create complete .index files + wikipedia passages with create_faiss_index.ipynb (to make sure the index is small enough to fit into memory)

After installing the correct packages + models the RAG demo web UI should be functional:
```
python3 main.py
```
