from flask import Flask, render_template, request
import rag_functions as rag

# Host
HOST_ADDRESS = '127.0.0.1'
PORT = 4444

# Default values
ATLAS_PATH = 'atlas'
MODEL_PATH = '/home/tkolb/data/models/atlas/base'
FAISS_INDEX_PATH = '/home/tkolb/data/faiss_index.index'
PASSAGES_PATH = '/home/tkolb/data/wiki_passages.pkl'
DUCKDB_FAISS_EXT = 'faiss/build/release/repository/v1.0.0/linux_amd64/faiss.duckdb_extension'
READER_MODEL = 'google/t5-base-lm-adapt'
READER_DEVICE = 'cuda:2'
GENERATION_MIN_LENGTH = 1
GENERATION_MAX_LENGTH = 16
GENERATION_NUM_BEAMS = 1
GENERATION_LENGTH_PENALTY = 0
CONTEXT_SIZE = 40
SEED = 42
DEFAULT_QUESTION = 'What is the first movie ever made?'
DEFAULT_ANSWER = ''

# Flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = DEFAULT_ANSWER

    # Paths
    atlas_path = ATLAS_PATH
    model_path = MODEL_PATH
    faiss_index_path = FAISS_INDEX_PATH
    passages_path = PASSAGES_PATH
    duckdb_faiss_path = DUCKDB_FAISS_EXT
    
    # Configuration
    reader_model = READER_MODEL
    reader_device = READER_DEVICE
    generation_min_length = GENERATION_MIN_LENGTH
    generation_max_length = GENERATION_MAX_LENGTH
    generation_num_beams = GENERATION_NUM_BEAMS
    generation_length_penalty = GENERATION_LENGTH_PENALTY
    context_size = CONTEXT_SIZE
    seed = SEED
    
    # Query
    question = DEFAULT_QUESTION
    
    if request.method == 'POST':
        atlas_path = request.form.get('atlas_path', ATLAS_PATH)
        model_path = request.form.get('model_path', MODEL_PATH)
        faiss_index_path = request.form.get('faiss_index_path', FAISS_INDEX_PATH)
        passages_path = request.form.get('passages_path', PASSAGES_PATH)
        duckdb_faiss_path = request.form.get('duckdb_faiss_path', DUCKDB_FAISS_EXT)
        
        reader_model = request.form.get('reader_model', READER_MODEL)
        reader_device = request.form.get('reader_device', READER_DEVICE)
        generation_min_length = int(request.form.get('generation_min_length', GENERATION_MIN_LENGTH))
        generation_max_length = int(request.form.get('generation_max_length', GENERATION_MAX_LENGTH))
        generation_num_beams = int(request.form.get('generation_num_beams', GENERATION_NUM_BEAMS))
        generation_length_penalty = int(request.form.get('generation_length_penalty', GENERATION_LENGTH_PENALTY))
        context_size = int(request.form.get('context_size', CONTEXT_SIZE))
        seed = int(request.form.get('seed', SEED))
        
        question = request.form.get('question', DEFAULT_QUESTION)
        
        result = rag.retrieval_augmented_generation(
            atlas_path, 
            model_path, 
            faiss_index_path, 
            passages_path,
            duckdb_faiss_path,
            reader_model,
            reader_device,
            generation_min_length,
            generation_max_length,
            generation_num_beams,
            generation_length_penalty,
            context_size,
            seed,
            question
        )
    
    return render_template(
        'index.html', 
        result=result, 
        atlas_path=atlas_path, 
        model_path=model_path, 
        faiss_index_path=faiss_index_path, 
        passages_path=passages_path, 
        duckdb_faiss_path=duckdb_faiss_path, 
        reader_model=reader_model,
        reader_device=reader_device,
        generation_min_length=generation_min_length,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams,
        generation_length_penalty=generation_length_penalty,
        context_size=context_size,
        seed=seed,
        question=question
    )

if __name__ == '__main__':
    app.run(debug=True, host=HOST_ADDRESS, port=PORT)
