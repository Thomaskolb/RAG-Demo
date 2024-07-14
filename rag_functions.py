import pandas as pd
import duckdb
import pickle
import sys
import os
import torch
import torch.cuda
import torch.distributed as dist

# Search top k relevant documents for query embedding
# Return top k document passages + scores
def search_relevant_docs(query_emb, duckdb_faiss_path, faiss_index_path, passages_path, k=5):
    # DuckDB session
    con = duckdb.connect(config = {'allow_unsigned_extensions': 'true'})

    # Load FAISS extension
    con.sql(f"LOAD '{duckdb_faiss_path}'")

    # Load Wikipedia index (21GB)
    con.sql(f"CALL faiss_load('index', '{faiss_index_path}');")
    
    # Load Wikipedia passages
    with open(passages_path, 'rb') as f:
        passages = pickle.load(f)
    
    # Create query df and read into duckdb
    query_df = pd.DataFrame({"query": query_emb.tolist()})
    
    # top k search
    topk_df = con.sql(f"SELECT UNNEST(faiss_search('index', {k}, query)) FROM query_df").to_df()
    top_k = [row[0] for _, row in topk_df.iterrows()]
    
    # Extend top k dictionaries with corresponding passages
    top_k_passages = []
    for dict_item in top_k:
        id = int(dict_item['label'])
        top_k_passages.append(passages[id])
        
    return top_k_passages

# Main function that performs RAG
def retrieval_augmented_generation(
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
    ):
    
    # Import ATLAS functions
    sys.path.append(f'{atlas_path}/')
    from src import slurm, util
    from src.model_io import load_or_initialize_atlas_model
    from src.options import get_options
    
    # Set arguments for ATLAS model
    args = f'''--reader_model_type {reader_model} --model_path {model_path} --generation_min_length {generation_min_length} --generation_max_length {generation_max_length} --generation_num_beams {generation_num_beams} --generation_length_penalty {generation_length_penalty} --text_maxlength 512 --per_gpu_batch_size 1 --n_context {context_size} --index_mode "flat" --task "qa"'''
    args = [f'/{atlas_path}/evaluate.py'] + args.replace('\'', '').replace('\"', '').replace('\n', '').split(' ')
    
    # Load options
    sys.argv = args
    options = get_options()
    opt = options.parse()
    torch.manual_seed(seed)
    slurm.init_distributed_mode(opt)
    
    # Load ATLAS
    model, _, _, _, _, opt, _ = load_or_initialize_atlas_model(opt, eval_only=True)
    model.eval()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    
    # Remove question mark from question
    question = question.replace('?', '')

    # Get query encoding
    query = [f'question: {question}: <extra_id_0>']
    query_enc = unwrapped_model.retriever_tokenize(query)
    query_ids_retriever = query_enc["input_ids"].cuda()
    query_mask_retriever = query_enc["attention_mask"].cuda()

    # Get query embedding from model
    unwrapped_model.retriever.eval()
    query_emb = unwrapped_model.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
    
    # Faiss search for relevant documents
    top_k_passages = search_relevant_docs(query_emb, duckdb_faiss_path, faiss_index_path, passages_path, context_size)
    
    # Move reader to device
    unwrapped_model.reader.to(reader_device)

    # Create tokens for reader
    reader_tokens, retriever_tokens = unwrapped_model.tokenize_passages(query, [top_k_passages])
    reader_ids = reader_tokens["input_ids"]
    reader_mask = reader_tokens["attention_mask"].bool()
    n_context_training = min(unwrapped_model.opt.n_context, reader_ids.size(1))

    # Get reader config
    cfg = unwrapped_model.reader.encoder.config
    cfg.bsz = reader_ids.size(0)
    cfg.n_context = n_context_training

    # Reshape reader ids
    reader_ids_gen = reader_ids[:, :n_context_training].contiguous()
    reader_mask_gen = reader_mask[:, :n_context_training].contiguous()
    reader_ids_gen = reader_ids_gen.view(reader_ids.size(0), -1)
    reader_mask_gen = reader_mask_gen.view(reader_mask.size(0), -1)

    # Generate with reader ids
    if unwrapped_model.opt.decoder_prompt_format is not None:
        prefix_str = [unwrapped_model.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
        prefix_allowed_tokens_fn = unwrapped_model.get_prefix_allowed_tokens_fn(prefix_str)

    outputs = unwrapped_model.reader.generate(
        input_ids=reader_ids_gen.to(reader_device),
        attention_mask=reader_mask_gen.to(reader_device),
        num_return_sequences=1,
        max_length=unwrapped_model.opt.generation_max_length,
        min_length=unwrapped_model.opt.generation_min_length,
        num_beams=unwrapped_model.opt.generation_num_beams,
        length_penalty=unwrapped_model.opt.generation_length_penalty
    )

    # Decode
    answer = unwrapped_model.reader_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(unwrapped_model.opt.generation_max_length, unwrapped_model.opt.generation_min_length)
    
    return answer
