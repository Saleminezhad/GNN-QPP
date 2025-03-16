from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import faiss
import torch
import gzip
import os
import pytrec_eval
import pandas as pd
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import tempfile

############################################################# Encode #################################################################

def encode_chunk(model, tokenizer, chunk, output_path, file_index):
    """
    Encodes a chunk of passages and saves the embeddings to disk.
    Ensures sequences do not exceed the max token length.
    """
    print(f"Encoding chunk {file_index} with {len(chunk)} passages...")

    # Get max sequence length dynamically
    max_length = tokenizer.model_max_length

    # Tokenize and truncate passages
    truncated_chunk = [passage[:max_length] for passage in chunk]

    try:
        corpus_embeddings = model.encode(truncated_chunk, convert_to_tensor=True, show_progress_bar=False, batch_size=128)
        save_path = f"{output_path}/corpus_tensor_{file_index}.pt"
        torch.save(corpus_embeddings, save_path)
        print(f"Saved chunk {file_index} to {save_path}")
    except Exception as e:
        print(f"Error encoding chunk {file_index}: {e}")

def encode_parallel(model_name, input_file, index_path, num_threads=4, chunk_size=1000000):
    """
    Multithreaded encoding function to process passages in parallel.
    Ensures all encoding jobs complete before moving to FAISS indexing.
    """
    output_path = index_path.rsplit("/", 1)[0]  # Get parent directory
    
    if not os.path.exists(output_path):
        print(f"Creating directory '{output_path}'")
        os.makedirs(output_path)

    model = SentenceTransformer(f"sentence-transformers/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")  # Dynamically get tokenizer

    # Retrieve the correct max sequence length
    model.max_seq_length = tokenizer.model_max_length
    print(f"Model {model_name} max sequence length: {model.max_seq_length}")

    embedding_dimension_size = model.get_sentence_embedding_dimension()
    print(f"Embedding Dimension Size: {embedding_dimension_size}")

    # Load passages
    passages = []
    with open(input_file, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            passages.append(passage)

    print(f"Total Passages: {len(passages)}")

    # Split passages into fixed-size chunks
    total_passages = len(passages)
    chunks = [passages[i:i + chunk_size] for i in range(0, total_passages, chunk_size)]

    # Run encoding in parallel using multiple threads and wait for completion
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(encode_chunk, model, tokenizer, chunk, output_path, i + 1): i for i, chunk in enumerate(chunks)}
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in encoding chunk {futures[future] + 1}: {e}")

    print("All encoding jobs completed.")

    # Build FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dimension_size)

    # Load embeddings and add to FAISS
    for i in range(len(chunks)):
        emb_path = f"{output_path}/corpus_tensor_{i + 1}.pt"
        if not os.path.exists(emb_path):  
            print(f"Warning: {emb_path} not found. Skipping...")
            continue
        all_corpus = torch.load(emb_path, map_location=torch.device('cuda')).detach().cpu().numpy()
        index.add(all_corpus)
        print(f"Added embeddings from {emb_path} to FAISS index.")

    print(f"FAISS Index Total Size: {index.ntotal}")
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    print("FAISS index saved successfully.")


############################################################# retrieval #################################################################


def retrieve(index_path, model_name, queries_file_path, output_run_path, top_k=1000):
    
    if not os.path.exists(output_run_path):
        print(f"Directory '{output_run_path}' does not exist. Creating it now...")
        os.makedirs(output_run_path)
        
    index = faiss.read_index(index_path)
    
    print('load index done')

    model = SentenceTransformer(f"sentence-transformers/{model_name}")
    embedding_dimension_size = model.get_sentence_embedding_dimension()
    model.max_seq_length=embedding_dimension_size
    
    out=open(f'{output_run_path}/run_file','w')
    qids=[]
    queries=[]
    with open(queries_file_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)
    xq = model.encode(queries)
    D, I = index.search(xq, top_k)  # search
    rank=1
    for q_id in range(len(I)):
        for rank in range(1,top_k+1):
            doc_id = I[q_id][rank-1]  # Retrieved document ID
            l2_distance = D[q_id][rank-1]  # L2 distance
            
            # Convert L2 distance to a similarity score
            similarity = 1 / (1 + l2_distance)  # Higher is better
            
            # Write in TSV format: qid, docid, rank, relevance_score
            out.write(f"{qids[q_id]}\t{doc_id}\t{rank}\t{similarity:.6f}\n")
            

    out.close()


    

############################################## parallel retrieval ##############################################

def retrieve_query_chunk(chunk_data):
    """
    Function to be executed in parallel.
    Each process encodes its chunk of queries, retrieves results from FAISS, and writes output.
    """
    print("Loading FAISS index...")
    print(f"Process {os.getpid()} started.")

    queries_chunk, qids_chunk, index_path, model_name, output_path, top_k, process_id = chunk_data

    # Move FAISS index to GPU
    # res = faiss.StandardGpuResources()
    index = faiss.read_index(index_path)
    # index = faiss.index_cpu_to_gpu(res, 0, index)

    # Load SentenceTransformer model and move to GPU
    model = SentenceTransformer(f"sentence-transformers/{model_name}")
    model.to("cuda")

    # Encode queries on GPU with reduced batch size
    with torch.no_grad():
        xq = model.encode(queries_chunk, convert_to_tensor=True, device="cuda", batch_size=16)

    xq = xq.cpu().numpy()
    D, I = index.search(xq, top_k)  # FAISS search

    # Write results to a temporary file
    temp_file = f"{output_path}/temp_run_file_{process_id}.tsv"
    with open(temp_file, "w") as out:
        for q_idx, qid in enumerate(qids_chunk):
            for rank in range(1, top_k + 1):
                doc_id = I[q_idx][rank - 1]  # Retrieved document ID
                l2_distance = D[q_idx][rank - 1]  # L2 distance
                
                # Convert L2 distance to a similarity score
                similarity = 1 / (1 + l2_distance)  # Higher is better
                
                # Write in TSV format: qid, docid, rank, similarity_score
                out.write(f"{qid}\t{doc_id}\t{rank}\t{similarity:.6f}\n")

    print(f"Process {os.getpid()} finished. Output saved to {temp_file}.")
    return temp_file  # Return temp file path

def retrieve_parallel(index_path, model_name, queries_file_path, output_run_file, top_k, chunk_size, num_workers=4):
    # retrieve_parallel(args.index_path, args.model_name, args.queries_file, args.output_run_file, args.top_k, args.chunk_size_retrieve, args.num_threads)

    """
    retrieve_parallel(args.index_path, args.model_name, args.queries_file, args.output_run_file, args.top_k, args.chunk_size_retrieve, args.num_threads)

    Parallelized retrieval function that splits queries into chunks and processes them in parallel.
    """
    output_run_path = output_run_file.rsplit("/", 1)[0]  # Get parent directory
    if not os.path.exists(output_run_path):
        print(f"Directory '{output_run_path}' does not exist. Creating it now...")
        os.makedirs(output_run_path)

    print("Loading FAISS index...")

    # Load queries
    qids = []
    queries = []
    with open(queries_file_path, "r", encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)
    
    print(f"Total Queries: {len(queries)}")

    # Split queries into chunks for parallel processing
    # num_workers = min(args.num_workers, multiprocessing.cpu_count(), len(queries))
    # chunk_size = max(1, len(queries) // num_workers)  # Prevent zero-sized chunks

    num_workers = min(num_workers, multiprocessing.cpu_count())  # Use available CPU cores
    # chunk_size = len(queries) // num_workers
    query_chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]
    qid_chunks = [qids[i:i + chunk_size] for i in range(0, len(qids), chunk_size)]
    
    print(f"Loaded {len(queries)} queries for retrieval.")
    print(f"number {len(query_chunks)} for retrieval.")
    print(f"Splitting queries into {num_workers} chunks...")
    print(f"Chunk size: {chunk_size}")
    # Prepare arguments for multiprocessing
    args = [
        (query_chunks[i], qid_chunks[i], index_path, model_name, output_run_path, top_k, i)
        for i in range(len(query_chunks))
    ]
    # args_p = [
    #     (index_path, model_name, output_run_path, top_k, i)
    #     for i in range(len(query_chunks))
    # ]
    print("Starting parallel retrieval...")
    print(f"Number of workers: {num_workers}")
    # print(args_p)
    # Start multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        temp_files = pool.map(retrieve_query_chunk, args)

    print("Merging results...")

    # Merge all temporary result files
    with open(output_run_file, "w") as final_out:
        for temp_file in temp_files:
            with open(temp_file, "r") as temp_in:
                final_out.write(temp_in.read())
            os.remove(temp_file)  # Delete temporary file after merging

    print(f"Retrieval completed. Output saved to {output_run_file}")
    
    
############################################################# Evaluation #################################################################
# def evaluate_run(run_file, qrels_file, output_file):
    
#     # evaluate_run(args.output_run_file, args.qrels_file, args.output_eval_file)
#     output_file_path = output_file.rsplit("/", 1)[0]  # Get parent directory
#     if not os.path.exists(output_file_path):
#         print(f"Directory '{output_file_path}' does not exist. Creating it now...")
#         os.makedirs(output_file_path)
        
#     # Load ground truth (qrels)
#     qrels = {}
#     with open(qrels_file, "r") as f:
#         for line in f:
#             qid, _, docid, relevance = line.strip().split()
#             if qid not in qrels:
#                 qrels[qid] = {}
#             qrels[qid][docid] = int(relevance)

#     # Load retrieval results
#     run = {}
#     with open(run_file, "r") as f:
#         for line in f:
#             qid, docid, rank,_ = line.strip().split()
#             if qid not in run:
#                 run[qid] = {}
#             run[qid][docid] = 1 / int(rank)  # Higher rank = lower score

#     # Define evaluation metrics
#     metrics = {'map', 'ndcg', 'recip_rank'}
#     evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
#     results = evaluator.evaluate(run)

#     # here we want to calculate the previous metrics for the @10
#     rank_cut = 10
#     metrics = {'map', 'ndcg', 'recip_rank'}
    
#     run = {}
#     with open(run_file, "r") as f:
#         for line in f:
#             qid, docid, rank,_ = line.strip().split()
#             if rank_cut < int(rank):
#                 continue
#             if qid not in run:
#                 run[qid] = {}
#             run[qid][docid] = 1 / int(rank)  # Higher rank = lower score

#     # Define evaluation metrics
#     metrics = {'map', 'ndcg', 'recip_rank'}
#     evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
#     results_10 = evaluator.evaluate(run)


#     # Convert results into a DataFrame for per-query analysis
#     per_query_scores = pd.DataFrame.from_dict(results, orient='index')
#     per_query_scores_10 = pd.DataFrame.from_dict(results_10, orient='index')
#     # Display the per-query metrics
#     # print(per_query_scores)

#     # Save to a file if needed
#     per_query_scores.to_csv(output_file, sep="\t", index=True)
#     per_query_scores_10.to_csv(output_file +"_10", sep="\t", index=True)

#     return per_query_scores  # Return in case you need it programmatically





# def evaluate_run_parallel(run_file, qrels_file, output_file, num_threads=4):
#     """
#     Parallelized evaluation function using multiprocessing.
#     """
#     output_file_path = output_file.rsplit("/", 1)[0]  # Get parent directory

#     if not os.path.exists(output_file_path):
#         print(f"Directory '{output_file_path}' does not exist. Creating it now...")
#         os.makedirs(output_file_path)
    
#     # Load ground truth (qrels)
#     qrels = {}
#     with open(qrels_file, "r") as f:
#         for line in f:
#             qid, _, docid, relevance = line.strip().split()
#             if qid not in qrels:
#                 qrels[qid] = {}
#             qrels[qid][docid] = int(relevance)

#     # Load retrieval results
#     run = {}
#     with open(run_file, "r") as f:
#         for line in f:
#             qid, docid, rank, _ = line.strip().split()
#             if rank_cut < int(rank):
#                 continue
#             if qid not in run:
#                 run[qid] = {}
#             run[qid][docid] = 1 / int(rank)  # Higher rank = lower score
    
#     num_threads = min(num_threads, multiprocessing.cpu_count())
#     chunk_size = max(1, len(qrels.keys()) // num_threads)

#     # Split queries into chunks
#     query_chunks = []
#     for i in range(0, len(qrels.keys()), chunk_size):    
#         qrels_temp = {k: qrels[k] for k in list(qrels.keys())[i:i+chunk_size]}
#         run_temp = {k: run[k] for k in list(qrels.keys())[i:i+chunk_size] if k in run}
#         query_chunks.append((qrels_temp, run_temp))

#     print(f"Loaded {len(run)} queries for evaluation.")

#     # Parallel Processing of Queries
#     with multiprocessing.Pool(processes=num_threads) as pool:
#         results_list = pool.starmap(evaluate_query_chunk, query_chunks)

#     # Merge all results
#     merged_results = {}
#     for result in results_list:
#         for qid, scores in result.items():
#             merged_results[qid] = scores

#     # Convert results to a DataFrame
#     per_query_scores = pd.DataFrame.from_dict(merged_results, orient='index')

#     # Save to a file
#     per_query_scores.to_csv(output_file, sep="\t", index=True)
#     print(f"Evaluation completed. Results saved to {output_file}")

#     return per_query_scores  # Return in case you need it programmatically

def evaluate_query_chunk(qrels_file, chunk_file, chunk_id, output_dir):
    """
    Evaluates a chunk of queries by reading the saved chunk file.
    Uses minimal memory by only keeping relevant queries in RAM.
    """
    metrics = {'map', 'ndcg', 'recip_rank'}

    # Load qrels (ground truth relevance) from disk
    qrels = {}
    with open(qrels_file, "r") as f:
        for line in f:
            qid, _, docid, relevance = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(relevance)

    # Load query chunk from the saved file
    with open(chunk_file, "r") as f:
        run_chunk = json.load(f)  # Read JSON

    # Evaluate the queries
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run_chunk)

    # Save results immediately to disk
    temp_path = os.path.join(output_dir, f"eval_chunk_{chunk_id}.tsv")
    per_query_scores = pd.DataFrame.from_dict(results, orient='index')
    per_query_scores.to_csv(temp_path, sep="\t", index=True)

    return temp_path  # Return only the file path

def evaluate_run_chunked(run_file, qrels_file, output_file, num_threads=4, chunk_size=20000, rank_cut=1000):
    """
    Evaluates retrieval results by processing the run_file in smaller chunks
    and saving intermediate results to disk instead of memory.
    """
    output_file_path = os.path.dirname(output_file)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    print(f"Output will be saved to {output_file}")

    temp_dir = tempfile.mkdtemp()  # Temporary directory for storing chunks and results

    # Save query chunks to disk instead of keeping them in memory
    chunk_files = []
    current_chunk = {}
    chunk_id = 0  # Track chunk number
    print(f"Loading run file in chunks of {chunk_size} queries...")
    
    with open(run_file, "r") as f:
        for line in f:
            qid, docid, rank, _ = line.strip().split()
            if int(rank) > rank_cut:
                continue
            if qid not in current_chunk:
                current_chunk[qid] = {}
            current_chunk[qid][docid] = 1 / int(rank)  # Higher rank = lower score

            # Save chunk to disk if it reaches `chunk_size`
            if len(current_chunk) >= chunk_size:
                chunk_file = os.path.join(temp_dir, f"query_chunk_{chunk_id}.json")
                with open(chunk_file, "w") as cf:
                    json.dump(current_chunk, cf)  # Save as JSON
                chunk_files.append(chunk_file)

                current_chunk = {}  # Reset memory
                chunk_id += 1

        # Save any remaining queries
        if current_chunk:
            chunk_file = os.path.join(temp_dir, f"query_chunk_{chunk_id}.json")
            with open(chunk_file, "w") as cf:
                json.dump(current_chunk, cf)  # Save last chunk
            chunk_files.append(chunk_file)

    print(f"Saved {len(chunk_files)} query chunks to {temp_dir}.")

    # Parallel Processing of Queries (each thread evaluates a chunk)
    with multiprocessing.Pool(processes=num_threads) as pool:
        temp_files = pool.starmap(evaluate_query_chunk, [(qrels_file, chunk_file, i, temp_dir) for i, chunk_file in enumerate(chunk_files)])

    # Merge all results from temp files (low memory usage)
    merged_results = pd.concat([pd.read_csv(temp_file, sep="\t", index_col=0) for temp_file in temp_files])

    # Save final results
    merged_results.to_csv(output_file + "_" + str(rank_cut), sep="\t", index=True)
    print(f"Evaluation completed. Results saved to {output_file}")

    # Cleanup temporary files
    for temp_file in temp_files + chunk_files:
        os.remove(temp_file)
    os.rmdir(temp_dir)  # Remove temporary directory

    return merged_results  # Return in case you need it programmatically

############## temp ##############

def main():
    
    parser = argparse.ArgumentParser()
    
    # general input arguments
    parser.add_argument("--encode", type=bool, default=False, help="Encode the collection")
    parser.add_argument("--retrieve", type=bool, default=False, help="Retrieve the queries")
    parser.add_argument("--evaluate", type=bool, default=False, help="Evaluate the retrieval results")
    parser.add_argument("--model_name", type=str, default='msmarco-MiniLM-L6-cos-v5', help="Model name to use for encoding")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel encoding")

    # parser.add_argument("--collection", type=str, default='v1', help="V1 or V2:Collection name to use for encoding")
    # parser.add_argument("--index_type", type=str, default='doc', help="Index type to use for encoding and naming the index")

    # encode arguments
    parser.add_argument("--chunk_size_encode", type=int, default=1000000, help="chunk size for parallel encoding")
    parser.add_argument("--input_encode_file", type=str, default='dataset/v1/collection.tsv', help="Input file to use for encoding")
    parser.add_argument("--index_path", type=str, default='faiss/faiss_index/v1/doc_all-MiniLM-L6-v2/faiss_index', help="Output path to save the index")
    # retrieve arguments
    parser.add_argument("--chunk_size_retrieve", type=int, default=10000, help="chunk size for parallel retrieval")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of documents to retrieve")
    parser.add_argument("--queries_file", type=str, default='dataset/v1/queries.train.tsv', help="Query file to use for retrieval")
    parser.add_argument("--output_run_file", type=str, default='faiss/faiss_run/v1/doc_all-MiniLM-L6-v2/queries.train.tsv/run_file', help="Output path to save the run file")
    # evaluate arguments
    parser.add_argument("--qrels_file", type=str, default='dataset/v1/qrels.train.tsv', help="Qrels file to use for evaluation")
    parser.add_argument("--output_eval_file", type=str, default='faiss/faiss_eval/v1/doc_all-MiniLM-L6-v2/queries.train.tsv/eval_file', help="Output path to save the evaluation file")
    parser.add_argument("--chunk_size_eval", type=int, default=20000, help="chunk size for parallel evaluation")
    parser.add_argument("--rank_cut", type=int, default=1000, help="chunk size for parallel evaluation")

    args = parser.parse_args()
    
    # collection = 'v1' # v2
    # model_name = 'all-MiniLM-L6-v2'
    # index_type = "doc" # doc or query based on the input_file choice if it is collection or queries that has been indexed
    
    # input_encode_file = f'dataset/{args.collection}/collection.tsv'
    ########      output_index_path = f'faiss_index/{args.collection}/{args.index_type}_{args.model_name}'
    
    # encode(model_name, input_file, output_path)
    if args.encode:
        # encode(args.model_name, args.input_encode_file, args.index_path, num_threads=args.num_threads, chunk_size=args.chunk_size_encode)

        encode_parallel(args.model_name, args.input_encode_file, args.index_path, num_threads=args.num_threads, chunk_size=args.chunk_size_encode)
    
    # top_k=1000
    # query_file = 'queries.train.tsv'
    # query_file_name = args.queries_file.split('/')[-1]
    
    # output_run_path = f'faiss_run/{args.collection}/{args.index_type}_{args.model_name}_{query_file}'
    # index_path = f'{output_index_path}/faiss_index'
    
    
    # retrieve(index_path, model_name, queries_file_path, output_run_path, top_k)
    if args.retrieve:
        retrieve_parallel(args.index_path, args.model_name, args.queries_file, args.output_run_file, args.top_k, args.chunk_size_retrieve, args.num_threads)
    # Ensure multiprocessing uses the correct number of CPUs


    # qrels_file = 'qrels.train.tsv'
    # qrels_file_path = f'dataset/{args.collection}/{args.qrels_file}'
    # output_eval_path = f'faiss_eval/{args.collection}/{args.index_type}_{args.model_name}_{args.query_file}'
    # run_file = f'{output_run_path}/run_file'
    if args.evaluate:
        # evaluate_run(args.output_run_file, args.qrels_file, args.output_eval_file)
        # evaluate_run_parallel(args.output_run_file, args.qrels_file, args.output_eval_file,  args.num_threads)
        evaluate_run_chunked(args.output_run_file, args.qrels_file, args.output_eval_file,  args.num_threads, args.chunk_size_eval, args.rank_cut)
if __name__ == '__main__':
    main()