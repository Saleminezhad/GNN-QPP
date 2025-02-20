# GNN-QPP

## Dataset
we need to generate a dataset consisting of all the queries and documents for the MS MARCO dataset. 
1.	Graph Structure:

Nodes: 
    -   Queries
        -   different attribution can be assigned to this  like relevance score
        -   first assumpsion is to generate the embedding and assign the embedding 
    - Documents
Edges:
    • Query-Query Similarity: Captures semantic or lexical similarity between queries. 
        • similarity can be measured by different cross-encoder models that is used for semantic search.
        • models: sentence-transformers/msmarco-MiniLM-L6-cos-v5, 
        • we will save the embedding to save the computation cost
    • Document-Document Similarity: Helps model document relationships based on content, embeddings, or retrieval scores.
        • same model for the Query-Query can be  used here
    • Query-Document Retrieval Relationship: Links queries to their retrieved documents, weighted by retrieval scores or ranking positions.
        • similarity score
        • relevance score (since we are using the dense retrievers we can say that the relevance score is actually the similarity score)
