# GNN-QPP

## Dataset
1. Download datasets

MS MARCO Passage V2:

`wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar`

MS MARCO Passage V1:

`wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz`

we need to generate a graph dataset consisting of all the queries and documents. for the MS MARCO V1 dataset. 
• Documents: 8,841,823 web documents (in the full document collection).
• Queries: 808,731 unique queries.
• Relevance Judgments: 532,761 (labelled query-document pairs).

2.	Graph Structure:

Nodes: 

- Queries
  - different attribution can be assigned to this  like relevance score
  - first assumption is to generate the embedding and assign the embedding as the initial value for the 
- Documents
  - Same embeddings generate by the dense retriever can be used here

Edges:

- Query-Query Similarity: Captures semantic or lexical similarity between queries. 
  - similarity can be measured by different cross-encoder models that is used for semantic search.
  - models: sentence-transformers/msmarco-MiniLM-L6-cos-v5, 
  - we will save the embedding to save the computation cost
- Document-Document Similarity: Helps model document relationships based on content, embeddings, or retrieval scores.
  - same model for the Query-Query can be  used here
- Query-Document Retrieval Relationship: Links queries to their retrieved documents, weighted by retrieval scores or ranking positions.
  - similarity score
  - relevance score (since we are using the dense retrievers we can say that the relevance score is actually the similarity score)
