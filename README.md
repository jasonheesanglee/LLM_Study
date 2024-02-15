# LLM Study
This repository is a record of learning LangChain.<br>
The [contents](https://www.youtube.com/playlist?list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v) are from a YouTube Channer [모두의AI](https://www.youtube.com/@AI-km1yn)

## What is LangChain?
<sub>(from [LangChain Github](https://github.com/langchain-ai/langchain))</sub><br>
LangChain is a framework for developing applications powered by language models.<br>
It enables applications that:<br>
**Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)<br>
**Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)<br>
---> LangChain is a Tool that helps utilize the Language Models at their best.<br>

## Why should we use LangChain?
- **ChatGPT**<sub>for example</sub>
  1. **Limited access to data**: ----> ***Searching the information based on VectorStore or Search Agent utilization.***<br>LLM used on ChatGPT is trained on the data up until a certain date, and if the user asks about a more recent event, ChatGPT will not provide an answer or provide a fake answer (hallucination).
  2. **Limited number of Tokens**: ----> ***Splitting the document using TextSplitter***<br>Each GPT model has limits on the token counts; therefore, it is hard to implement in real-life/business usage.
  3. **Hallucination**: ----> ***Insert a prompt to limit the deep learning model to answer only based on the given document.***<br>It often responds with irrelevant or fake information when the user asks about Facts.<br>
***LangChain can overcome those limitations of ChatGPT.***<br><br>
- **Fine-Tuning**:<br>Updating the weights of the deep learning models to the desired purpose.
- **N-Shot Learning**:<br>Suggest 0 ~ N number of the output example, it controls the deep learning model to produce a similar output that fits the desired purpose.
- **In-Context Learning**: ----> ***LangChain***<br> Suggests a context to the deep learning model and controls the deep learning model to produce the output based on this context.

##  Types of LangChain and their functions

**LLM**: Large Language Model -> Key component of LangChain that functions as an engine of the Generative Models.<br>
> Example: GPT-3.5, PaLM-2, LLaMA, StableVicuna, WizardLM, MPT, etc...

**Prompts**: A Statement instructing LLM<br>
> Elements: Prompt Templates, Chat Prompt Templates, Example Selectors, Output Parsers.

**Index**: A structuring module for LLM to search the document easily and conveniently.
> Example: Document Loaders, Text Splitters, VectorStores, Retrievers, etc...

**Memory**: A module that enables a continuous conversation based on the chat history.
> Example: ConversationBufferMemory, Entity Memory, Conversation Knowledge Graph Memory, etc...

**Chain**: A key component that enables a continuously calling LLM by forming a chain of LLM.
> Example: LLM Chain, Question Answering, Summarization, Retrieval Question/Answering, etc...

**Agents**: A module that enables the LLM to do the job that used to cannot be done with the existing Prompt Template.
> Example: Custom Agent, Custom MultiAction Agent, Conversation Agent, etc...

<p align='center'>
  <img width=750 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/d88773d7-2723-4282-9050-ea5fa2ae6553'>
</p>

### Example: Building PDF ChatBot
1. Uploading the document using Document Loader
   - Document Loader helps LLM to answer "Based on xxx.pdf page ### "
   - Confluence and Notion pages can also be imported as well if a different loader is used.
2. Splitting Text using TextSplitter
   - Splitting PDF documents enables the LLM to find relevant information from large PDFs (out of token limits).
3. Embed to VectorStore
   - Convert the PDF to numerics to enable the LLM to understand the document.
4. VectorStore Retriever
   - Search the embeddings to extract the information that has a high correlation with the question.
5. QA Chain
   - Extract the document that has a high correlation with the question.
   1. Hand over the question and a relevant text to the LLM as a Prompt, than the LLM understands it.
   2. Hand over to LLM again to repeat step i.
   3. Hand over again to format the answer output.

## LangChain Structure
Most of the NLP models are based on Transformer Architecture.<br>
Encoder and Decoder are utilized either independently or together based on the model's purpose.<br>
**Encoder** -> Understanding the Context<br>
**Decoder** -> Returns the output based on the output of the Encoder.<br>

There are:<br>
Encoder Only Models <sub>in Pink</sub><br>
Decoder Only Models <sub>in Blue</sub><br>
Encoder-Decoder Models <sub>in Green</sub><br>
<p align='center'>
  <img width=550 src="https://github.com/jasonheesanglee/LLM_Study/assets/123557477/66d0d578-2682-4c28-8665-dce79b173360">
</p>

## What should we learn to build a ChatBot
[ChatGPT Sample ipynb](https://github.com/jasonheesanglee/LLM_Study/blob/main/chatgpt_api_sample.ipynb)
1. Learn to conversate using ChatGPT API 
2. Learn the base structure of ChatGPT API
3. Learn the useful functions for building a ChatBot

## What is Prompt?
Prompt means the input to the model.<br>
This input is barely hard-coded but, in many cases, is composed of various components.<br>
"Prompt Template" is responsible for the structure of the input.<br>
LangChain supports various Classes and Functions to make the building prompt process easier.<br>

### Prompt Template
[Prompt Templates](https://github.com/jasonheesanglee/LLM_Study/blob/main/Prompt_Templates.ipynb)<br>
Recommend a food that contains {food 1} and {food 2} and tell me the recipe for it.

## Retrieval

### What is RAG?
RAG -> Retrieval Augmented Generation<br>
RAG is a framework that helps LLM refer to an external document to answer.<br>

**RAG Process**
1. Document Loaders -><br>Responsible for loading the external document
2. Text Splitters -><br>Responsible for splitting long documents into small pieces
3. Vector Embedding module -><br>Responsible for converting the strings (user question) into numbers and storing them in Vector Stores<br>Responsible for converting the strings (document) into numbers and storing them in Vector Stores
4. Retrievers -><br>Responsible for searching the most similar sentence in document with the user's question.<br>Hand it over to Chain
5. Chain -><br>Responsible for generating the answer.

### Document Retrieval
***<u>RAG Structure</u>***
1. User: Input question
2. Q/A system: search for a similar sentence in an external Data Storage<sub>(Vector DB / Feature Store, etc.)</sub>, and question LLM with the external sentence. 
3. LLM: Answers the question referring to the document.

### Document Loaders
A module that loads various types of documents as a special object for RAG
Document's structure loaded by Document Loaders
- page_content
  - Contents of the document
- metadata
  - Information about the document (where it is stored, what is the title of the document, what part of the document the information is (source of information), etc.)

### Text Splitters
A module that divides a document into a number of chunks so that an LLM that has token limits can refer to a number of sentences.<br>
The chunks are stored in the Vector store as embedding vectors.<br>
Each Embedding vectors is each chunk.<br>
1 Chunk = 1 Embedding Vectors (Converted into numerics)<br>
Finding the vectors of sentences that have high correlations with the vectors of the user's question.<br><br>
<p align='center'>
  <img width=550 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/ffcdee60-0daf-4948-94f0-be24b81b2693'>
</p>

#### Types of Text Splitters
- Character TextSplitter
  -   It splits the text per each separator. (if the selected separator is enter, only enter will be considered)
  -   This might not fit into max_token
    -   `chunk_overlap` allows different chunks to include a part of the previous chunk
- ***Recursive TextSplitter*** -> Used in most cases
  -  It splits the text per different separators, recursively. (enter -> full stop -> comma -> ... (recursively))
  -  This can fit into max_token
 
- Other Splitters
  - In most of the cases, Character Text Splitter and Recursive Text Splitter will do the work.
  - However, for some cases like "codes" and "LaTex", these two splitters might fail on reading the document.
  - Therefore, some special additions are required.<br><br>
  - For codes, we can import `Language` from langchain.text_splitter

- Token Level Splitters
  - The purpose of the text splits is for LLM to take tokens as input as much as its max_len.
  - Therefore, we can also split the text based on the number of tokens by importing modules such as `tiktoken`.
 
### Text Embeddings
Text Embeddings converts the texts into numeric values, enables to compare the similarity between different sentences.<br>
In most cases, it easily embeds with pre-trained models with large-scale corpus.<br>
By using these Pre-trained Embedding Models, we can embed our sentences without training.<br>
<p align='center'>
  <img width=550 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/7e3064d8-496b-410d-82b2-44a2f66923a0'>
</p> 

### Vector Store
Vector Store stores the embeddings of the text converted by the text embedding module/
- ***Pure Vector Databases***
  - Can store only vectors in the database
  - Contains many convenient tools such as update, delete, etc.
  - eg. Pinecone, Qdrant, Chroma (Free), Weaviate
- ***Vector Libraries***
  - Specializes in Vector Similarity computation.
  - Can store the vectors as well, but the performance as a Database is worse than the ones of Pure Vector Databases.
  - eg. Faiss (AI Semantic Search Tool)
- Text Search Databases
- Vector-Capable NoSQL Databases
- Vector-Capable SQL Databases
 
#### Chroma
Chroma is an OpenSourced Vector Database<br>
Basically, VectorStore stores the vectors temporarily.<br>
When passing the selected Text and Embedding Function to the `from_documents()` function, it converts the text to vectors with the given embedding function and stores it in the created temporal DataBase.<br>
Then, pass the query to the `similarity_search()` function; it searches the vector with high vector similarity and returns the output in the natural language.<br><br>
However, in many real-life cases, we need to store the document on a personal disk and call it whenever we need it.<br>
Saving Vector Store in a local database with the `persist()` function, we can pass the location of the local database where the vector store is saved when we call Chroma next time.<br>

#### FAISS
FAISS (Facebook AI Semantic Search) is a library that offers effective similarity searching and clustering in high-density vectors developed by Facebook.<br>
The algorithm that searches all size vector unions is included and can also search vectors that don't fit into RAM.<br>
Also, evaluation metrics and parameter tuning functions are included in the package as well.<br>

`max_marginal_relevance_search` -> It doesn't only return the query-relevant part of the document but varies the content of the output to output.<br>(texts from different parts of the document) (다양성 추구)

### Retrieval
Retriever takes the embedded text from the vector store as a context with the embedded query and hands it over to LLM.<br>
Retriever is an interface that returns the document when the atypical query is given.<br>
Retriever can only return or search the document without saving the document.<br>
In other words, Retriever is a module that makes the search easy.<br>

#### Chain
4 chains in Retriever.

- Stuff Documents Chain
  - The most simple structure of all chains.
  - A Token length issue could be raised.
  - Passes Query & Chunks of similar texts directly to LLM as a prompt: {'Question':query, 'Context':docs[0] + docs[1] + ... + docs[n]}
<p align='center'>
  <img width=750 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/3124106d-0797-474d-8b85-23936c3cc7d2)'>
</p>

- Map Reduce Documents Chain
  - Summarizes each text chunk (Map process), Generates the final summarization from text chunk summarization (Reduce process), passes the query and final summary to LLM.
  - Token length issue could be solved.
  - High time & computation cost.
<p align='center'>
  <img width=750 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/ac35143f-9a33-4a53-a388-252bc878984f)'>
</p>

- Refine Documents Chain
  - Aimed to get high-quality responses.
  - High time & computation cost.
  - Iterates over each chunk, passes query & chunk as a prompt, hands it over to LLM, takes each output, and adds it to the next iteration.
  - Not really used as LLM generates answers for each query.
  - If have enough time cost and requires a highly accurate response, this method could be implemented.
<p align='center'>
  <img width=750 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/208a1bc0-1cff-47a0-a1c6-ef0142f81516)'>
</p>

- Map Re-Rank Documents Chain
  - With a query, LLM generates responses for each chunk; the response consists of the answer and score (query & chunk's correlation score), ranks each response, and returns the response with the highest correlation score.
  - High time & computation cost.
  - This is used when accurate response is required.
<p align='center'>
  <img width=750 src='https://github.com/jasonheesanglee/LLM_Study/assets/123557477/7331483e-3ddc-47e1-8538-37cd6e6520d1)'>
</p>

## SQL Agent
Why do we need SQL?
- Easy to work with a large amount of data.
- We don't need to use SQL, but use LLM to use SQL.
Code: [https://github.com/jasonheesanglee/LLM_Study/blob/main/SQLAgent.ipynb](https://github.com/jasonheesanglee/LLM_Study/blob/main/SQLAgent.ipynb)

## Creating RAG system with Gemini Pro
Code: [https://github.com/jasonheesanglee/LLM_Study/blob/main/ChatBot_with_GeminiPro.ipynb](https://github.com/jasonheesanglee/LLM_Study/blob/main/ChatBot_with_GeminiPro.ipynb)

## Creating RAG system with OpenSource LLM
Code: [https://github.com/jasonheesanglee/LLM_Study/blob/main/Creating_RAG_with_OpenSource_LLM.ipynb](https://github.com/jasonheesanglee/LLM_Study/blob/main/Creating_RAG_with_OpenSource_LLM.ipynb)
### Why OpenSource LLM?
|API | OpenSource |
| :---: | :---: |
|1. Cost per usage | 1. High initial cost for server establishment, but no additional cost |
|2. Service Quality could decrease due to the usage of API | 2. No operational risk |
|3. Possible Data Leak | 3. No issue with Data Leak |

### Quantization
Most of the GPU on personal PC or server can barely load or infer 7B parameter LLM models.<br>
Therefore quantization is needed in prior using them.<br>
#### Process of quantization
1. Defining parameters with `BitsandBytes` `Config`.
  - `load_in_4bit = True` : Assigns the module to convert and load the model in 4bit.
  - `bnb_4bit_use_double_quant=True` : Utilizes double quantization on training and inference to increase the memory efficiency.
  - `bnb_4bit_quant_type="nf4"`: Utilizes NF4 (Normal 4 Bit Float) quantization instead of (4 bit Float) quantization.<br><sub>Details from [QLoRA](https://github.com/artidoro/qlora?tab=readme-ov-file#quantization) : Note that there are two supported quantization datatypes fp4 (four bit float) and nf4 (normal four bit float).<br>The latter is theoretically optimal for normally distributed weights and we recommend using nf4. </sub>
  - `bnb_4bit_comput_dtype=torch.bfloat16` : Utilizes torch.bfloat16 dtype during the computation.<br>Default : float32
2. Load the model with Quantization Config
  ```
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  ```
3. Print Model
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(78464, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=512, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=512, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear4bit(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=78464, bias=False)
)
```

## RAG techniques
It is comparatively easy to simply build RAG system.<br>
A high-end technique is needed to enhance the RAG quality for real-life product development.

### Struggles when building RAG
- Retriever's task is to provide a high-quality response by understanding the user's purpose.
  - People expects to get a great answer even with a simple query. ----> Multi-Query Retriever (개떡 찰떡)
  - Context in prior and after, should be well-considered by LLM. ----> Parent-Document
  - Do not need Semantic search but need query. ----> Self-Querying Retriever
    - Semantic Search simply compares the embeddings of the query and the data and sends it to LLM.
    - This may not show consistency when we ask the same question in a bit different wording to the model.
  - if the request is needed to be precisesly cut the data, query works better than the semantic search. ----> Time-Weighted Retriever
    - Take less reference from the older document. (Make preference on the recent documents.)

### Multi-Query Retriever
Regenerating a simple question to a number of similar question.

| Query | Regen | Vector Store | LLM |
| :--- | :--- | :--- | :--- |
| How good is Bank B's loan service? | -- Without Multi Query --> | ----> | Bank B's loan service is extra-ordinary.|
|  How good is Bank B's loan service? |  How good is the interest rate of Bank B's loan service?<br>How good is the conditions of Bank B's loan service?<br>How good is the user review of Bank B's loan service? | ---> | Summary of Bank B's Service:<br> - Interest Rate: ...<br> - Conditions: ...<br> - User Reviews...|
