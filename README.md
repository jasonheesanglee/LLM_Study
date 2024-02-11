# LLM Study
This repository is a record of learning LangChain.
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

