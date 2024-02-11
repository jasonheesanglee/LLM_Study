# LLM_Study

## What is LangChain?
<sub>(from [LangChain Github](https://github.com/langchain-ai/langchain))</sub><br>
LangChain is a framework for developing applications powered by language models.<br>
It enables applications that:<br>
**Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)<br>
**Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)<br>
---> LangChain is a Tool that helps utilize the Language Models at their best.<br>

## Why should we use LangChain?
- ChatGPT
  1. **Limited access to data**: ----> ***Searching the information based on VectorStore or Search Agent utilization.***<br>LLM used on ChatGPT is trained on the data up until a certain date, and if the user asks about a more recent event, ChatGPT will not provide an answer or provide a fake answer (hallucination).
  2. **Limited number of Tokens**: ----> ***Splitting the document using TextSplitter***<br>Each GPT model has limits on the token counts; therefore, it is hard to implement in real-life/business usage.
  3. **Hallucination**: ----> ***Insert a prompt to limit the deep learning model to answer only based on the given document.***<br>It often responds with irrelevant or fake information when the user asks about Facts.<br>
***LangChain can overcome those limitations of ChatGPT.***<br><br>
- **Fine-Tuning**:<br>Updating the weights of the deep learning models to the desired purpose.
- **N-Shot Learning**:<br>Suggest 0 ~ N number of the output example, it controls the deep learning model to produce a similar output that fits the desired purpose.
- **In-Context Learning**: ----> ***LangChain***<br> Suggests a context to the deep learning model, and controls the deep learning model to produce the output based on this context.
