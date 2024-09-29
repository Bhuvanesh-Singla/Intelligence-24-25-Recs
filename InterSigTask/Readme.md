# GDGxIntel
## Task ID: Hogwarts Q&A  
#### `NLP`, `Generative AI`, `RAG`, `ChatBot`

**Difficulty**: `Hard`

### Description  
Create a **RAG (Retrieval-Augmented Generation)** system that answers questions using the book **"Harry Potter and the Prisoner of Azkaban"** as the primary knowledge base. This system should be able to handle queries related to characters, spells, locations, and magical events, retrieving contextually accurate information and generating lore-true responses.

Participants will leverage this text to build a structured index for retrieval, generate embeddings, and deploy a Harry Potter-themed Q&A system. For added fun, include context references and quotes in the generated answers.

You can access the full text of the book from the following link:  
**[Harry Potter and the Prisoner of Azkaban PDF](https://ia902903.us.archive.org/12/items/FantasyFictionebookcollection/Harry%20Potter/3%20-%20Harry%20Potter%20and%20the%20Prisoner%20of%20Azkaban.pdf)**.

### **Steps to Complete the Challenge**:

1. **Data Collection & Ingestion**:  
   Download the book from the provided link and parse it into a machine-readable format. Make sure to structure the data, preserving the chapters and significant sections. You may want to split the text based on chapters, events, or specific scenes.

2. **Data Chunking & Preprocessing**:  
   Break down large paragraphs into smaller chunks of 100-150 words. Ensure that each chunk contains coherent, self-contained information.

3. **Embedding Generation**:  
   - Use a pre-trained embedding model like ``all-MiniLM-L6-v2`` from Sentence Transformers to convert each text chunk into dense vector representations.
   - The embeddings should capture semantic meaning, making it easy to retrieve the most contextually relevant text.

4. **Vector Database Integration**:  
   Store the embeddings in a vector database such as `ChromaDB`, `FAISS`, or `Milvus` for efficient similarity searches and quick lookups.

5. **Query Handling & Retrieval**:  
   - Implement a query pipeline using the embedding model to process user queries.
   - Convert the query into an embedding and use the vector database to find the `top N` most relevant text chunks.

6. **Contextual Response Generation**:  
   - Use the retrieved chunks with a generative language model (like `Gemini` or `LLaMA`) to create a coherent response that incorporates quotes and references to specific parts of the book.
   - Ensure that the generated output maintains the tone and style of the Harry Potter universe.

7. **Serve via FastAPI**:  
   - Expose your RAG system through **FastAPI** endpoints. 
The /query endpoint should accept user queries like *“What is the significance of the Marauder’s Map?”* or *“How does Sirius Black escape from Hogwarts?”* and return a contextually accurate and engaging answer. 
The endpoint should:
Accept `POST` requests with a `JSON` body containing the user's question.
Process the query through your RAG pipeline.
Return a `JSON` response with the answer, relevant quotes, and metadata
   
8. **Develop an Interface**:  
   - Build a basic web application. It should have a chat-like interface for asking questions and receiving answers. 

Bonus Feature ( Optional ):

1. **Time-Turner**:  
   - Create a "Time-Turner" feature that allows users to view the conversation history and jump back to previous points in the chat.
   - You can store the previous conversations in the local storage itself instead of a database.

### **Useful Resources**:
- [Harry Potter and the Prisoner of Azkaban PDF](https://ia902903.us.archive.org/12/items/FantasyFictionebookcollection/Harry%20Potter/3%20-%20Harry%20Potter%20and%20the%20Prisoner%20of%20Azkaban.pdf)
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Sentence Transformers for Embeddings](https://huggingface.co/sentence-transformers)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)

### Tips:
1. When chunking the text, experiment with different chunk sizes and overlaps to find the optimal balance between context preservation and retrieval quality.
2. Consider adding metadata tags (chapter names, etc.) to the chunks at the time of preprocessing, this can help with retrieval.
