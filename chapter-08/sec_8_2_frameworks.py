"""Section 8.2: the same tasks with LangChain and LlamaIndex.

    pip install langchain langchain-openai llama-index
    export OPENAI_API_KEY=...
    python sec_8_2_frameworks.py langchain openai:gpt-4o-mini
    python sec_8_2_frameworks.py llamaindex notices/

Use any current "provider:model-name" string that LangChain supports. LlamaIndex uses
OpenAI for both the model and the embeddings by default, so it also needs the key; the
folder should hold a few text or PDF files to index.
"""
import sys

which = sys.argv[1] if len(sys.argv) > 1 else ""
if which == "langchain":
    from langchain.chat_models import init_chat_model
    model = init_chat_model(sys.argv[2])                 # the same call for many providers
    reply = model.invoke([("system", "Explain tax notices in plain English."),
                          ("user", "What is a demand notice?")])
    print(reply.content)
elif which == "llamaindex":
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    documents = SimpleDirectoryReader(sys.argv[2]).load_data()   # parse a folder of files
    index = VectorStoreIndex.from_documents(documents)           # chunk, embed and index
    engine = index.as_query_engine(similarity_top_k=5)
    print(engine.query("Which notices allow 30 days for a response?"))
else:
    print(__doc__)
