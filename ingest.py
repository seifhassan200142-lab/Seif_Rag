import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec  # ✅ مش PineconeGRPC
from config import *


def load_documents(data_path="data"):
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{INDEX_NAME}' created.")
        else:
            print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

        return pc.Index(INDEX_NAME)

    except Exception as e:
        print("❌ Pinecone Error:", e)
        return None


def embed_and_store(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = init_pinecone()

    if index is None:
        return

    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk.page_content).tolist()
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": chunk.page_content}
        })

        # batch upload كل 50
        if len(vectors) == 50:
            index.upsert(vectors=vectors)
            print(f"  📤 Uploaded batch up to chunk {i+1}")
            vectors = []

    # الـ batch الأخير
    if vectors:
        index.upsert(vectors=vectors)
        print(f"  📤 Uploaded final batch ({len(vectors)} chunks)")

    print("✅ Data successfully uploaded to Pinecone")


if __name__ == "__main__":
    docs = load_documents()
    print(f"📄 Loaded {len(docs)} pages")
    chunks = split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks")
    embed_and_store(chunks)