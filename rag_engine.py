import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from config import (
    EMBEDDING_MODEL,
    INDEX_NAME,
    PINECONE_API_KEY,
    OPENROUTER_API_KEY,
    LLM_MODEL,
    GENERATION_CONFIG,
)

# تحميل متغيرات البيئة من ملف .env
load_dotenv()


class RAGEngine:
    def __init__(self):
        # 1. إعداد الـ Embedder
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 2. إعداد OpenRouter Client
        # OpenRouter بيستخدم نفس interface الـ OpenAI SDK بس بـ base_url مختلف
        api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        if not api_key:
            print("❌ Error: OPENROUTER_API_KEY not found in .env file")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        # 3. إعداد Pinecone
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY") or PINECONE_API_KEY
            pc = Pinecone(api_key=pinecone_api_key)
            self.index = pc.Index(INDEX_NAME)
        except Exception as e:
            print("❌ Pinecone Connection Failed:", e)
            self.index = None

    def retrieve(self, query, top_k=5):
        if self.index is None:
            return []

        # تحويل السؤال لـ Vector
        query_embedding = self.embedder.encode(query).tolist()

        # البحث في Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        return [
            match.metadata["text"]
            for match in results.matches
            if match.metadata and "text" in match.metadata
        ]

    def build_prompt(self, query, context):
        context_text = "\n\n".join(context)

        return [
            {
                "role": "system",
                "content": (
                    "You are MedAssist, an intelligent and knowledgeable medical assistant. "
                    "You have access to a curated medical knowledge base, and your job is to give clear, accurate, and helpful answers based on it.\n\n"

                    "Guidelines:\n"
                    "- Answer naturally and conversationally, like a knowledgeable doctor explaining to a patient.\n"
                    "- Be thorough: explain the condition, causes, symptoms, or treatment as relevant to the question.\n"
                    "- Use the provided context as your primary source. If the answer is fully or partially there, use it.\n"
                    "- If the context only partially covers the question, answer what you can and note what's missing.\n"
                    "- Only if the topic is completely absent from the context, say: 'This specific topic isn't covered in my current knowledge base.'\n"
                    "- Never fabricate drug names, dosages, or clinical data.\n"
                    "- Keep a professional but warm and approachable tone."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the relevant medical context:\n\n{context_text}\n\n"
                    f"Patient question: {query}"
                ),
            },
        ]

    def call_llm(self, messages):
        if self.client is None:
            return "❌ OpenRouter client is not initialized. Check your OPENROUTER_API_KEY."

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                max_tokens=GENERATION_CONFIG["max_output_tokens"],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ OpenRouter API Error: {str(e)}"

    def ask(self, query):
        # العملية كاملة: Retrieval -> Prompt -> Generation
        context = self.retrieve(query)

        if not context:
            return "⚠️ No relevant medical context found in the database."

        messages = self.build_prompt(query, context)
        answer = self.call_llm(messages)
        return answer


# --- تجربة الكود ---
if __name__ == "__main__":
    engine = RAGEngine()
    user_query = input("Ask a medical question: ")
    print("\n--- Generating Answer ---")
    print(engine.ask(user_query))