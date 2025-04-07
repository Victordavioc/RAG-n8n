import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai

# 1. Carrega as variáveis de ambiente e configura Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("A chave GEMINI_API_KEY não foi encontrada no .env")
genai.configure(api_key=GEMINI_API_KEY)

# 2. Carrega o PDF com estrutura semântica (títulos, listas etc.)
loader = UnstructuredPDFLoader("documents/Catalogo_Forever_Living.pdf", mode="elements")
docs = loader.load()

# 3. Junta todos os textos em um só (para dividir por produtos via regex)
texto_completo = "\n".join([doc.page_content for doc in docs])

# 4. Separa por blocos de produtos com base em padrões como "Forever XYZ®"
padroes = re.split(r"(?=\n?Forever\s[A-Z][^\n]+®)", texto_completo)

# 5. Cria documentos estruturados, ignorando lixo e blocos curtos
produtos_docs = []
for bloco in padroes:
    bloco_limpo = bloco.strip()
    if len(bloco_limpo) > 100 and "Forever" in bloco_limpo:
        produtos_docs.append(Document(page_content=bloco_limpo))

# 6. Split adicional leve (caso algum bloco seja grande demais)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " "]
)
docs_split = splitter.split_documents(produtos_docs)

# 7. Geração de embeddings com modelo semântico robusto
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs_split, embedding_model)

# 8. Retriever com k=10 para maior cobertura semântica
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
model = genai.GenerativeModel("gemini-1.5-flash")

# 9. Loop de interação
while True:
    pergunta = input("Você: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break

    documentos = retriever.invoke(pergunta)
    contexto = "\n".join([doc.page_content for doc in documentos])
    print("🔍 Contexto enviado ao Gemini:\n", contexto[:1000], "\n---")

    prompt = f"""
Você é um assistente especializado no catálogo de produtos da Forever Living.
Responda **com base apenas no conteúdo abaixo**, sem inventar informações externas.

CONTEÚDO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:"""

    resposta = model.generate_content(prompt)
    print(f"Bot: {resposta.text.strip()}")
