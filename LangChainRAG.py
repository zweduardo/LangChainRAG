import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

#Carrega as variáveis de ambiente do arquivo .env
#load_dotenv()

#Define o caminho para a pasta onde os PDFs estão armazenados
pasta = "FOLDERPATH"  # Substitua por seu caminho real
documentos_carregados = []


# Carrega os documentos PDF da pasta especificada
for file in os.listdir(pasta):
    if file.endswith(".pdf"):
        caminho_completo = os.path.join(pasta, file)
        loader = PyPDFLoader(caminho_completo)
        documentos_carregados.extend(loader.load())

#Divide os Documentos em Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documentos_carregados)

# Cria o vetor de embeddings e o banco de dados Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

#Inicializa o modelo de linguagem com Ollama
llm = ChatOllama(model="gemma3n:e4b")

# Cria um retriever a partir do vetor de embeddings
retriever = vectorstore.as_retriever()

#Define a mensagem inicial do assistente
template = """Você é um assistente que responde perguntas sobre um conjunto de documentos.
Use os trechos de contexto fornecidos para responder à pergunta. Cada trecho de contexto
vem com o nome do arquivo de onde foi extraído. Sempre que possível, mencione o nome do
arquivo fonte em sua resposta. Se você não sabe a resposta, diga que não encontrou
informações nos documentos fornecidos.

Contexto: {context}

Pergunta: {question}

Resposta útil:"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    """Formata os documentos para exibir o nome do arquivo e o conteúdo."""
    return "\n\n".join(f"Fonte: {os.path.basename(doc.metadata.get('source', ''))}\nConteúdo: {doc.page_content}" for doc in docs)

# Cria o pipeline de RAG (Retrieval-Augmented Generation)
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

#Define a pergunta e executa o pipeline
pergunta = "O que muda de um transformer tradicional para um transformer de energia?"
response = rag_chain.invoke({"question": pergunta})
print(response)