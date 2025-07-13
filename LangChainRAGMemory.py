import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Define o caminho para a pasta onde os PDFs estão armazenados
# Usar 'r' antes da string ajuda a evitar problemas com barras invertidas no Windows
pasta = "PASTA_COM_DOCUMENTOS_PDF"
documentos_carregados = []

# Carrega os documentos PDF da pasta especificada
print("Carregando documentos...")
for file in os.listdir(pasta):
    if file.endswith(".pdf"):
        caminho_completo = os.path.join(pasta, file)
        loader = PyPDFLoader(caminho_completo)
        documentos_carregados.extend(loader.load())

# Divide os Documentos em Chunks
print("Dividindo documentos em pedaços...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documentos_carregados)

# Cria o vetor de embeddings e o banco de dados Chroma
print("Criando e armazenando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Carrega do disco se já existir, senão, cria um novo
# (Para evitar reprocessar tudo, você pode adicionar uma lógica de verificação aqui)
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
print("Banco de vetores pronto.")

# Inicializa o modelo de linguagem com Ollama
llm = ChatOllama(model="gemma3n:e4b")

# Cria um retriever a partir do vetor de embeddings
retriever = vectorstore.as_retriever()

# --- Definição dos Prompts para a Conversa ---

contextualize_q_system_prompt = """Dada uma conversa e uma última pergunta do usuário, \
reescreva a última pergunta para que seja uma pergunta independente, caso a pergunta \
seja contextual. Não precisa reescrever se a pergunta já for independente."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """Você é um assistente para tarefas de perguntas e respostas. \
Use os seguintes trechos de contexto recuperados para responder à pergunta. \
Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta. \
Sempre que possível, mencione o nome do arquivo fonte em sua resposta.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- Criação da Cadeia Conversacional (A forma correta) ---

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Renomeei 'Youtube_chain' para um nome mais genérico
Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

# Esta é a definição CORRETA da rag_chain com memória
rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)


# --- Loop Interativo para a Conversa ---

chat_history = []

print("\n--- Agente pronto! Converse com seus documentos. Digite 'sair' para terminar. ---")

while True:
    pergunta = input("Sua pergunta: ")
    if pergunta.lower() == 'sair':
        print("Até logo!")
        break

    # Invoca a RAG chain com a pergunta e o histórico
    response = rag_chain.invoke({"input": pergunta, "chat_history": chat_history})

    # Imprime a resposta
    print("\nResposta:", response["answer"])
    print("-" * 50)

    # Atualiza o histórico do chat com a nova interação
    chat_history.extend([HumanMessage(content=pergunta), AIMessage(content=response["answer"])])