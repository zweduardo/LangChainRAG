📄 Chat with Your Docs: Uma Base de Conhecimento com IA (CLI)
Este projeto é um script de linha de comando inteligente construído em Python usando LangChain e Ollama. Ele permite processar um conjunto de documentos PDF e fazer perguntas sobre eles, transformando uma pasta de arquivos em uma base de conhecimento consultável.

A aplicação utiliza uma arquitetura RAG (Retrieval-Augmented Generation) para encontrar as informações mais relevantes nos seus documentos e gerar uma resposta coesa usando um modelo de linguagem rodando 100% localmente.

✨ Features Atuais
Processamento de Múltiplos Documentos: Lê e processa todos os arquivos PDF de uma pasta especificada.

100% Local e Privado: Utiliza o Ollama para rodar modelos de linguagem (como o gemma3n:e4b) no seu computador. Seus documentos e perguntas nunca saem da sua máquina.

Banco de Vetores Persistente: Utiliza o ChromaDB para salvar os embeddings dos documentos em disco (chroma_db/). Isso significa que você só precisa processar os PDFs uma vez, tornando as execuções futuras muito mais rápidas.

Observabilidade com LangSmith: Pronto para ser integrado com o LangSmith para rastrear, depurar e monitorar cada passo da cadeia de RAG.

🛠️ Como Funciona
O script segue uma arquitetura clássica de RAG:

Carregamento de Dados: Os arquivos PDF da pasta designada são carregados na memória.

Divisão (Chunking): O texto de cada documento é dividido em pedaços menores e gerenciáveis.

Embedding e Armazenamento: Cada pedaço de texto é convertido em um vetor numérico (embedding) usando o modelo all-MiniLM-L6-v2 e armazenado em um banco de dados de vetores local e persistente (ChromaDB).

Recuperação (Retrieval): Ao fazer uma pergunta, o script busca os pedaços de texto mais relevantes no ChromaDB.

Geração: Os pedaços de texto recuperados são enviados como contexto para o modelo de linguagem (via Ollama), que gera a resposta final e a exibe no console.

🚀 Setup e Instalação
Siga os passos abaixo para rodar o projeto localmente.

1. Pré-requisitos
Python 3.9+

Ollama instalado e rodando.

Um modelo de linguagem baixado no Ollama (o script usa gemma3n:e4b, então rode ollama run gemma3n:e4b no seu terminal).

2. Prepare o Projeto
Clone este repositório (ou simplesmente use seu script LangChainRAG.py em uma nova pasta).

Crie uma pasta chamada documentos (ou o nome que preferir) e coloque seus arquivos PDF dentro dela.

3. Instale as Dependências
É recomendado criar um ambiente virtual:

Bash

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
Instale todas as bibliotecas necessárias. Crie um arquivo requirements.txt com o seguinte conteúdo e depois rode pip install -r requirements.txt.

Plaintext

# Conteúdo para requirements.txt
langchain
langchain-community
langchain-huggingface
langchain-ollama
python-dotenv
pypdf
chromadb
sentence-transformers
tensorflow # ou torch, dependendo da sua instalação
4. Configure o Script e as Variáveis de Ambiente
No script LangChainRAG.py, altere a variável pasta para o caminho da sua pasta de documentos:

Python

pasta = "documentos"  # Exemplo
Para o LangSmith, crie um arquivo chamado .env na raiz do projeto e adicione suas chaves:

Snippet de código

# --- Configuração do LangSmith (Opcional) ---
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="SUA_CHAVE_DE_API_DO_LANGSMITH"
LANGCHAIN_PROJECT="Meu-Chat-com-Docs-CLI"
No script LangChainRAG.py, descomente a linha #load_dotenv() para ativar a leitura do arquivo .env.

ιχ Observabilidade com LangSmith
Este projeto está pronto para usar o LangSmith para fornecer total transparência sobre o que acontece na execução. Ativando o LangSmith, você poderá visualizar o "trace" (rastreamento) de cada pergunta, o que é inestimável para depurar e entender o fluxo de dados.

🔗 Link para o Projeto Público no LangSmith
Você pode acompanhar as interações e a performance deste projeto em tempo real através do link público abaixo:

https://smith.langchain.com/public/9fe16d63-9e03-4f80-b5ed-ac4430703d0d/r

🏃 Como Rodar o Script
Certifique-se de que o serviço do Ollama está rodando.

No script, altere a variável pergunta para a pergunta que você deseja fazer sobre seus documentos:

Python

pergunta = "Qual é a principal conclusão do documento X?"
Execute o seguinte comando no seu terminal:

Bash

python LangChainRAG.py
A resposta será impressa diretamente no console.
