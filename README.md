## 🤖🔍 Multi-Agent System – Project Perplexity.ai
Um sistema de múltiplos agentes de IA que realizam buscas inteligentes na web e compilam os resultados em uma resposta estruturada. Tudo isso integrado a uma interface web com Streamlit.

## 📌 Descrição
Este projeto simula um comportamento colaborativo entre cinco agentes de IA:

4 agentes de busca (Search Agents): utilizam a API do Tavily para realizar buscas paralelas na web.

1 agente compilador (Final Writer Agent): recebe os resultados de todos os agentes de busca e gera uma resposta final estruturada, de forma coesa e clara.

O processo é iniciado a partir de uma interface web desenvolvida com Streamlit, onde o usuário insere um prompt. A resposta é construída dinamicamente em tempo real.


## 🚀 Tecnologias Utilizadas
Python

LangGraph

Gemini API (Google AI)

Tavily API

Pydantic

Streamlit

## ⚙️ Como Executar
### Clone o repositório:
`git clone https://github.com/seuusuario/ai-agent-research-wikipedia.git`

### Crie e ative um ambiente virtual (opcional, mas recomendado):
`python -m venv venv`

`.\venv\Scripts\activate`


### Instale as dependências:
`pip install streamlit langgraph langchain python-dotenv langchain-google-genai langchain-ollama tavily-python`


### Configure sua chave de API da OpenAI como variável de ambiente (.env):
GEMINI_API_KEY=sua-chave-aqui
TAVILY_API_KEY=sua-chave-aqui

### Execute o projeto via terminal:
`python .\graph.py`
