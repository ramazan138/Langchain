import os
from typing import Any, Dict, List

import bs4
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# Default embedding endpoint credentials to the LLM endpoint if not provided separately.
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL") or LLM_API_URL
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or LLM_API_KEY
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


if not LLM_API_URL:
    raise ValueError("Missing LLM_API_URL. Set it in your environment or .env file.")

if not EMBEDDING_API_URL:
    raise ValueError(
        "Missing EMBEDDING_API_URL. Set it (or reuse LLM_API_URL) in your environment or .env file."
    )


def _build_headers(api_key: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalise_role(message: BaseMessage) -> str:
    role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
    return role_mapping.get(message.type, message.type)


def invoke_internal_llm(prompt_value) -> str:
    messages = []
    for message in prompt_value.to_messages():
        messages.append({"role": _normalise_role(message), "content": message.content})

    payload: Dict[str, Any] = {"messages": messages}
    if LLM_MODEL_NAME:
        payload["model"] = LLM_MODEL_NAME

    response = requests.post(
        LLM_API_URL,
        headers=_build_headers(LLM_API_KEY),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        if "content" in data:
            return data["content"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "")

    raise ValueError(
        "Unexpected response from LLM API. Adapt `invoke_internal_llm` to match your service."
    )


class InternalAPIEmbeddings:
    """Embedding client that calls a company-internal HTTPS endpoint."""

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload: Dict[str, Any] = {"input": texts}
        if self.model_name:
            payload["model"] = self.model_name

        response = requests.post(
            self.api_url,
            headers=_build_headers(self.api_key),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            if "embeddings" in data:
                return data["embeddings"]
            if "data" in data:
                return [item["embedding"] for item in data["data"]]

        raise ValueError(
            "Unexpected response from embedding API. Adjust `_embed_batch` accordingly."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


llm = RunnableLambda(invoke_internal_llm)
embedding_client = InternalAPIEmbeddings(
    api_url=EMBEDDING_API_URL,
    api_key=EMBEDDING_API_KEY,
    model_name=EMBEDDING_MODEL_NAME,
)


# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(f"Loaded {(docs)} documents from the blog.")  

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_client)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        print(chunk, end="", flush=True)
