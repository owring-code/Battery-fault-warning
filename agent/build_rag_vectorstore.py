import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 配置 Hugging Face 国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def build_industrial_knowledge_base(pdf_folder_path: str, vector_store_path: str) -> None:
    print("步骤 1：开始加载工业运维手册...")
    documents = []

    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, file_name)
            loader = PDFPlumberLoader(file_path)
            documents.extend(loader.load())

    print(f"共加载了 {len(documents)} 页文档。")

    print("步骤 2：进行文本切块 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "！", "？", " "],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档被切分成了 {len(chunks)} 个文本块。")

    print("步骤 3：加载本地 Embedding 模型进行向量化...")
    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    print("步骤 4：构建 FAISS 向量数据库并持久化保存...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"知识库已成功构建并保存至 {vector_store_path}！")


if __name__ == "__main__":
    pdf_dir = "./manuals"
    save_dir = "./faiss_industrial_index"

    os.makedirs(pdf_dir, exist_ok=True)

    # 取消注释后可直接构建
    # build_industrial_knowledge_base(pdf_dir, save_dir)
