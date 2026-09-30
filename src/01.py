from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 加载本地markdown文件
markdown_path = "../data/C1/markdown/easy-rl-chapter1.md"
loader = DoclingLoader(file_path=markdown_path)
docs = loader.load()

# 文本分块
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 嵌入模型
