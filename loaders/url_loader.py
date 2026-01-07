# from langchain_community.document_loaders import UnstructuredURLLoader

# def load_urls(urls):
#     urls = [u for u in urls if u.strip()]
#     loader = UnstructuredURLLoader(urls=urls)
#     return loader.load()
from langchain_community.document_loaders import WebBaseLoader

def load_urls(urls):
    clean_urls = [u for u in urls if u.strip()]
    loader = WebBaseLoader(clean_urls)
    return loader.load()


