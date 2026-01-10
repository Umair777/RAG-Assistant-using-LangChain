from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
    )
    return text_splitter.split_text(text)
