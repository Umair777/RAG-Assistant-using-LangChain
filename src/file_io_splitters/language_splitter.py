from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")
    
    # Call the function
    hello_world()
"""

if __name__ == "__main__":
    # for each_lang in Language:
    #     print(f"Language: {each_lang}, ISO Code: {each_lang.value}")
        
    # print([e.value for e in Language])
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=50,
        chunk_overlap=10,
    )
    
    python_chunks = python_splitter.create_documents([PYTHON_CODE])
    print(python_chunks)
    
    