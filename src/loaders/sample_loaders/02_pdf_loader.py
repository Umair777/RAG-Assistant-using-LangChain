from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

if __name__ == "__main__":
    pages = load_pdf("data/instructlab.pdf")
    print(f"Loaded {len(pages)} pages.")
    print("---- Page 1 preview ----")
    # print(pages[0].page_content[:500])
    
    for i, page in enumerate(pages[:3], start=1):
        print(f"---- Page {i} content ----")
        print(page.page_content[:500])
    