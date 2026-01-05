from pathlib import Path
from langchain_community.document_loaders import JSONLoader
import json

# def load_json(path: str):
#     loader = JSONLoader(path, jq_schema=".messages[]",
#                         content_key="content",
#                         text_content=False     # Required if content_key is used with dict inputs
#                         )
#     docs = loader.load()
#     return docs

if __name__ == "__main__":
    # documents = load_json("data/facebook_chat.json")
    data = json.load(open("data/facebook_chat.json", "r"))
    # file_path='facebook-chat.json'
    # data = json.loads(Path(file_path).read_text())
    # print(data)
    print(json.dumps(data["messages"][:2], indent=2))
    # print(f"Loaded {len(documents)} documents.")
    # print(documents[0].page_content[:500])