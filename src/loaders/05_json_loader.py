from pathlib import Path
from langchain_community.document_loaders import JSONLoader
import json
from pprint import pprint

def load_json(path: str):
    loader = JSONLoader(path, jq_schema=".messages[].content",
                        text_content=False     # Required if content_key is used with dict inputs
                        )
    data = loader.load()
    return data

if __name__ == "__main__":
    data = load_json("data/facebook_chat.json")
    # data = json.load(open("data/facebook_chat.json", "r"))
    # file_path='data/facebook_chat.json'
    # data = json.loads(Path(file_path).read_text())
    pprint(data)
    
    # print(json.dumps(data["messages"][:2], indent=2))
    # print(f"Loaded {len(documents)} documents.")
    # print(documents[0].page_content[:500])