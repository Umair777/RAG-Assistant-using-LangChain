from pathlib import Path

if __name__ == "__main__":
    print("This is a utility module for loading text files using LangChain's TextLoader.")
    # project_root = Path(__file__).resolve().parents[2]
    # print(f"Project root: {project_root}")
    # data_path = project_root / "data" / "new-Policies.txt"
    # with open(data_path, encoding="utf-8") as f:
    #     content = f.read()
    # print(f"Content length: {len(content)} characters.")
    # project_root = Path(__file__).resolve().parents[2]
    with open("../../data/new-Policies.txt") as f:
        content = f.read()
        
    print(f"Content length: {len(content)} characters.")
    