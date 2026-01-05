from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from pprint import pprint

if __name__ == "__main__":
    # loader = CSVLoader(file_path='data/mlb_teams_2012.csv')
    # data = loader.load()
    # pprint(data)
    loader = UnstructuredCSVLoader(file_path='data/mlb_teams_2012.csv', mode='elements')
    data = loader.load()
    # pprint(data[0].page_content)
    print(data[0].metadata["text_as_html"])
    