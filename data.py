import os

import pydantic
from datasets import Dataset

# each file is a md wiki page
DATA_DIR = "/Users/williambrown/dev/agents-course/agent-engineering/lec1-agent-patterns/data/wiki"


class Page(pydantic.BaseModel):
    title: str
    id: str
    content: str


def title_to_id(title: str) -> str:
    return title.lower().replace(" ", "_")


def get_wiki_data():
    page_filenames = os.listdir(DATA_DIR)
    pages = []
    for filename in page_filenames:
        with open(os.path.join(DATA_DIR, filename), "r") as f:
            content = f.read()
            title = filename.replace(".md", "")
            pages.append(dict(title=title, id=title_to_id(title), content=content))
    return pages


if __name__ == "__main__":
    pages = get_wiki_data()
    dataset = Dataset.from_list(pages)
    dataset.push_to_hub("willcb/rare-wiki-pages")
