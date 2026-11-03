import os
import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import gradio as gr

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

PERSIST_DIR = "chroma_books"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Saved Chroma DB load ho raha hai...")
    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    documents = [d for d in documents if d.page_content.strip()]
    print("Total documents:", len(documents))

    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        db_books.add_documents(documents[i:i + batch_size])
        print(f"{min(i + batch_size, len(documents))}/{len(documents)} done")


def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    isbn_order = list(dict.fromkeys(
        int(rec.page_content.strip('"').split()[0]) for rec in recs
    ))
    rank = {isbn: i for i, isbn in enumerate(isbn_order)}

    book_recs = books[books["isbn13"].isin(isbn_order)].copy()
    book_recs["rank"] = book_recs["isbn13"].map(rank)
    book_recs = book_recs.sort_values("rank")

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    book_recs = book_recs.head(final_top_k)

    tone_column = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_column:
        book_recs = book_recs.sort_values(by=tone_column[tone], ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    if not query or not query.strip():
        return []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] if isinstance(row["description"], str) else ""
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors = row["authors"] if isinstance(row["authors"], str) else "Unknown author"
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


if __name__ == "__main__":
    dashboard.launch()