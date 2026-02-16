import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    # !pip install marimo datasets git+https://github.com/suasgn/Top2Vec sentence_transformers torch numba
    return


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    import re

    import json
    import torch
    from datasets import load_dataset

    # modified top2vec library
    from top2vec import Top2Vec

    print(sys.version)
    return Top2Vec, json, load_dataset, mo


@app.cell
def _():
    # # check if cuda or mps available, if available, use one of them, otherwise use cpu

    # device = torch.device("cpu")

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("using cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     # os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
    #     #     "1"  # This is tracked as pytorch issue #98222
    #     # )
    #     print("using mps")
    # else:
    #     device = torch.device("cpu")
    #     print("using cpu")
    return


@app.cell
def _():
    def _detokenize_sentence(tokens):
        if isinstance(tokens, list):
            return " ".join([t for t in tokens if isinstance(t, str)]).strip()
        if isinstance(tokens, str):
            return tokens.strip()
        return ""


    def get_full_paragraph_and_summary(data):
        paragraph_sentences = []
        summary_sentences = []

        for each_paragraph in data["paragraphs"]:
            for each_sentence in each_paragraph:
                paragraph_sentences.append(_detokenize_sentence(each_sentence))

        for each_summary in data["summary"]:
            summary_sentences.append(_detokenize_sentence(each_summary))

        paragraph = " ".join([s for s in paragraph_sentences if s]).strip()
        summary = " ".join([s for s in summary_sentences if s]).strip()
        return {"document": paragraph, "summary": summary}
    return (get_full_paragraph_and_summary,)


@app.cell
def _(get_full_paragraph_and_summary, load_dataset):
    ds = load_dataset("joshuasiagian/indosum")

    ds = ds.map(
        get_full_paragraph_and_summary, remove_columns=ds["train"].column_names
    )

    ds
    return (ds,)


@app.cell
def _(ds, json):
    # explore the first 5 data in the dataset
    print(json.dumps(ds["train"][:1], indent=4))
    return


@app.cell
def _(Top2Vec, ds):
    documents = ds["train"]["document"]

    # Basic cleaning: strip whitespace and remove empty entries
    documents = [
        d.strip() for d in documents if isinstance(d, str) and len(d.strip()) > 0
    ]

    len(documents)

    top2vec_model = Top2Vec(
        documents=documents,
        ngram_vocab=True,
        contextual_top2vec=True,
        # embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # modified top2vec. The original top2vec only supports "all-MiniLM-L6-v2" and "all-mpnet-base-v2"
        embedding_model="all-mpnet-base-v2",
    )

    top2vec_model
    return (top2vec_model,)


@app.cell
def _(top2vec_model):
    def _():
        # Inspect topics learned by the model
        num_topics = top2vec_model.get_num_topics()
        topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
        top_terms_per_topic = []
        for topic_num in topic_nums:
            words, word_scores, _ = top2vec_model.get_topics(topic_num)
            top_terms_per_topic.append(
                {
                    "topic": int(topic_num),
                    "size": int(topic_sizes[topic_nums.tolist().index(topic_num)]),
                    "top_terms": words[:10],
                    "term_scores": word_scores[:10].tolist(),
                }
            )

        # Display a compact summary
        summary = {
            "num_topics": int(num_topics),
            "largest_topics": [
                {
                    "topic": int(topic_nums[i]),
                    "size": int(topic_sizes[i]),
                    "top_terms": top_terms_per_topic[i]["top_terms"],
                }
                for i in range(min(10, len(topic_nums)))
            ],
        }
        return summary


    _()
    return


@app.cell
def _(top2vec_model):
    # Save the trained model for later reuse
    model_path = "models/top2vec_indosum_mpnet"
    top2vec_model.save(model_path)
    {"saved_path": model_path}
    return


@app.cell
def _(top2vec_model):
    # Count documents with top topic score higher/lower than 0.7
    threshold = 0.5
    high_score_count = 0
    low_score_count = 0

    docs = None
    if hasattr(top2vec_model, "documents") and top2vec_model.documents is not None:
        docs = top2vec_model.documents
    else:
        try:
            docs, _, _ = top2vec_model.get_documents(
                list(range(top2vec_model.get_num_documents()))
            )
        except Exception:
            docs = None

    if docs is None:
        raise ValueError(
            "Cannot access documents from model. Ensure keep_documents=True when training."
        )

    topic_dist = top2vec_model.get_document_topic_distribution()
    # topic_sizes, topic_nums = top2vec_model.get_topic_sizes()

    for _doc_id in range(len(docs)):
        _dist = topic_dist[_doc_id]
        top_score = _dist.max()
        if top_score > threshold:
            high_score_count += 1
        else:
            low_score_count += 1

    print(f"Documents with top topic score > {threshold}: {high_score_count}")
    print(f"Documents with top topic score <= {threshold}: {low_score_count}")
    print(f"Total documents: {len(docs)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Embedding models

    ### paraphrase-multilingual-MiniLM-L12-v2

    Documents with top topic score > 0.5: 10569
    Documents with top topic score <= 0.5: 3693
    Total documents: 14262

    Documents with top topic score > 0.7: 7432
    Documents with top topic score <= 0.7: 6830
    Total documents: 14262

    ### all-MiniLM-L6-v2

    Documents with top topic score > 0.5: 4248
    Documents with top topic score <= 0.5: 10014
    Total documents: 14262


    Documents with top topic score > 0.7: 1160
    Documents with top topic score <= 0.7: 13102
    Total documents: 14262

    ### all-mpnet-base-v2

    Documents with top topic score > 0.5: 12920
    Documents with top topic score <= 0.5: 1342
    Total documents: 14262



    Documents with top topic score > 0.7: 10813
    Documents with top topic score <= 0.7: 3449
    Total documents: 14262
    """)
    return


if __name__ == "__main__":
    app.run()
