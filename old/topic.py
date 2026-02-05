# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    import re

    import json
    import torch
    from datasets import load_dataset
    from top2vec import Top2Vec

    print(sys.version)
    return Top2Vec, json, load_dataset, torch


@app.cell
def _():
    return


@app.cell
def _(torch):
    # check if cuda or mps available, if available, use one of them, otherwise use cpu

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
        #     "1"  # This is tracked as pytorch issue #98222
        # )
        print("using mps")
    else:
        device = torch.device("cpu")
        print("using cpu")
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
        return {"paragraph_text": paragraph, "summary_text": summary}
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
def _(ds):
    # Prepare documents from the loaded dataset for Top2Vec training
    # Try to identify the appropriate text column in the 'train' split
    # train_ds = ds["train"]

    # # Detect likely text columns
    # candidate_text_cols = [
    #     "paragraph_text",
    #     "text",
    #     "article",
    #     "content",
    #     "document",
    #     "news",
    #     "raw",
    #     "body",
    #     "story",
    # ]
    # available_cols = list(train_ds.features.keys())

    # text_col = None
    # for col in candidate_text_cols:
    #     if col in available_cols:
    #         text_col = col
    #         break

    # # If a clear text column isn't found, construct text by concatenating all string fields
    # def _row_to_text(row):
    #     parts = []
    #     for k, v in row.items():
    #         if isinstance(v, str):
    #             parts.append(v)
    #         elif isinstance(v, list):
    #             parts.extend([x for x in v if isinstance(x, str)])
    #     return " ".join(parts).strip()

    # if text_col:
    #     documents = train_ds[text_col]
    # else:
    #     # Build documents by concatenating all string-like fields
    #     documents = [_row_to_text(train_ds[i]) for i in range(len(train_ds))]

    documents = ds["train"]["paragraph_text"]

    # Basic cleaning: strip whitespace and remove empty entries
    documents = [
        d.strip() for d in documents if isinstance(d, str) and len(d.strip()) > 0
    ]

    len(documents)
    return (documents,)


@app.cell
def _(documents):
    documents[:5]
    return


@app.cell
def _(ds):
    ds["train"][:55]["paragraph_text"]
    return


@app.cell
def _(Top2Vec, documents):
    # Create a Contextual Top2Vec model

    top2vec_model = Top2Vec(
        documents=documents, ngram_vocab=False, contextual_top2vec=True
    )

    top2vec_model
    return (top2vec_model,)


@app.cell
def _(Top2Vec, documents):
    # Create a Contextual Top2Vec model (ngram_vocab)

    top2vec_ngram_model = Top2Vec(
        documents=documents, ngram_vocab=True, contextual_top2vec=True
    )

    top2vec_ngram_model
    return (top2vec_ngram_model,)


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
def _(top2vec_ngram_model):
    def _():
        # Inspect topics learned by the model
        num_topics = top2vec_ngram_model.get_num_topics()
        topic_sizes, topic_nums = top2vec_ngram_model.get_topic_sizes()
        top_terms_per_topic = []
        for topic_num in topic_nums:
            words, word_scores, _ = top2vec_ngram_model.get_topics(topic_num)
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
    model_path = "top2vec_indosum_contextual"
    top2vec_model.save(model_path)
    {"saved_path": model_path}
    return


@app.cell
def _(top2vec_ngram_model):
    # Save the trained model for later reuse
    ngram_model_path = "top2vec_indosum_ngram_contextual"
    top2vec_ngram_model.save(ngram_model_path)
    {"saved_path": ngram_model_path}
    return


@app.cell
def _(top2vec_ngram_model):
    top2vec_ngram_model.get_document_token_topic_assignment()[:5]
    return


@app.cell
def _(top2vec_ngram_model):
    top2vec_ngram_model.get_document_tokens()[:5]
    return


if __name__ == "__main__":
    app.run()
