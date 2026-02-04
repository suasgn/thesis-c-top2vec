import marimo

__generated_with = "0.19.7"
app = marimo.App()


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
    return Top2Vec, re


@app.cell
def _(Top2Vec):
    model_path = "top2vec_indosum_ngram_contextual"
    top2vec_model = Top2Vec.load(model_path)
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
def _(re, top2vec_model):
    # Show top topic for the first 2 documents with their paragraphs
    num_docs_to_show = 100

    docs = None
    if hasattr(top2vec_model, "documents") and top2vec_model.documents is not None:
        docs = top2vec_model.documents
    else:
        try:
            docs, _, _ = top2vec_model.get_documents(list(range(top2vec_model.get_num_documents())))
        except Exception:
            docs = None

    if docs is None:
        raise ValueError("Cannot access documents from model. Ensure keep_documents=True when training.")

    topic_dist = top2vec_model.get_document_topic_distribution()
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()

    # Try to use model tokenization if available
    try:
        model_doc_tokens = top2vec_model.get_document_tokens()
    except Exception:
        model_doc_tokens = None

    def _flatten_words(words):
        out = []
        for w in words:
            if isinstance(w, (list, tuple)):
                out.extend(list(w))
            elif hasattr(w, "tolist") and not isinstance(w, str):
                w_list = w.tolist()
                if isinstance(w_list, list):
                    out.extend(w_list)
                else:
                    out.append(str(w_list))
            else:
                out.append(w)
        # unique preserve order
        seen = set()
        uniq = []
        for w in out:
            ws = str(w)
            if ws not in seen:
                seen.add(ws)
                uniq.append(ws)
        return uniq

    def _doc_tokens(doc_id, doc_text):
        if model_doc_tokens is not None:
            try:
                return set(str(t) for t in model_doc_tokens[doc_id])
            except Exception:
                pass
        return set(re.findall(r"[A-Za-zÀ-ÿ0-9_]+", doc_text.lower()))

    def _doc_specific_keywords(doc_id, topic_num, max_words=8):
        words, _, _ = top2vec_model.get_topics(topic_num)
        topic_words = _flatten_words(words[:50])

        doc_tokens = _doc_tokens(doc_id, docs[doc_id])
        # prefer topic words that appear in the document
        in_doc = [w for w in topic_words if str(w).lower() in doc_tokens]
        if len(in_doc) >= max_words:
            return in_doc[:max_words]
        # pad with remaining topic words
        remaining = [w for w in topic_words if w not in in_doc]
        return (in_doc + remaining)[:max_words]
        # return (topic_words)[:max_words]

    for doc_id in range(min(num_docs_to_show, len(docs))):
        # doc_text = docs[doc_id]
        dist = topic_dist[doc_id]
        top_idx = int(dist.argmax())
        topic_num = int(topic_nums[top_idx]) if len(topic_nums) == len(dist) else int(top_idx)
        # top_words = ", ".join(_doc_specific_keywords(doc_id, topic_num, max_words=8))
        print(f"Document {doc_id}")
        # print("Paragraph:")
        # print(doc_text)
        print(f"Top topic: {topic_num} (p={dist[top_idx]:.4f})")
        # print(f"Top words: {top_words}")
        print("-" * 80)
    return docs, topic_dist


@app.cell
def _(docs, topic_dist):
    # Count documents with top topic score higher/lower than 0.7
    threshold = 0.7
    high_score_count = 0
    low_score_count = 0

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


if __name__ == "__main__":
    app.run()
