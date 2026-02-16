import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import marimo as mo

    # Force Transformers to use the PyTorch backend only.
    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    return


@app.cell
def _():
    from transformers import DataCollatorForSeq2Seq, AutoTokenizer
    from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from transformers import pipeline
    from transformers import BertTokenizer, AutoModel

    # modified top2vec library
    from top2vec import Top2Vec

    # aliasing the module names
    import importlib
    import sys

    sys.modules["_top2vec"] = importlib.import_module("top2vec")
    sys.modules["_top2vec.top2vec"] = importlib.import_module("top2vec.top2vec")

    # indobart
    from indobenchmark import IndoNLGTokenizer

    return AutoModelForSeq2SeqLM, IndoNLGTokenizer, Top2Vec


@app.cell
def _(AutoModelForSeq2SeqLM):
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("indobenchmark/indobart-v2")

    model = bart_model

    model
    return


@app.cell
def _(IndoNLGTokenizer):
    indonlg_tokenizer = IndoNLGTokenizer.from_pretrained(
        "indobenchmark/indobart-v2"
    )

    indonlg_tokenizer.add_special_tokens({"additional_special_tokens": ["<tag>"]})
    tokenizer = indonlg_tokenizer

    tokenizer
    return


@app.cell
def _(Top2Vec):
    # Load C-Top2Vec model
    top2vec_model = Top2Vec.load("models/top2vec_indosum_mpnet")
    return (top2vec_model,)


@app.cell
def _():
    # def add_topic(example, idx):
    #     # if already have <tag>, return the example
    #     if "<tag>" in example["document"]:
    #         return example

    #     curr_topic = " ".join(topic_document_info["Representation"].values[idx])
    #     example["document"] = f"<tag> {curr_topic} <tag> {example['document']}"

    #     return example

    # # get the processor number and set the number of process
    # ds["train"] = ds["train"].map(
    #     add_topic, with_indices=True, num_proc=os.cpu_count()
    # )
    return


@app.cell
def _(top2vec_model):
    top2vec_model
    return


@app.cell
def _(top2vec_model):
    top2vec_model.documents
    return


@app.cell
def _():
    doc_idx = 0
    return (doc_idx,)


@app.cell
def _(doc_idx, top2vec_model):
    top2vec_model.documents[doc_idx]
    return


@app.cell
def _(top2vec_model):
    topic_dist = top2vec_model.get_document_topic_distribution()
    topic_dist
    return (topic_dist,)


@app.cell
def _(doc_idx, topic_dist):
    _dist = topic_dist[doc_idx]

    _dist
    return


@app.cell
def _(topic_dist):
    import numpy as np

    # 2. Identify Dominant Topic
    dominant_topics = np.argmax(topic_dist, axis=1)
    dominant_topics
    return (dominant_topics,)


@app.cell
def _(top2vec_model):
    # 3. Topic index mapping from contextual API
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()

    topic_nums
    return topic_nums, topic_sizes


@app.cell
def _(topic_sizes):
    topic_sizes
    return


@app.cell
def _(doc_idx, dominant_topics):
    # A. Basic Info
    topic_idx = int(dominant_topics[doc_idx])
    topic_idx
    return (topic_idx,)


@app.cell
def _(doc_idx, topic_dist, topic_idx):
    p_value = float(topic_dist[doc_idx, topic_idx])
    p_value
    return


@app.cell
def _(doc_idx, topic_dist, topic_idx, topic_nums):
    topic_num = (
        int(topic_nums[topic_idx])
        if len(topic_nums) == len(topic_dist[doc_idx])
        else int(topic_idx)
    )
    topic_num
    return


@app.cell
def _(top2vec_model):
    # # B. Get Top 10 Keywords from contextual API
    # words, word_scores, _ = top2vec_model.get_topics(topic_num)
    # top_words_list = []
    # for w in words[:10]:
    #     if isinstance(w, (list, tuple)):
    #         top_words_list.extend(list(w))
    #     elif hasattr(w, "tolist") and not isinstance(w, str):
    #         w_list = w.tolist()
    #         if isinstance(w_list, list):
    #             top_words_list.extend(w_list)
    #         else:
    #             top_words_list.append(str(w_list))
    #     else:
    #         top_words_list.append(w)
    # keywords_str = " ".join(map(str, top_words_list))

    # keywords_str

    words, word_scores, _ = top2vec_model.get_topics()
    words
    return (words,)


@app.cell
def _(topic_idx, words):
    words[topic_idx]
    return


@app.cell
def _(topic_idx, words):
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
        return out
        for w in out:
            ws = str(w)
            if ws not in seen:
                seen.add(ws)
                uniq.append(ws)
        return uniq


    fw = _flatten_words(words[topic_idx])

    len(fw)
    return


@app.cell
def _():
    # # Prepare and tokenize dataset
    # def preprocess_function(examples):
    #     model_inputs = tokenizer(examples["document"], max_length=768, truncation=True)

    #     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs


    # tokenized_ds = ds.map(preprocess_function, batched=True)

    # # example first 5 data
    # for i in range(5):
    #     print("raw: " + tokenizer.decode(tokenized_ds["train"][i]["input_ids"]))
    #     print("token: ", tokenizer.convert_ids_to_tokens(tokenized_ds["train"][i]["input_ids"]))
    #     print("tokenized: " + " ".join(map(str, tokenized_ds["train"][i]["input_ids"])))
    #     print("")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
