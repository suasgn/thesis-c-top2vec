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
    return (Top2Vec,)


@app.cell
def _(Top2Vec):
    model_path = "top2vec_indosum_contextual"
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
def _(top2vec_model):
    # Prepare data for IndoBART: Concatenate Top 10 Topic Words + Document Paragraph
    import numpy as np

    def prepare_augmented_data(model):
        # 1. Get Topic Distribution for all documents
        print("Retrieving topic distributions...")
        try:
            topic_dists = model.get_document_topic_distribution()
        except Exception as e:
            print(f"Error getting distribution: {e}")
            # Fallback (approximate with doc_top)
            topic_dists = np.zeros((len(model.documents), model.get_num_topics()))
            for i, t in enumerate(model.doc_top):
                topic_dists[i, t] = 1.0

        # 2. Identify Dominant Topic
        dominant_topics = np.argmax(topic_dists, axis=1)

        # 3. Topic index mapping from contextual API
        topic_sizes, topic_nums = model.get_topic_sizes()

        # 4. Get Document IDs (to match user's IDs)
        #    model.document_ids usually stores the IDs provided during training
        doc_ids = model.document_ids if hasattr(model, 'document_ids') else list(range(len(model.documents)))

        augmented_docs = []

        print("Generating augmented documents...")
        count = len(model.documents)

        for idx in range(count):
            # A. Basic Info
            topic_idx = int(dominant_topics[idx])
            actual_doc_id = doc_ids[idx]
            p_value = float(topic_dists[idx, topic_idx])
            topic_num = int(topic_nums[topic_idx]) if len(topic_nums) == len(topic_dists[idx]) else int(topic_idx)

            # B. Get Top 10 Keywords from contextual API
            words, word_scores, _ = model.get_topics(topic_num)
            top_words_list = []
            for w in words[:10]:
                if isinstance(w, (list, tuple)):
                    top_words_list.extend(list(w))
                elif hasattr(w, "tolist") and not isinstance(w, str):
                    w_list = w.tolist()
                    if isinstance(w_list, list):
                        top_words_list.extend(w_list)
                    else:
                        top_words_list.append(str(w_list))
                else:
                    top_words_list.append(w)
            keywords_str = " ".join(map(str, top_words_list[:10]))

            # C. Content
            doc_content = model.documents[idx]

            # D. Format
            combined_text = f"<tag> {keywords_str} <tag> {doc_content}"

            augmented_docs.append({
                "idx": idx,                 # Internal index
                "doc_id": actual_doc_id,    # User-facing ID
                "topic_id": int(topic_num),
                "topic_p": p_value,
                "topic_keywords": keywords_str,
                "original_text": doc_content,
                "augmented_text": combined_text
            })

        return augmented_docs

    augmented_data = prepare_augmented_data(top2vec_model)

    print(f"\nGenerated {len(augmented_data)} augmented examples.")

    # Display mapping for first 5 
    print("\nFirst 5 Documents:")
    print(f"{'Index':<6} {'Doc ID':<10} {'Topic':<6} {'p':<8} {'Keywords (First 3)'}")
    print("-" * 70)
    for item in augmented_data[:5]:
        kws = " ".join(item['topic_keywords'].split()[:3])
        print(f"{item['idx']:<6} {str(item['doc_id']):<10} {item['topic_id']:<6} {item['topic_p']:<8.4f} {kws}...")

    # Check specifically for docs 1, 2, 3 to compare with previous analysis
    print("\nCheck for Doc IDs 1, 2, 3:")
    target_map = {str(d['doc_id']): d for d in augmented_data if str(d['doc_id']) in ['1', '2', '3']}

    for tid in ['1', '2', '3']:
        if tid in target_map:
            d = target_map[tid]
            kws = " ".join(d['topic_keywords'].split()[:5])
            print(f"Doc {tid}: Assigned Topic {d['topic_id']} (p={d['topic_p']:.4f}) -> Keywords: {kws}...")
        else:
            print(f"Doc {tid}: Not found in augmented data")
    return


if __name__ == "__main__":
    app.run()
