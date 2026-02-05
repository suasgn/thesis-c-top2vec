import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import os
    from huggingface_hub import hf_hub_download

    # Define the repository ID and the filename you want to download
    repo_id = "joshuasiagian/top2vec_indosum_mpnet_model"
    filename = "top2vec_indosum_mpnet"

    # Define the local directory where you want to save the file
    local_dir = "./models"
    os.makedirs(local_dir, exist_ok=True)

    # Download the file
    local_filepath = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    print(f"File downloaded to: {local_filepath}")
    return


if __name__ == "__main__":
    app.run()
