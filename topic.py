import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    # import sys
    # print(sys.version)
    return


@app.cell
def _():
    from top2vec import Top2Vec
    return


if __name__ == "__main__":
    app.run()
