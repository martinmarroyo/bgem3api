from fastapi import FastAPI
from models.endpoint import EmbedRequest, EmbedResponse
from FlagEmbedding import BGEM3FlagModel

app = FastAPI()

Embedder = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)

@app.post("/embedding")
async def embed_text(text: EmbedRequest):
    """Embeds the given text using the BAAI BGE-M3 Model that can return dense, sparse, and/or ColBERT embeddings.
       To request an embedding, simply add it to your request and mark it's value as 'true'.

       Example: Request only ColBERT embeddings:
       ```json
       {
        "text" : "Some text to embed",
        "colbert" : "true"
       }
       ```

       Example: Request dense + sparse embeddings
       ```json
       {
        "text" : "Some text to embed",
        "dense": "true",
        "sparse": "true"
       }
       ```

       You can mix and match embedding type requests as you like. For more information on the model, visit:
       [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
    """
    # Get embedding type request
    encode_opts = {
        "return_dense": text.dense,
        "return_sparse": text.sparse,
        "return_colbert_vecs": text.colbert
    }
    embeddings = Embedder.encode(text.text, **encode_opts)
    response = EmbedResponse.from_dict({
        "text": text.text,
        "dense": embeddings.get("dense_vecs"),
        "sparse": embeddings.get("lexical_weights"),
        "colbert": embeddings.get("colbert_vecs")
    })
    return response.model_dump_json()