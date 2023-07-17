from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # load model
    print("loading model to CPU...")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", use_cache=True)
    print("done")

    # conditionally load model to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")

    # load tokenizer
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
    print("done")
    
    # build context to return model and tokenizer with
    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()