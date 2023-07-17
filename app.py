from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deep_translator import GoogleTranslator
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
    # get model and tokeinzer from context
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    # parse out arguments from request
    prompt = request.json.get("prompt")
    document = request.json.get("document")
    task_prefix = request.json.get("task_prefix")
    params = request.json.get("params")
    
    # handle missing arguments
    if document == None:
        return Response(
            json = {"message": "No document provided"}, 
            status=500
        )

    if task_prefix == None:
        task_prefix = ""

    if prompt == None:
        return Response(
            json = {"message": "No prompt provided"}, 
            status=500
        )
    
    if params == None:
        params = {}

    # translate the document to english
    document_en = GoogleTranslator(source='auto', target='en').translate(document[:4500])
    
    # initialize pipeline
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, **params)

    # run generation pipline
    output = gen_pipe(f"{task_prefix} {document_en} {prompt}")

    # get output text
    output_text = output[0]['generated_text'].split(prompt)[1].split("</s>")[0]

    # translate output back to german
    output_text_de = GoogleTranslator(source='auto', target='de').translate(output_text)

    # return the result
    return Response(
        json = {"output": output_text_de}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()