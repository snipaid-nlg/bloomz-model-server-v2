# This file runs during container build time to get model weights built into the container

# Here: A Huggingface BLOOMZ model
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model checkpoint...")
    AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", use_cache=True)
    print("done")

    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
    print("done")

if __name__ == "__main__":
    download_model()