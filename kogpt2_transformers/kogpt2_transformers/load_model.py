from transformers import GPT2LMHeadModel
from .tokenization_kogpt2 import KoGPT2TokenizerFast

def get_kogpt2_model():
    model = GPT2LMHeadModel.from_pretrained('taeminlee/kogpt2')
    return model

def get_kogpt2_tokenizer():
    tokenizer = KoGPT2TokenizerFast.from_pretrained('taeminlee/kogpt2')
    return tokenizer
