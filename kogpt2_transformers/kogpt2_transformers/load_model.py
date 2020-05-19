from transformers import XLNetTokenizer, GPT2LMHeadModel

def get_kogpt2_model():
    model = GPT2LMHeadModel.from_pretrained('taeminlee/kogpt2')
    return model

def get_kogpt2_tokenizer():
    tokenizer = XLNetTokenizer.from_pretrained('taeminlee/kogpt2')
    return tokenizer
