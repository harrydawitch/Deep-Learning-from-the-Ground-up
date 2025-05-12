
def tokenizer(sentence: str, max_seq= 100):
    
    sequence= list(sentence)
    

def get_vocab(txt_file: str):
    
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    unique = sorted(list(set(text)))

    return unique

def mapping(tokens, vocab_path, mode= "toidx"):
    vocabulary = get_vocab(vocab_path) # type: list
    
    if mode == "toidx":
        str2idx = {ch: i for i, ch in enumerate(vocabulary)}
        encode = [str2idx[c] for c in tokens]
        return encode
    
    elif mode == "tostr":
        idx2str = {i: ch for i, ch in enumerate(vocabulary)}
        decode = "".join([idx2str[idx] for idx in tokens])
        return decode


if __name__ == "__main__":
    vocab_path = r"Transformer\input.txt"
    
    sentence = "Chu Hoang Thien Long 123"
    tokens = tokenizer(sentence)
    
    encode = mapping(tokens, vocab_path, mode="toidx")
    decode = mapping(encode, vocab_path, mode="tostr") 
    print(encode)
    print(decode)
    
    