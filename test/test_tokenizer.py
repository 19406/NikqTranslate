from src.utils import build_vocab

def test_encode():    
    tokenizer = build_vocab("data/corpus.json", lang="en")
    text = input("Enter text: ")
    
    encoded = tokenizer.encode(text)

    print(encoded)
    
def test_decode():    
    tokenizer = build_vocab("data/corpus.json", lang="en")
    text = input("Enter token IDs (comma-separated):")
    token_ids = [int(x.strip()) for x in text.split(",")]
    
    decoded = tokenizer.decode(token_ids)

    print(decoded)