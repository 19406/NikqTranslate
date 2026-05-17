from src import i2T, T2i, build_seq2seq

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

def test_seq2seq():
    model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", EMBEDDING_DIM, HIDDEN_DIM)
    
    src = i2T([en_tokenizer.encode("how are you")])
    tgt = i2T([vi_tokenizer.encode("bạn khoẻ không")])
    
    output = model(src, tgt)
    
    print("Output shape:", output.shape)
    
    predicted_ids = output.argmax(dim=-1)
    print("Predicted token IDs:", predicted_ids)
    print("Predicted sentence:", vi_tokenizer.decode(T2i(predicted_ids.reshape(-1))))