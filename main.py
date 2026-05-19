import sys
from test import *

def main():
    if len(sys.argv) < 2:
        print("Use command 'uv run main.py help' for usage instructions.")
        return
    
    else:
        command = sys.argv[1]
        if command == "help" and len(sys.argv) == 2:
            print("Usage:")
            print("uv run main.py test-tokenizer-encode: Test the encoding functionality of the tokenizer.")        
            print("uv run main.py test-tokenizer-decode: Test the decoding functionality of the tokenizer.")
            print("uv run main.py test-encoder: Test the Encoder module.")
            print("uv run main.py test-decoder: Test the Decoder module.")
            print("uv run main.py test-seq2seq: Test the Seq2Seq model.")
            print("uv run main.py train: Train the translation model.")
            print("uv run main.py inference: Run the translation inference loop.")
            print("uv run main.py app: Run the translation web application.")
            
        elif command == "test-tokenizer-encode" and len(sys.argv) == 2: test_encode()
        elif command == "test-tokenizer-decode" and len(sys.argv) == 2: test_decode()
        elif command == "test-encoder" and len(sys.argv) == 2: test_encoder()
        elif command == "test-decoder" and len(sys.argv) == 2: test_decoder()
        elif command == "test-seq2seq" and len(sys.argv) == 2: test_seq2seq()
        elif command == "train":
            from train import train_loop
            
            if len(sys.argv) == 3:
                option = sys.argv[2]
                if option == "-rb" or option == "--rebuild": train_loop(True)
                else: print(f"Unknown option: {option}. The 'train' command has only '-rb'/'--rebuild' option.")
            elif len(sys.argv) == 2: train_loop()
            else: print(f"Unknown command: {command}. Use 'uv run main.py help'")    
        elif command == "inference" and len(sys.argv) == 2:
            from inference import translation
            translation()
        elif command == "app" and len(sys.argv) == 2: print("Not implemented yet.")
        else: print(f"Unknown command: {command}. Use 'uv run main.py help'")
        
        
if __name__ == "__main__":
    main()