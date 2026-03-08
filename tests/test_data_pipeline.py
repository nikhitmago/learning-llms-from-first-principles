import os
import torch
from data.dataloader import create_dataloader_v1

def test_data_pipeline():
    # 1. Path to data
    # Note: data_path should be relative to the PROJECT root when running the test
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "the-verdict.txt"))
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 2. Read the text
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 3. Create the dataloader
    batch_size = 4
    max_length = 8
    stride = 8
    
    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=batch_size, 
        max_length=max_length, 
        stride=stride,
        shuffle=False
    )

    # 4. Verify a batch
    print(f"Total samples in dataset: {len(dataloader.dataset)}")
    print(f"Dataloader created with {len(dataloader)} batches (batch_size={batch_size}).")
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    for x, y in dataloader:
        print("\n--- First Batch Visualization ---")
        print("\nInput Tensor (X):\n", x)
        print("\nTarget Tensor (Y):\n", y)
        
        # Verification check
        assert y[0, 0] == x[0, 1], "Target is not shifted version of input!"
        print("\nVerification: Target values are correctly shifted by 1.")

        print("\n--- Decoded First Sample in Batch ---")
        print(f"Input (X) Text: {tokenizer.decode(x[0].tolist())}")
        print(f"Target (Y) Text: {tokenizer.decode(y[0].tolist())}")
        
        break

if __name__ == "__main__":
    test_data_pipeline()
