from configs_llada_dsa import ModelConfig
from model_dsa import LLaDAModel
import torch

def test_dsa():
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=4,
        max_sequence_length=4096,
        use_dsa=True,
        index_n_heads=32,
        index_head_dim=64,
        index_topk=512,
        rope=True,
    )
    
    model = LLaDAModel(config, init_params=True)
    model.eval()
    
    x_short = torch.randint(0, config.vocab_size, (2, 128))
    out_short = model(x_short)
    print(f"✓ Short sequence: {out_short.logits.shape}")

    x_long = torch.randint(0, config.vocab_size, (2, 2048))
    out_long = model(x_long)
    print(f"✓ Long sequence: {out_long.logits.shape}")
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_dsa()