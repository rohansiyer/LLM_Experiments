"""Quick test to examine quantized parameter structure."""

import os
os.environ['HF_HOME'] = 'D:/hf_cache'

import torch
import transformers

def test_quantized_parameters():
    """Test what we actually get from quantized model parameters."""
    print("Loading quantized model...")
    model_id = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
    
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda",
    )
    
    print("Model loaded. Examining parameters...")
    
    # Look at a few key parameters
    sample_params = []
    for name, param in model.named_parameters():
        if len(sample_params) < 5:  # Just examine first 5
            sample_params.append((name, param))
        if 'layers.0.self_attn.q_proj.weight' in name:
            sample_params.append((name, param))
            break
    
    for name, param in sample_params:
        print(f"\n--- Parameter: {name} ---")
        print(f"Type: {type(param)}")
        print(f"Dtype: {param.dtype}")
        print(f"Shape: {param.shape}")
        print(f"Device: {param.device}")
        print(f"Requires grad: {param.requires_grad}")
        print(f"Data type: {type(param.data)}")
        
        # Check if it has quantization attributes
        if hasattr(param, 'quant_state'):
            print(f"Has quant_state: {param.quant_state}")
        
        # Try to access actual data
        try:
            print(f"Data shape: {param.data.shape}")
            print(f"Data dtype: {param.data.dtype}")
            print(f"First few values: {param.data.flatten()[:5]}")
        except Exception as e:
            print(f"Error accessing param.data: {e}")

if __name__ == "__main__":
    test_quantized_parameters()