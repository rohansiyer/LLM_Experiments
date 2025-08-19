import os
os.environ['HF_HOME'] = 'D:/hf_cache'

import torch
import transformers

def load_model():
    model_id = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
    
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda",
        attn_implementation="eager"
    )
    
    return model, tokenizer

def debug_layer_shapes(model):
    """Debug function to see actual layer shapes"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dtype == torch.uint8:
            # Get the actual layer to see its features
            layer_path = name.rsplit('.weight', 1)[0]
            try:
                layer = model.get_submodule(layer_path)
                print(f"{name}: param.shape={param.shape}, in_features={layer.in_features}, out_features={layer.out_features}")
            except:
                print(f"{name}: param.shape={param.shape}, (couldn't get layer)")

if __name__ == "__main__":
    model, tokenizer = load_model()
    debug_layer_shapes(model)