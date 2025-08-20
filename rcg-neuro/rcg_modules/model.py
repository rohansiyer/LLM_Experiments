"""Factory function for loading the RCG-Neuro model."""

import torch
import transformers
import config

def load_rcg_model(logger):
    """Loads the specified language model with quantization and returns model, tokenizer, and pipeline."""
    logger.info("Loading model...")
    
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.MODEL_ID)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=quantization_config,
        device_map="cuda",
        attn_implementation="eager"  # Required for attention weight capture
    )
    
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    logger.info("Model loaded successfully.")
    return model, tokenizer, text_generator
