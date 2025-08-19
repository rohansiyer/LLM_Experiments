import torch
import transformers

# --- 1. Configuration: Define the Model and Quantization Settings ---

# This is the official identifier for Llama 3.1 8B. 
# We're using a version from "unsloth" which is optimized for fine-tuning and inference
# on consumer hardware. The "bnb-4bit" part indicates it's pre-quantized for us.
model_id = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"

# This is the magic that makes the model fit in your 8GB of VRAM.
# We are telling the `transformers` library to load the model's weights not as
# 32-bit or 16-bit numbers, but as tiny 4-bit numbers.
quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # The data type for calculations
)

print("--- Configuration Defined ---")


# --- 2. Loading the Components ---

# The Tokenizer translates human-readable text into numbers (tokens) the model understands.
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# The Model itself. This is the big download.
# - `model_id`: Which model to get from the Hugging Face Hub.
# - `quantization_config`: The 4-bit loading settings we defined above.
# - `device_map="auto"`: This tells the `accelerate` library to automatically figure out
#   how to load the model. It will put as much as it can onto your GPU.
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda",
)

print("--- Tokenizer and Model Loaded ---")


# --- 3. Creating the Inference Pipeline ---

# The `pipeline` is a high-level, easy-to-use wrapper for doing inference.
# It handles all the complex steps of tokenization, generation, and decoding for you.
text_generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("--- Inference Pipeline Created ---")


# --- 4. Running a Test Inference ---

# Let's create a simple prompt to test the model.
# The "messages" format is how you interact with instruction-tuned models.
messages = [
    {"role": "user", "content": "Explain the concept of emergent phenomena in a few sentences, like you are a physics enthusiast."},
]

# The `terminators` tell the model when to stop generating text.
# `eos_token_id` is the standard "end-of-sentence" token.
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

print("\n--- Running Test Inference... ---")

# Run the pipeline!
outputs = text_generator(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Print the result
print("\n--- MODEL OUTPUT ---")
print(outputs[0]["generated_text"][-1]['content'])
print("--------------------")

print("\nStep 1 complete! If you see this message and the text above, your setup is working.")