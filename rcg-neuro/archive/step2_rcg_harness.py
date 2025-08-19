import os
os.environ['HF_HOME'] = 'D:/hf_cache'

# Standard, reliable imports
import torch
import transformers
import time
import random
import re

def load_model():
    """Loads the model using the standard, reliable Hugging Face method."""
    print("Loading model (Standard Method)...")
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
        device_map="cuda", # Use the explicit "cuda" mapping
    )
    
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("Model loaded successfully.")
    return model, tokenizer, text_generator

def create_l_field(model):
    print("Creating l-field...")
    l_field = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    print("l-field created.")
    return l_field

# Placeholders remain the same
def apply_stress(l_field, model, failed_weights):
    pass

def check_for_collapse(l_field, threshold):
    return False

def perform_collapse(l_field, model):
    print("!!! COLLAPSE TRIGGERED !!! (Placeholder)")
    pass

def run_experiment(model, tokenizer, text_generator, l_field):
    """
    The main experimental loop. This version includes the full model response in the printout.
    """
    print("\n--- Starting RCG Math Experiment (Stable Harness) ---")
    
    # Open output file
    with open("step2test.txt", "w", encoding="utf-8") as output_file:
        output_file.write("RCG Math Experiment Results\n")
        output_file.write("=" * 50 + "\n\n")
        
        num_iterations = 50
        collapse_threshold = 10.0 # An arbitrary starting value
        correct_answers = 0

        for i in range(num_iterations):
            # a. Generate a simple math problem
            a, b = random.randint(100, 999), random.randint(100, 999)
            prompt = f"What is {a} * {b}?"
            correct_answer = a * b

            messages = [{"role": "user", "content": prompt}]
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
            start_time = time.time()
            outputs = text_generator(
                messages, 
                max_new_tokens=40,
                eos_token_id=terminators, 
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            end_time = time.time()
            
            response_text = outputs[0]["generated_text"][-1]['content']
            
            # c. Check the answer programmatically
            found_numbers = re.findall(r'[\d,]+', response_text)
            
            model_answer = None
            if found_numbers:
                try:
                    # Clean the number (remove commas) and convert to integer
                    model_answer = int(found_numbers[-1].replace(',', ''))
                except ValueError:
                    model_answer = None

            elapsed_time = end_time - start_time

            # --- THIS IS THE CORRECTED PART ---
            if model_answer == correct_answer:
                # We now show the model's full response even when correct
                result_line = f"✅ Iteration {i+1}/{num_iterations}: -> '{response_text.strip()}' (Correct!) ({elapsed_time:.2f}s)"
                print(result_line)
                output_file.write(result_line + "\n")
                correct_answers += 1
            else:
                # We now show the model's full response when it's incorrect
                result_line = f"❌ Iteration {i+1}/{num_iterations}: -> '{response_text.strip()}' (Incorrect, answer is {correct_answer}) ({elapsed_time:.2f}s)"
                print(result_line)
                output_file.write(result_line + "\n")
                
                # This is where our RCG logic will go
                apply_stress(l_field, model, failed_weights=None) 
            
            # Check for collapse
            if check_for_collapse(l_field, collapse_threshold):
                collapse_msg = "!!! COLLAPSE TRIGGERED !!! (Placeholder)"
                print(collapse_msg)
                output_file.write(collapse_msg + "\n")
                perform_collapse(l_field, model)
                
        final_msg = f"\n--- Experiment Complete ---\nFinal Accuracy: {correct_answers / num_iterations * 100:.2f}%"
        print(final_msg)
        output_file.write(final_msg + "\n")

# Main Execution Block
if __name__ == "__main__":
    model, tokenizer, text_generator = load_model()
    l_field = create_l_field(model)
    run_experiment(model, tokenizer, text_generator, l_field)