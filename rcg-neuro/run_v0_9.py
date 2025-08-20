"""Main entry point for RCG-Neuro experiment."""

import time
import datetime
import random
import re
import os

# Import settings and "toolboxes"
import config
from rcg_modules import model as model_loader
from rcg_modules.incoherence import IncoherenceMonitor
from rcg_modules.stress import StressField
from rcg_modules.collapse import CollapseExecutor  # Now using batched operations with VRAM management
from utils import setup_logging, format_result_line

def check_answer(response_text, correct_answer):
    """Extract and validate the model's answer."""
    found_numbers = re.findall(r'[\d,]+', response_text)
    model_answer = None
    if found_numbers:
        try:
            model_answer = int(found_numbers[-1].replace(',', ''))
        except ValueError:
            model_answer = None
    return model_answer == correct_answer

def run_experiment():
    """Main experimental loop with RCG-Neuro incoherence detection."""
    # --- 1. SETUP PHASE ---
    # Set HF cache directory
    os.environ['HF_HOME'] = config.HF_HOME_CACHE
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{config.LOG_FILE_PREFIX}_{timestamp}.txt"
    logger = setup_logging(log_filename)
    logger.info("RCG-Neuro V0.9 Experiment Starting...")
    
    # Load the model using the factory function
    model, tokenizer, _ = model_loader.load_rcg_model(logger)
    
    # Create the "toolboxes"
    stress_field = StressField(model, logger)
    incoherence_monitor = IncoherenceMonitor(model, logger)
    collapse_executor = CollapseExecutor(model, stress_field)
    
    # Log experiment configuration
    logger.info("=" * 50)
    logger.info(f"Timestamp: {datetime.datetime.now()}")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Model: {model.config._name_or_path}")
    logger.info(f"Stress storage: CPU (preserves full RCG fidelity)")
    logger.info(f"Collapse threshold: {config.COLLAPSE_THRESHOLD}")
    logger.info(f"Snap threshold: {config.SNAP_THRESHOLD}")
    logger.info(f"Snap candidate count: {config.SNAP_CANDIDATE_COUNT_THRESHOLD}")
    logger.info(f"Weight transfer %: {config.DELTA_PERCENTAGE * 100:.1f}%")
    logger.info(f"Iterations planned: {config.NUM_ITERATIONS}")
    logger.info("=" * 50 + "\n")
    
    # --- 2. MAIN LOOP ---
    correct_answers = 0
    
    for i in range(config.NUM_ITERATIONS):
        # Reset traces for new generation
        incoherence_monitor.reset()
        
        # Generate math problem
        a, b = random.randint(100, 999), random.randint(100, 999)
        prompt = f"What is {a} * {b}?"
        correct_answer = a * b
        
        messages = [{"role": "user", "content": prompt}]
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        start_time = time.time()
        
        # Direct generation with scores
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        generation_output = model.generate(
            input_ids,
            max_new_tokens=40,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_tokens = generation_output.sequences[0]
        scores = generation_output.scores
        response_text = tokenizer.decode(generated_tokens[input_ids.shape[-1]:], skip_special_tokens=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate total incoherence
        stress_score, components = incoherence_monitor.calculate(scores, response_text)
        
        # Check answer
        is_correct = check_answer(response_text, correct_answer)
        if is_correct:
            correct_answers += 1
        else:
            # Apply stress to CPU tensors
            stress_field.apply(incoherence_monitor.activation_traces, stress_score, logger)
        
        # Log the results
        result_line = format_result_line(i, response_text, is_correct, stress_score, components)
        logger.info(result_line)
        
        # Write detailed breakdown
        detail_line = f"    Components: Entropy={components['h_entropy']:.4f}, Repetition={components['h_repetition']:.4f}, Attention={components['h_attention']:.4f}, Dynamism={components['h_dynamism']:.4f}"
        logger.info(detail_line)
        
        # Add activation trace summary
        if incoherence_monitor.activation_traces:
            memory_stats = incoherence_monitor.get_trace_memory_usage()
            trace_summary = f"    Activation traces: {memory_stats['num_traces']} layers, {memory_stats['total_elements']} elements, {memory_stats['memory_mb']:.3f} MB"
            logger.info(trace_summary)
        
        # Check for collapse
        collapse_candidates = stress_field.check_for_collapse(config.COLLAPSE_THRESHOLD, logger)
        if collapse_candidates:
            unique_matrices = len(set(c['name'] for c in collapse_candidates))
            unique_groups = len(set(c['group_key'] for c in collapse_candidates))
            collapse_msg = (f"!!! COLLAPSE TRIGGERED !!! "
                          f"({len(collapse_candidates)} stress points across {unique_matrices} matrices, "
                          f"{unique_groups} target regions)")
            logger.info(collapse_msg)
            collapse_executor.execute(collapse_candidates, logger)
        
        # Show stress statistics every few iterations
        if (i + 1) % 5 == 0:
            stress_field.get_statistics(logger)
    
    # --- 3. TEARDOWN PHASE ---
    final_msg = f"\n--- Experiment Complete ---\nFinal Accuracy: {correct_answers / config.NUM_ITERATIONS * 100:.2f}%"
    logger.info(final_msg)
    
    # Cleanup
    incoherence_monitor.cleanup()

if __name__ == "__main__":
    run_experiment()
