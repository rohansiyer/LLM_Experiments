"""Standalone helper functions for RCG-Neuro framework."""

import logging
import datetime

def setup_logging(log_filename):
    """Configures a logger to write to both a file and the console."""
    logger = logging.getLogger('RCG_Neuro')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def format_result_line(iteration, response_text, is_correct, stress_score, components):
    """Formats a single line of output for the log."""
    components_str = f"[E:{components['h_entropy']:.3f} R:{components['h_repetition']:.3f} A:{components['h_attention']:.3f} D:{components['h_dynamism']:.3f}]"
    
    if is_correct:
        result = f"✅ Iteration {iteration+1}: {response_text.strip()[:50]}... (Correct!)"
    else:
        result = f"❌ Iteration {iteration+1}: {response_text.strip()[:50]}... (Incorrect)"
    
    return f"{result} Stress: {stress_score:.3f} {components_str}"

def set_module_by_name(model, name, module):
    """Helper function to replace a module deep in the model hierarchy."""
    path = name.split('.')
    parent = model
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], module)
