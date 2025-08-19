"""A testbench to test incoherence detection and stress accumulation in LLMs using RCG-Neuro."""

import os
os.environ['HF_HOME'] = 'D:/hf_cache'

import torch
import transformers
import time
import random
import re
import numpy as np
from collections import defaultdict, Counter
import math

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
        device_map="cuda",
        attn_implementation="eager"  # Required for attention weight capture
    )
    
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("Model loaded successfully.")
    return model, tokenizer, text_generator

def create_l_field(model):
    """Creates stress field mirroring model weights with proper float32 dtype."""
    print("Creating l-field...")
    l_field = {}
    
    for name, param in model.named_parameters():
        # Create stress tensor with float32 dtype regardless of original parameter type
        # This avoids issues with quantized weights (int4, int8, etc.)
        l_field[name] = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
    
    print(f"l-field created with {len(l_field)} matrices.")
    return l_field

class IncoherenceCalculator:
    """Calculates all four incoherence signals during LLM generation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_entropies = []
        self.mlp_magnitudes = []
        self.activation_traces = {}
        self.running_mlp_avg = 0.0
        self.mlp_sample_count = 0
        
        # Hook storage
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks for attention and MLP layers."""
        print("Setting up model hooks...")
        
        attention_count = 0
        mlp_count = 0
        
        # Hook attention layers
        for name, module in self.model.named_modules():
            if 'self_attn' in name and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
                attention_count += 1
            elif ('mlp' in name or 'feed_forward' in name) and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._mlp_hook(name))
                self.hooks.append(hook)
                mlp_count += 1
        
        print(f"Registered {len(self.hooks)} hooks: {attention_count} attention, {mlp_count} MLP")
        
        # Debug: print first few module names to check structure
        print("Sample module names:")
        for i, (name, _) in enumerate(self.model.named_modules()):
            if i < 10:
                print(f"  {name}")
            if i >= 10:
                print("  ...")
                break
    
    def _attention_hook(self, layer_name):
        """Hook function to capture attention weights."""
        def hook(module, input, output):
            try:
                # Debug: Check what we're getting
                if hasattr(module, '__class__'):
                    module_type = module.__class__.__name__
                else:
                    module_type = str(type(module))
                
                # Handle different output formats
                if isinstance(output, tuple):
                    # Standard transformer: (hidden_states, attention_weights, ...)
                    if len(output) >= 2 and output[1] is not None:
                        attn_weights = output[1]
                    elif len(output) >= 3 and output[2] is not None:
                        attn_weights = output[2]  # Some models put attn in third position
                    else:
                        return  # No attention weights available
                elif isinstance(output, torch.Tensor):
                    # Only hidden states, no attention weights exposed
                    return
                else:
                    return
                
                # Validate attention weights tensor
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    if attn_weights.dim() >= 3:  # Should be [batch, heads, seq, seq] or similar
                        # Calculate entropy of attention distribution
                        attn_probs = torch.softmax(attn_weights.float(), dim=-1)
                        attn_probs = torch.clamp(attn_probs, min=1e-8, max=1.0)
                        entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
                        
                        if torch.isfinite(entropy).all():
                            avg_entropy = entropy.mean().item()
                            self.attention_entropies.append(avg_entropy)
                            
                            # --- NEW LIGHTWEIGHT TRACE ---
                            # Store indices of worst-performing heads (top 25% by entropy)
                            if attn_weights.dim() >= 3:
                                head_entropies = []
                                for head_idx in range(attn_weights.shape[1]):  # num_heads
                                    head_attn = attn_weights[0, head_idx, -1, :]  # Last token attention
                                    head_probs = torch.softmax(head_attn.float(), dim=-1)
                                    head_entropy = -torch.sum(head_probs * torch.log(head_probs + 1e-8))
                                    head_entropies.append(head_entropy.item())
                                
                                # Store indices of worst-performing heads (top 25%)
                                k = max(1, int(len(head_entropies) * 0.25))
                                worst_heads = torch.tensor(head_entropies).topk(k).indices
                                self.activation_traces[layer_name] = worst_heads.detach()
                            # --- END NEW TRACE ---
                            
                            # Debug: Print first successful capture
                            if len(self.attention_entropies) == 1:
                                print(f"First attention capture: {layer_name}, shape: {attn_weights.shape}, entropy: {avg_entropy:.3f}")
                
            except Exception as e:
                # Only print first few errors to avoid spam
                if len(self.attention_entropies) == 0 and 'debug_attn_error_printed' not in self.__dict__:
                    print(f"Attention hook error in {layer_name}: {e}")
                    self.debug_attn_error_printed = True
        return hook
    
    def _mlp_hook(self, layer_name):
        """Hook function to capture MLP activation magnitudes."""
        def hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    # Calculate L2 norm of activations for H_dynamism
                    magnitude = torch.norm(output, p=2, dim=-1).mean().item()
                    self.mlp_magnitudes.append(magnitude)
                    
                    # Update running average for dynamism calculation
                    self.mlp_sample_count += 1
                    self.running_mlp_avg += (magnitude - self.running_mlp_avg) / self.mlp_sample_count
                    
                    # --- NEW LIGHTWEIGHT TRACE ---
                    # We only care about the last token's activations
                    activations = output[0, -1, :]  # Shape: [hidden_dim]
                    
                    # Store indices of top 5% most active neurons (much smaller!)
                    k = max(1, int(activations.numel() * 0.05))
                    _, top_k_indices = torch.topk(activations.abs(), k)
                    self.activation_traces[layer_name] = top_k_indices.detach()
                    # --- END NEW TRACE ---
                    
            except Exception as e:
                pass
        return hook
    
    def reset_traces(self):
        """Clear traces for new generation."""
        self.attention_entropies = []
        self.mlp_magnitudes = []
        self.activation_traces = {}
    
    def get_trace_memory_usage(self):
        """Quick check of memory usage for activation traces."""
        total_elements = 0
        for trace in self.activation_traces.values():
            if isinstance(trace, torch.Tensor):
                total_elements += trace.numel()
        
        # Assuming int32/float32 indices, 4 bytes each
        memory_bytes = total_elements * 4
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'total_elements': total_elements,
            'memory_mb': memory_mb,
            'num_traces': len(self.activation_traces)
        }
    
    def calculate_from_scores(self, scores):
        """Calculates entropy directly from pipeline scores - no double forward pass!"""
        if not scores or not isinstance(scores, (list, tuple)):
            return 0.0
        
        total_entropy = 0.0
        num_tokens = 0
        
        try:
            for token_scores in scores:
                if not isinstance(token_scores, torch.Tensor) or token_scores.numel() == 0:
                    continue
                    
                # Handle different score formats (some pipelines return batched scores)
                if token_scores.dim() > 1:
                    token_scores = token_scores.squeeze(0)
                    
                probs = torch.softmax(token_scores.float(), dim=-1)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                
                entropy = -torch.sum(probs * torch.log(probs)).item()
                if math.isfinite(entropy):
                    total_entropy += entropy
                    num_tokens += 1
            
            return total_entropy / num_tokens if num_tokens > 0 else 0.0
            
        except Exception as e:
            print(f"H_entropy calculation from scores failed: {e}")
            return 0.0
    
    def calculate_h_repetition(self, response_text):
        """Calculate repetition score (H_repetition)."""
        words = response_text.lower().split()
        if len(words) < 4:
            return 0.0
        
        repetition_score = 0.0
        
        # Check for 2-gram, 3-gram, and 4-gram repetitions
        for n in range(2, 5):
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            
            # Score based on repeated n-grams
            for ngram, count in ngram_counts.items():
                if count > 1:
                    repetition_score += (count - 1) * n  # Higher penalty for longer repeated sequences
        
        # Normalize by text length
        return repetition_score / max(len(words), 1)
    
    def calculate_h_attention(self):
        """Calculate attention entropy average (H_attention)."""
        if not self.attention_entropies:
            return 0.0
        return np.mean(self.attention_entropies)
    
    def calculate_h_dynamism(self):
        """Calculate MLP activation dynamism (H_dynamism)."""
        if not self.mlp_magnitudes or self.running_mlp_avg == 0:
            return 0.0
        
        # Calculate variance from running average
        dynamism = 0.0
        for magnitude in self.mlp_magnitudes:
            spike = abs(magnitude - self.running_mlp_avg) / self.running_mlp_avg
            dynamism += spike
        
        return dynamism / len(self.mlp_magnitudes)
    
    def calculate_total_incoherence(self, scores, response_text, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        """Calculate total incoherence score combining all four signals."""
        h_entropy = self.calculate_from_scores(scores)
        h_repetition = self.calculate_h_repetition(response_text)
        h_attention = self.calculate_h_attention()
        h_dynamism = self.calculate_h_dynamism()
        
        total_l = (alpha * h_entropy + 
                  beta * h_repetition + 
                  gamma * h_attention + 
                  delta * h_dynamism)
        
        print(f"Incoherence Components - Entropy: {h_entropy:.3f}, Repetition: {h_repetition:.3f}, "
              f"Attention: {h_attention:.3f}, Dynamism: {h_dynamism:.3f} -> Total: {total_l:.3f}")
        
        return total_l, {
            'h_entropy': h_entropy,
            'h_repetition': h_repetition,
            'h_attention': h_attention,
            'h_dynamism': h_dynamism
        }
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()

# Phase 3: Stress Accumulation System
def apply_stress(l_field, model, activation_traces, stress_score, output_file=None):
    """Apply stress to l_field based on lightweight activation indices."""
    if stress_score <= 0 or not activation_traces:
        return
    
    log_msg = f"Applying stress score {stress_score:.3f} to l_field using index-based targeting..."
    print(log_msg)
    if output_file:
        output_file.write(f"  {log_msg}\n")
    
    # Track how much stress we apply
    total_stress_applied = 0.0
    layers_affected = 0
    
    for layer_name, indices in activation_traces.items():
        try:
            if 'self_attn' in layer_name:
                # Map attention indices to attention weight matrices
                base_layer = layer_name.replace('.self_attn', '')
                
                # Apply stress to attention projection weights (q, k, v, o)
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    weight_name = f"{base_layer}.self_attn.{proj}.weight"
                    if weight_name in l_field and isinstance(indices, torch.Tensor):
                        weight_shape = l_field[weight_name].shape
                        
                        if len(weight_shape) >= 2 and indices.numel() > 0:
                            # Apply concentrated stress to problematic attention heads
                            stress_to_apply = stress_score * 0.1  # Base scale factor
                            
                            # Get total number of heads for this layer
                            num_heads = 32  # Standard for Llama models
                            head_size = weight_shape[0] // num_heads if weight_shape[0] >= num_heads else 1
                            
                            # Apply stress to weight regions corresponding to problematic heads
                            for head_idx in indices:
                                if head_idx.item() < num_heads:
                                    start_idx = head_idx.item() * head_size
                                    end_idx = min(start_idx + head_size, weight_shape[0])
                                    
                                    # Apply concentrated stress to this head's weights
                                    head_stress = stress_to_apply / indices.numel()  # Distribute stress
                                    l_field[weight_name][start_idx:end_idx, :] += head_stress
                                    total_stress_applied += head_stress * (end_idx - start_idx) * weight_shape[1]
                            
                            layers_affected += 1
                        
            elif 'mlp' in layer_name:
                # Map MLP neuron indices to MLP weight matrices
                base_layer = layer_name.replace('.mlp', '')
                
                # Apply stress to MLP weights (gate, up, down projections)
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    weight_name = f"{base_layer}.mlp.{proj}.weight"
                    if weight_name in l_field and isinstance(indices, torch.Tensor):
                        weight_shape = l_field[weight_name].shape
                        
                        if len(weight_shape) >= 2 and indices.numel() > 0:
                            # Apply stress to specific neuron connections
                            stress_to_apply = stress_score * 0.05  # Smaller scale for MLP
                            
                            # Apply stress to weight connections for top-k active neurons
                            for neuron_idx in indices:
                                if neuron_idx.item() < weight_shape[1]:  # Check bounds
                                    # Apply stress to all weights connecting to this neuron
                                    neuron_stress = stress_to_apply / indices.numel()
                                    l_field[weight_name][:, neuron_idx.item()] += neuron_stress
                                    total_stress_applied += neuron_stress * weight_shape[0]
                            
                            layers_affected += 1
                            
        except Exception as e:
            print(f"Error applying stress to {layer_name}: {e}")
    
    summary_msg = f"  Applied index-based stress to {layers_affected} weight matrices, total stress: {total_stress_applied:.2f}"
    print(summary_msg)
    if output_file:
        output_file.write(f"  {summary_msg}\n")

def get_stress_statistics(l_field, top_k=5, output_file=None):
    """Get statistics about stress distribution in l_field."""
    stress_levels = {}
    
    for name, stress_tensor in l_field.items():
        if stress_tensor.numel() > 0:
            max_stress = torch.max(stress_tensor).item()
            mean_stress = torch.mean(stress_tensor).item()
            stress_levels[name] = {
                'max': max_stress,
                'mean': mean_stress,
                'total': torch.sum(stress_tensor).item()
            }
    
    # Sort by maximum stress
    sorted_stress = sorted(stress_levels.items(), key=lambda x: x[1]['max'], reverse=True)
    
    stats_header = f"\nTop {top_k} most stressed weight matrices:"
    print(stats_header)
    if output_file:
        output_file.write(f"{stats_header}\n")
        
    for i, (name, stats) in enumerate(sorted_stress[:top_k]):
        stats_line = f"  {i+1}. {name}: max={stats['max']:.4f}, mean={stats['mean']:.6f}, total={stats['total']:.2f}"
        print(stats_line)
        if output_file:
            output_file.write(f"{stats_line}\n")
    
    # Overall statistics
    total_matrices = len([s for s in stress_levels.values() if s['max'] > 0])
    max_overall = max([s['max'] for s in stress_levels.values()]) if stress_levels else 0
    overall_line = f"  Overall: {total_matrices} matrices with stress, max={max_overall:.4f}"
    print(overall_line)
    if output_file:
        output_file.write(f"{overall_line}\n\n")
    
    return sorted_stress

def check_for_collapse(l_field, threshold=5.0):
    """Check if any weights exceed collapse threshold."""
    collapse_candidates = []
    
    for name, stress_tensor in l_field.items():
        if stress_tensor.numel() > 0:
            max_stress = torch.max(stress_tensor).item()
            if max_stress >= threshold:
                # Find the exact location of maximum stress
                max_indices = torch.unravel_index(torch.argmax(stress_tensor), stress_tensor.shape)
                collapse_candidates.append({
                    'name': name,
                    'max_stress': max_stress,
                    'indices': max_indices,
                    'mean_stress': torch.mean(stress_tensor).item()
                })
    
    if collapse_candidates:
        # Sort by stress level, highest first
        collapse_candidates.sort(key=lambda x: x['max_stress'], reverse=True)
        
        print(f"\n🚨 COLLAPSE THRESHOLD EXCEEDED! {len(collapse_candidates)} matrices need attention:")
        for i, candidate in enumerate(collapse_candidates[:3]):  # Show top 3
            print(f"  {i+1}. {candidate['name']}: max_stress={candidate['max_stress']:.4f} at {candidate['indices']}")
        
        return True, collapse_candidates
    
    return False, []

def perform_collapse(l_field, model):
    """Perform weight transfer with conservation (Phase 4)."""
    print("!!! COLLAPSE TRIGGERED !!! (Placeholder - Phase 4)")
    pass

def run_experiment(model, tokenizer, text_generator, l_field):
    """Main experimental loop with RCG-Neuro incoherence detection."""
    print("\n--- Starting RCG-Neuro Experiment ---")
    
    # Initialize incoherence calculator
    incoherence_calc = IncoherenceCalculator(model, tokenizer)
    
    # Create detailed log file with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"step3_rcg_detailed_{timestamp}.txt"
    
    # Define experiment parameters first
    num_iterations = 20  # Increased for a more robust test
    collapse_threshold = 1.0  # Lower threshold - stress is distributed across large matrices
    correct_answers = 0
    
    with open(log_filename, "w", encoding="utf-8") as output_file:
        output_file.write("RCG-Neuro Detailed Experiment Log\n")
        output_file.write("=" * 50 + "\n")
        output_file.write(f"Timestamp: {datetime.datetime.now()}\n")
        output_file.write(f"Log file: {log_filename}\n")
        output_file.write(f"Model: {model.config._name_or_path}\n")
        output_file.write(f"Attention: eager (for RCG monitoring)\n")
        output_file.write(f"Collapse threshold: {collapse_threshold}\n")
        output_file.write(f"Iterations planned: {num_iterations}\n")
        output_file.write("=" * 50 + "\n\n")
        
        for i in range(num_iterations):
            # Reset traces for new generation
            incoherence_calc.reset_traces()
            
            # Generate math problem
            a, b = random.randint(100, 999), random.randint(100, 999)
            prompt = f"What is {a} * {b}?"
            correct_answer = a * b
            
            messages = [{"role": "user", "content": prompt}]
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
            start_time = time.time()
            
            # --- NEW DIRECT GENERATION BLOCK ---
            # 1. Manually tokenize the input
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # 2. Use model.generate() directly - this respects output_scores
            generation_output = model.generate(
                input_ids,
                max_new_tokens=40,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                output_scores=True,          # This will now work
                return_dict_in_generate=True # This gives us a neat output object
            )

            # 3. Separate the generated tokens from the scores
            generated_tokens = generation_output.sequences[0]
            scores = generation_output.scores

            # 4. Decode the generated text
            # We need to slice off the input tokens to get only the response
            response_text = tokenizer.decode(generated_tokens[input_ids.shape[-1]:], skip_special_tokens=True)
            # --- END DIRECT GENERATION BLOCK ---
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate total incoherence using scores from generation
            stress_score, components = incoherence_calc.calculate_total_incoherence(scores, response_text)
            
            # Check answer
            found_numbers = re.findall(r'[\d,]+', response_text)
            model_answer = None
            if found_numbers:
                try:
                    model_answer = int(found_numbers[-1].replace(',', ''))
                except ValueError:
                    model_answer = None
            
            # Format detailed component breakdown for debugging
            components_str = f"[E:{components['h_entropy']:.3f} R:{components['h_repetition']:.3f} A:{components['h_attention']:.3f} D:{components['h_dynamism']:.3f}]"
            
            if model_answer == correct_answer:
                result_line = f"✅ Iteration {i+1}: {response_text.strip()[:50]}... (Correct!) Stress: {stress_score:.3f} {components_str} ({elapsed_time:.2f}s)"
                correct_answers += 1
            else:
                result_line = f"❌ Iteration {i+1}: {response_text.strip()[:50]}... (Incorrect: {correct_answer}) Stress: {stress_score:.3f} {components_str} ({elapsed_time:.2f}s)"
                
                # Apply stress (Phase 3)
                apply_stress(l_field, model, incoherence_calc.activation_traces, stress_score, output_file)
            
            print(result_line)
            output_file.write(result_line + "\n")
            
            # Write detailed breakdown to file for debugging
            detail_line = f"    Components: Entropy={components['h_entropy']:.4f}, Repetition={components['h_repetition']:.4f}, Attention={components['h_attention']:.4f}, Dynamism={components['h_dynamism']:.4f}"
            output_file.write(detail_line + "\n")
            
            # Add activation trace summary with memory monitoring
            if incoherence_calc.activation_traces:
                memory_stats = incoherence_calc.get_trace_memory_usage()
                trace_summary = f"    Activation traces: {memory_stats['num_traces']} layers, {memory_stats['total_elements']} elements, {memory_stats['memory_mb']:.3f} MB"
                output_file.write(trace_summary + "\n")
                
                # Sample a few traces for detailed logging
                for j, (layer_name, indices) in enumerate(list(incoherence_calc.activation_traces.items())[:3]):
                    if isinstance(indices, torch.Tensor) and indices.numel() > 0:
                        idx_stats = f"      {layer_name}: {indices.numel()} indices, max_idx={indices.max().item() if indices.numel() > 0 else -1}, min_idx={indices.min().item() if indices.numel() > 0 else -1}"
                        output_file.write(idx_stats + "\n")
            
            # Check for collapse
            needs_collapse, collapse_candidates = check_for_collapse(l_field, collapse_threshold)
            if needs_collapse:
                collapse_msg = f"!!! COLLAPSE TRIGGERED !!! ({len(collapse_candidates)} matrices)"
                print(collapse_msg)
                output_file.write(collapse_msg + "\n")
                perform_collapse(l_field, model)
        
            # Show stress statistics every few iterations
            if (i + 1) % 5 == 0:
                get_stress_statistics(l_field, output_file=output_file)
        
        final_msg = f"\n--- Experiment Complete ---\nFinal Accuracy: {correct_answers / num_iterations * 100:.2f}%"
        print(final_msg)
        output_file.write(final_msg + "\n")
    
    # Cleanup
    incoherence_calc.cleanup()

# Main execution block
if __name__ == "__main__":
    model, tokenizer, text_generator = load_model()
    l_field = create_l_field(model)
    run_experiment(model, tokenizer, text_generator, l_field)