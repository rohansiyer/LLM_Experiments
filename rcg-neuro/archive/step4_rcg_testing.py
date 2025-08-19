"""A testbench to test incoherence detection and mitigation in LLMs using RCG-Neuro.
Full stress accumulation fidelity with CPU storage for stress tensors.
"""
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
import bitsandbytes.functional as bnb_functional
import bitsandbytes as bnb

# Phase 4 hyperparameters
COLLAPSE_THRESHOLD = 1.0    # Current trigger level
SNAP_THRESHOLD = 5.0        # For radical intervention (not implemented yet)
DELTA_PERCENTAGE = 0.01     # Start conservative: 1% weight transfer

def set_module_by_name(model, name, module):
    """Helper function to replace a module deep in the model hierarchy."""
    path = name.split('.')
    parent = model
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], module)

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
    """Creates stress field with CORRECT LOGICAL SHAPES on CPU."""
    print("Creating l-field with CORRECT LOGICAL SHAPES on CPU...")
    l_field = {}
    total_memory_mb = 0
    
    for name, param in model.named_parameters():
        if param.dtype == torch.uint8 and 'weight' in name:
            # Determine correct logical shape
            if 'q_proj' in name or 'o_proj' in name:
                shape = (4096, 4096)
            elif 'k_proj' in name or 'v_proj' in name:
                shape = (1024, 4096)
            elif 'gate_proj' in name or 'up_proj' in name:
                shape = (14336, 4096)
            elif 'down_proj' in name:
                shape = (4096, 14336)
            else:
                shape = param.shape
            
            # Create stress tensor on CPU with float32 precision
            l_field[name] = torch.zeros(shape, dtype=torch.float32, device='cpu')
            
            # Calculate memory usage
            memory_mb = (l_field[name].numel() * 4) / (1024 * 1024)  # 4 bytes per float32
            total_memory_mb += memory_mb
            
            print(f"  Created stress tensor for '{name}': {shape} -> {memory_mb:.1f} MB (CPU)")
        else:
            # For non-quantized params, we don't track stress
            pass
    
    print(f"l-field created with {len(l_field)} matrices.")
    print(f"Total CPU memory for stress tensors: {total_memory_mb:.1f} MB")
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
    
    def _attention_hook(self, layer_name):
        """Hook function to capture attention weights."""
        def hook(module, input, output):
            try:
                # Handle different output formats
                if isinstance(output, tuple):
                    if len(output) >= 2 and output[1] is not None:
                        attn_weights = output[1]
                    elif len(output) >= 3 and output[2] is not None:
                        attn_weights = output[2]
                    else:
                        return
                elif isinstance(output, torch.Tensor):
                    return
                else:
                    return
                
                # Validate attention weights tensor
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    if attn_weights.dim() >= 3:
                        # Calculate entropy of attention distribution
                        attn_probs = torch.softmax(attn_weights.float(), dim=-1)
                        attn_probs = torch.clamp(attn_probs, min=1e-8, max=1.0)
                        entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
                        
                        if torch.isfinite(entropy).all():
                            avg_entropy = entropy.mean().item()
                            self.attention_entropies.append(avg_entropy)
                            
                            # Store indices of worst-performing heads (top 25% by entropy)
                            if attn_weights.dim() >= 3:
                                head_entropies = []
                                for head_idx in range(attn_weights.shape[1]):
                                    head_attn = attn_weights[0, head_idx, -1, :]
                                    head_probs = torch.softmax(head_attn.float(), dim=-1)
                                    head_entropy = -torch.sum(head_probs * torch.log(head_probs + 1e-8))
                                    head_entropies.append(head_entropy.item())
                                
                                k = max(1, int(len(head_entropies) * 0.25))
                                worst_heads = torch.tensor(head_entropies).topk(k).indices
                                self.activation_traces[layer_name] = worst_heads.detach().cpu()  # Store on CPU
                
            except Exception as e:
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
                    
                    # Store indices of top 5% most active neurons
                    activations = output[0, -1, :]
                    k = max(1, int(activations.numel() * 0.05))
                    _, top_k_indices = torch.topk(activations.abs(), k)
                    self.activation_traces[layer_name] = top_k_indices.detach().cpu()  # Store on CPU
                    
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
        
        memory_bytes = total_elements * 4
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'total_elements': total_elements,
            'memory_mb': memory_mb,
            'num_traces': len(self.activation_traces)
        }
    
    def calculate_from_scores(self, scores):
        """Calculates entropy directly from pipeline scores."""
        if not scores or not isinstance(scores, (list, tuple)):
            return 0.0
        
        total_entropy = 0.0
        num_tokens = 0
        
        try:
            for token_scores in scores:
                if not isinstance(token_scores, torch.Tensor) or token_scores.numel() == 0:
                    continue
                    
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
        
        for n in range(2, 5):
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if count > 1:
                    repetition_score += (count - 1) * n
        
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

# Phase 3: Stress Accumulation System - CPU Operations
def apply_stress(l_field, model, activation_traces, stress_score, output_file=None):
    """Apply stress to l_field based on activation indices - CPU operations for original RCG fidelity."""
    if stress_score <= 0 or not activation_traces:
        return
    
    log_msg = f"Applying stress score {stress_score:.3f} to CPU l_field using original RCG logic..."
    print(log_msg)
    if output_file:
        output_file.write(f"  {log_msg}\n")
    
    total_stress_applied = 0.0
    layers_affected = 0
    
    for layer_name, indices in activation_traces.items():
        try:
            if 'self_attn' in layer_name:
                base_layer = layer_name.replace('.self_attn', '')
                
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    weight_name = f"{base_layer}.self_attn.{proj}.weight"
                    if weight_name in l_field and isinstance(indices, torch.Tensor):
                        stress_tensor = l_field[weight_name]  # CPU tensor
                        weight_shape = stress_tensor.shape
                        
                        if len(weight_shape) >= 2 and indices.numel() > 0:
                            stress_to_apply = stress_score * 0.1
                            num_heads = 32
                            head_size = weight_shape[0] // num_heads if weight_shape[0] >= num_heads else 1
                            
                            # Apply stress to problematic attention heads - original logic on CPU
                            for head_idx in indices:
                                if head_idx.item() < num_heads:
                                    start_idx = head_idx.item() * head_size
                                    end_idx = min(start_idx + head_size, weight_shape[0])
                                    
                                    head_stress = stress_to_apply / indices.numel()
                                    # CPU tensor slice operation - much faster than sparse dict
                                    stress_tensor[start_idx:end_idx, :] += head_stress
                                    total_stress_applied += head_stress * (end_idx - start_idx) * weight_shape[1]
                            
                            layers_affected += 1
                        
            elif 'mlp' in layer_name:
                base_layer = layer_name.replace('.mlp', '')
                
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    weight_name = f"{base_layer}.mlp.{proj}.weight"
                    if weight_name in l_field and isinstance(indices, torch.Tensor):
                        stress_tensor = l_field[weight_name]  # CPU tensor
                        weight_shape = stress_tensor.shape
                        
                        if len(weight_shape) >= 2 and indices.numel() > 0:
                            stress_to_apply = stress_score * 0.05
                            
                            # Apply stress to problematic neurons - original logic on CPU
                            for neuron_idx in indices:
                                if neuron_idx.item() < weight_shape[1]:
                                    neuron_stress = stress_to_apply / indices.numel()
                                    # CPU tensor slice operation
                                    stress_tensor[:, neuron_idx.item()] += neuron_stress
                                    total_stress_applied += neuron_stress * weight_shape[0]
                            
                            layers_affected += 1
                            
        except Exception as e:
            print(f"Error applying stress to {layer_name}: {e}")
    
    summary_msg = f"  Applied RCG stress to {layers_affected} weight matrices, total stress: {total_stress_applied:.2f}"
    print(summary_msg)
    if output_file:
        output_file.write(f"  {summary_msg}\n")

def get_stress_statistics(l_field, top_k=5, output_file=None):
    """Get statistics about stress distribution in CPU l_field."""
    stress_levels = {}
    
    for name, stress_tensor in l_field.items():
        if stress_tensor.numel() > 0:
            # CPU operations
            max_stress = torch.max(stress_tensor).item()
            mean_stress = torch.mean(stress_tensor).item()
            total_stress = torch.sum(stress_tensor).item()
            
            stress_levels[name] = {
                'max': max_stress,
                'mean': mean_stress,
                'total': total_stress
            }
    
    if not stress_levels:
        print("No stress accumulated yet.")
        return []
    
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
    """Check if any weights exceed collapse threshold in CPU l_field."""
    collapse_candidates = []
    
    for name, stress_tensor in l_field.items():
        if stress_tensor.numel() > 0:
            max_stress = torch.max(stress_tensor).item()
            if max_stress >= threshold:
                # Find the exact location of maximum stress
                max_indices = torch.unravel_index(torch.argmax(stress_tensor), stress_tensor.shape)
                mean_stress = torch.mean(stress_tensor).item()
                
                collapse_candidates.append({
                    'name': name,
                    'max_stress': max_stress,
                    'indices': max_indices,
                    'mean_stress': mean_stress
                })
    
    if collapse_candidates:
        collapse_candidates.sort(key=lambda x: x['max_stress'], reverse=True)
        
        print(f"\n🚨 COLLAPSE THRESHOLD EXCEEDED! {len(collapse_candidates)} matrices need attention:")
        for i, candidate in enumerate(collapse_candidates[:3]):
            print(f"  {i+1}. {candidate['name']}: max_stress={candidate['max_stress']:.4f} at {candidate['indices']}")
        
        return True, collapse_candidates
    
    return False, []

def perform_collapse(l_field, model, collapse_candidates, output_file=None):
    """Perform weight transfer with conservation - original RCG logic."""
    msg = f"!!! COLLAPSE TRIGGERED !!! Processing {len(collapse_candidates)} candidates"
    print(msg)
    if output_file:
        output_file.write(f"{msg}\n")
    
    if not collapse_candidates:
        return
    
    # Target selection - highest stress candidate
    candidate = collapse_candidates[0]
    name = candidate['name']
    max_stress = candidate['max_stress']
    
    # Get stress tensor from CPU
    stress_tensor = l_field[name]
    logical_shape = stress_tensor.shape
    
    # Find max stress location
    flat_index = torch.argmax(stress_tensor)
    unraveled_indices = torch.unravel_index(flat_index, logical_shape)
    
    msg = f"Selected target: {name} with stress {max_stress:.4f} at logical indices {unraveled_indices}"
    print(msg)
    if output_file:
        output_file.write(f"  {msg}\n")
    
    # Determine intervention type
    if 'mlp' in name:
        target_neuron_idx = unraveled_indices[1]
        intervention_type = "MLP_neuron"
    elif 'attn' in name or 'self_attn' in name:
        num_heads = 32
        head_size = logical_shape[0] // num_heads
        target_head_idx = unraveled_indices[0] // head_size
        intervention_type = "attention_head"
    else:
        print(f"Unknown parameter type for {name} - skipping")
        return
    
    print(f"Intervention type: {intervention_type}")
    
    # Analyze stress in target region
    if intervention_type == "MLP_neuron":
        stress_slice = stress_tensor[:, target_neuron_idx]
    else:  # attention_head
        start_row = target_head_idx * head_size
        end_row = start_row + head_size
        stress_slice = stress_tensor[start_row:end_row, :]
    
    avg_stress = stress_slice.mean().item()
    print(f"Average stress in target region: {avg_stress:.4f}")
    
    if avg_stress > SNAP_THRESHOLD:
        print(f"Stress {avg_stress:.4f} exceeds snap threshold {SNAP_THRESHOLD} - SNAP not implemented yet")
        return
    else:
        print(f"Performing conservative reroute (stress {avg_stress:.4f} < snap threshold {SNAP_THRESHOLD})")
        execute_reroute(l_field, model, name, unraveled_indices, intervention_type, output_file)

def execute_reroute(l_field, model, param_name, stress_indices, intervention_type, output_file=None):
    """Conservative weight transfer with original RCG logic - CPU/GPU coordination."""
    msg = f"Starting reroute for {param_name} ({intervention_type})"
    print(msg)
    if output_file:
        output_file.write(f"  {msg}\n")
    
    try:
        # Phase 1: Dequantization (GPU operations)
        print("Phase 1: Dequantizing weights...")
        param_dict = dict(model.named_parameters())
        if param_name not in param_dict:
            print(f"ERROR: Parameter {param_name} not found in model")
            return
        
        param = param_dict[param_name]
        print(f"Original param shape: {param.shape}, dtype: {param.dtype}")
        
        # Store original quantization state
        original_quant_state = param.quant_state
        
        # Dequantize the 4-bit weights
        dequantized_weight = bnb_functional.dequantize_4bit(param.data, param.quant_state)
        print(f"Dequantized shape: {dequantized_weight.shape}, dtype: {dequantized_weight.dtype}")
        
        # Reshape to logical 2D shape
        stress_tensor = l_field[param_name]  # CPU tensor
        logical_shape = stress_tensor.shape
        param_fp = dequantized_weight.view(logical_shape)
        print(f"Successfully reshaped to logical dimensions: {param_fp.shape}")
        
        # Store original weight sum for conservation verification
        original_sum = torch.sum(param_fp).item()
        print(f"Original weight sum: {original_sum:.6f}")
        
        # Phase 2: Execute Reroute - Transfer stress region to GPU temporarily
        print("Phase 2: Executing conservative reroute...")
        
        # Get stress and param slices based on intervention type
        if intervention_type == "MLP_neuron":
            target_neuron_idx = stress_indices[1]
            # Transfer only the target column to GPU
            stress_slice = stress_tensor[:, target_neuron_idx].to('cuda')  # CPU -> GPU transfer
            param_slice = param_fp[:, target_neuron_idx]
            print(f"Target slice: MLP neuron column {target_neuron_idx}")
        else:  # attention_head
            num_heads = 32
            head_size = logical_shape[0] // num_heads
            target_head_idx = stress_indices[0] // head_size
            start_row = target_head_idx * head_size
            end_row = start_row + head_size
            # Transfer only the target head region to GPU
            stress_slice = stress_tensor[start_row:end_row, :].flatten().to('cuda')  # CPU -> GPU transfer
            param_slice = param_fp[start_row:end_row, :].flatten()
            print(f"Target slice: attention head {target_head_idx} (rows {start_row}:{end_row})")
        
        print(f"Working with {stress_slice.numel()} connections in target region")
        
        # Original donor/receptor selection logic
        N = max(1, min(5, stress_slice.numel() // 10))
        donor_indices = torch.topk(stress_slice, k=N).indices
        print(f"Selected {N} donor connections with highest stress")
        
        combined_score = stress_slice + torch.abs(param_slice)
        M = max(1, min(5, stress_slice.numel() // 10))
        receptor_indices = torch.topk(combined_score, k=M, largest=False).indices
        print(f"Selected {M} receptor connections with low stress+magnitude")
        
        # Transfer budget calculation
        delta_budget = 0.0
        donor_deltas = []
        
        for idx in donor_indices:
            delta = DELTA_PERCENTAGE * abs(param_slice[idx].item())
            donor_deltas.append(delta)
            delta_budget += delta
        
        receptor_delta = delta_budget / len(receptor_indices)
        print(f"Transfer budget: {delta_budget:.6f}, each receptor gets: {receptor_delta:.6f}")
        
        # Execute transfers
        if intervention_type == "MLP_neuron":
            target_neuron_idx = stress_indices[1]
            for i, donor_idx in enumerate(donor_indices):
                param_fp[donor_idx, target_neuron_idx] -= donor_deltas[i]
            
            for receptor_idx in receptor_indices:
                param_fp[receptor_idx, target_neuron_idx] += receptor_delta
        else:  # attention_head
            # Convert back to 2D coordinates within head region
            num_cols = logical_shape[1]
            target_head_idx = stress_indices[0] // head_size
            start_row = target_head_idx * head_size
            
            for i, flat_idx in enumerate(donor_indices):
                row_in_slice = flat_idx // num_cols
                col_in_slice = flat_idx % num_cols
                param_fp[start_row + row_in_slice, col_in_slice] -= donor_deltas[i]
            
            for flat_idx in receptor_indices:
                row_in_slice = flat_idx // num_cols
                col_in_slice = flat_idx % num_cols
                param_fp[start_row + row_in_slice, col_in_slice] += receptor_delta
        
        # Conservation verification
        final_sum = torch.sum(param_fp).item()
        conservation_error = abs(final_sum - original_sum)
        print(f"Final weight sum: {final_sum:.6f}, conservation error: {conservation_error:.8f}")
        
        if conservation_error > 1e-6:
            print(f"WARNING: Conservation error {conservation_error:.8f} exceeds threshold!")
            return
        
        # Phase 3: Create and load new layer
        print("Phase 3: Creating new Linear4bit layer...")
        layer_path = param_name.rsplit('.weight', 1)[0]
        old_layer = model.get_submodule(layer_path)
        
        new_layer = bnb.nn.Linear4bit(
            input_features=old_layer.in_features,
            output_features=old_layer.out_features,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type='nf4',
        ).to('cuda')
        
        # Reshape param_fp to match original quantized shape
        param_fp_reshaped = param_fp.reshape(param.shape[0], -1)
        
        # Requantize and load
        new_data, new_state = bnb_functional.quantize_4bit(param_fp_reshaped)
        
        # Copy essential metadata from original state
        new_state.shape = original_quant_state.shape
        new_state.dtype = original_quant_state.dtype
        
        with torch.no_grad():
            new_layer.weight.data.copy_(new_data)
            new_layer.weight.quant_state = new_state
        
        # Replace layer in model
        set_module_by_name(model, layer_path, new_layer)
        print(f"Successfully replaced {layer_path}")
        
        # Phase 4: Reset stress in CPU tensor
        print("Phase 4: Resetting stress in target region...")
        if intervention_type == "MLP_neuron":
            target_neuron_idx = stress_indices[1]
            stress_tensor[:, target_neuron_idx] = 0.0
            print(f"Reset stress in MLP neuron column {target_neuron_idx}")
        else:  # attention_head
            target_head_idx = stress_indices[0] // head_size
            start_row = target_head_idx * head_size
            end_row = start_row + head_size
            stress_tensor[start_row:end_row, :] = 0.0
            print(f"Reset stress in attention head {target_head_idx} (rows {start_row}:{end_row})")
        
        msg = f"Successfully completed reroute for {param_name}"
        print(msg)
        if output_file:
            output_file.write(f"    {msg}\n")
        
    except Exception as e:
        print(f"ERROR in execute_reroute for {param_name}: {e}")
        import traceback
        traceback.print_exc()

def run_experiment(model, tokenizer, text_generator, l_field):
    """Main experimental loop with RCG-Neuro incoherence detection."""
    print("\n--- Starting RCG-Neuro Experiment ---")
    
    # Initialize incoherence calculator
    incoherence_calc = IncoherenceCalculator(model, tokenizer)
    
    # Create detailed log file with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"step4_rcg_testing_{timestamp}.txt"
    
    # Define experiment parameters
    num_iterations = 50
    collapse_threshold = COLLAPSE_THRESHOLD
    correct_answers = 0
    
    with open(log_filename, "w", encoding="utf-8") as output_file:
        output_file.write("RCG-Neuro Detailed Experiment Log\n")
        output_file.write("=" * 50 + "\n")
        output_file.write(f"Timestamp: {datetime.datetime.now()}\n")
        output_file.write(f"Log file: {log_filename}\n")
        output_file.write(f"Model: {model.config._name_or_path}\n")
        output_file.write(f"Stress storage: CPU (preserves full RCG fidelity)\n")
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
            stress_score, components = incoherence_calc.calculate_total_incoherence(scores, response_text)
            
            # Check answer
            found_numbers = re.findall(r'[\d,]+', response_text)
            model_answer = None
            if found_numbers:
                try:
                    model_answer = int(found_numbers[-1].replace(',', ''))
                except ValueError:
                    model_answer = None
            
            components_str = f"[E:{components['h_entropy']:.3f} R:{components['h_repetition']:.3f} A:{components['h_attention']:.3f} D:{components['h_dynamism']:.3f}]"
            
            if model_answer == correct_answer:
                result_line = f"✅ Iteration {i+1}: {response_text.strip()[:50]}... (Correct!) Stress: {stress_score:.3f} {components_str} ({elapsed_time:.2f}s)"
                correct_answers += 1
            else:
                result_line = f"❌ Iteration {i+1}: {response_text.strip()[:50]}... (Incorrect: {correct_answer}) Stress: {stress_score:.3f} {components_str} ({elapsed_time:.2f}s)"
                
                # Apply stress to CPU tensors
                apply_stress(l_field, model, incoherence_calc.activation_traces, stress_score, output_file)
            
            print(result_line)
            output_file.write(result_line + "\n")
            
            # Write detailed breakdown
            detail_line = f"    Components: Entropy={components['h_entropy']:.4f}, Repetition={components['h_repetition']:.4f}, Attention={components['h_attention']:.4f}, Dynamism={components['h_dynamism']:.4f}"
            output_file.write(detail_line + "\n")
            
            # Add activation trace summary
            if incoherence_calc.activation_traces:
                memory_stats = incoherence_calc.get_trace_memory_usage()
                trace_summary = f"    Activation traces: {memory_stats['num_traces']} layers, {memory_stats['total_elements']} elements, {memory_stats['memory_mb']:.3f} MB"
                output_file.write(trace_summary + "\n")
            
            # Check for collapse
            needs_collapse, collapse_candidates = check_for_collapse(l_field, collapse_threshold)
            if needs_collapse:
                collapse_msg = f"!!! COLLAPSE TRIGGERED !!! ({len(collapse_candidates)} matrices)"
                print(collapse_msg)
                output_file.write(collapse_msg + "\n")
                perform_collapse(l_field, model, collapse_candidates, output_file)
        
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
