"""IncoherenceMonitor class for detecting and measuring incoherence in LLM outputs."""

import torch
import numpy as np
from collections import Counter
import math

class IncoherenceMonitor:
    def __init__(self, model):
        """Initializes the monitor, sets up state, and registers hooks."""
        self.model = model
        # Internal state - no need to manage these from the outside!
        self.hooks = []
        self.attention_entropies = []
        self.mlp_magnitudes = []
        self.activation_traces = {}
        self.running_mlp_avg = 0.0
        self.mlp_sample_count = 0
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

    def calculate(self, scores, response_text):
        """Calculates total incoherence score combining all four signals."""
        h_entropy = self.calculate_from_scores(scores)
        h_repetition = self.calculate_h_repetition(response_text)
        h_attention = self.calculate_h_attention()
        h_dynamism = self.calculate_h_dynamism()
        
        total_l = (h_entropy + h_repetition + h_attention + h_dynamism)
        
        print(f"Incoherence Components - Entropy: {h_entropy:.3f}, Repetition: {h_repetition:.3f}, "
              f"Attention: {h_attention:.3f}, Dynamism: {h_dynamism:.3f} -> Total: {total_l:.3f}")
        
        return total_l, {
            'h_entropy': h_entropy,
            'h_repetition': h_repetition,
            'h_attention': h_attention,
            'h_dynamism': h_dynamism
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

    def reset(self):
        """Clears the internal traces for the next generation."""
        self.attention_entropies.clear()
        self.mlp_magnitudes.clear()
        self.activation_traces.clear()

    def cleanup(self):
        """Removes all registered hooks from the model."""
        for hook in self.hooks:
            hook.remove()
