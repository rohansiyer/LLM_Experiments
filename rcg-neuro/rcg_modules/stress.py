"""StressField class for managing stress accumulation in model weights."""

import torch

class StressField:
    def __init__(self, model, logger):
        """Initializes the stress field with correct shapes on the CPU."""
        self.l_field = self._create_l_field(model, logger)
    
    def _create_l_field(self, model, logger):
        """Creates stress field with CORRECT LOGICAL SHAPES on CPU."""
        logger.info("Creating l-field with CORRECT LOGICAL SHAPES on CPU...")
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
                
                logger.info(f"  Created stress tensor for '{name}': {shape} -> {memory_mb:.1f} MB (CPU)")
            else:
                # For non-quantized params, we don't track stress
                pass
        
        logger.info(f"l-field created with {len(l_field)} matrices.")
        logger.info(f"Total CPU memory for stress tensors: {total_memory_mb:.1f} MB")
        return l_field

    def apply(self, activation_traces, stress_score, logger):
        """Apply stress to l_field based on activation indices."""
        if stress_score <= 0 or not activation_traces:
            return
        
        logger.info(f"Applying stress score {stress_score:.3f} to CPU l_field...")
        
        total_stress_applied = 0.0
        layers_affected = 0
        
        for layer_name, indices in activation_traces.items():
            try:
                if 'self_attn' in layer_name:
                    base_layer = layer_name.replace('.self_attn', '')
                    
                    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        weight_name = f"{base_layer}.self_attn.{proj}.weight"
                        if weight_name in self.l_field and isinstance(indices, torch.Tensor):
                            stress_tensor = self.l_field[weight_name]  # CPU tensor
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
                        if weight_name in self.l_field and isinstance(indices, torch.Tensor):
                            stress_tensor = self.l_field[weight_name]  # CPU tensor
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
                logger.info(f"Error applying stress to {layer_name}: {e}")
        
        logger.info(f"Applied RCG stress to {layers_affected} weight matrices, total stress: {total_stress_applied:.2f}")

    def get_statistics(self, logger, top_k=5):
        """Get statistics about stress distribution in CPU l_field."""
        stress_levels = {}
        
        for name, stress_tensor in self.l_field.items():
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
            logger.info("No stress accumulated yet.")
            return []
        
        # Sort by maximum stress
        sorted_stress = sorted(stress_levels.items(), key=lambda x: x[1]['max'], reverse=True)
        
        logger.info(f"\nTop {top_k} most stressed weight matrices:")
        for i, (name, stats) in enumerate(sorted_stress[:top_k]):
            logger.info(f"  {i+1}. {name}: max={stats['max']:.4f}, mean={stats['mean']:.6f}, total={stats['total']:.2f}")
        
        # Overall statistics
        total_matrices = len([s for s in stress_levels.values() if s['max'] > 0])
        max_overall = max([s['max'] for s in stress_levels.values()]) if stress_levels else 0
        logger.info(f"  Overall: {total_matrices} matrices with stress, max={max_overall:.4f}")
        
        return sorted_stress

    def check_for_collapse(self, threshold, logger):
        """Check if any weights exceed collapse threshold in CPU l_field."""
        collapse_candidates = []
        
        for name, stress_tensor in self.l_field.items():
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
            
            logger.info(f"\n🚨 COLLAPSE THRESHOLD EXCEEDED! {len(collapse_candidates)} matrices need attention:")
            for i, candidate in enumerate(collapse_candidates[:3]):
                logger.info(f"  {i+1}. {candidate['name']}: max_stress={candidate['max_stress']:.4f} at {candidate['indices']}")
        
        return collapse_candidates

    def reset_region(self, param_name, indices):
        """Reset stress in a specific region after collapse."""
        if param_name not in self.l_field:
            return
        
        stress_tensor = self.l_field[param_name]
        if 'mlp' in param_name:
            target_neuron_idx = indices[1]
            stress_tensor[:, target_neuron_idx] = 0.0
        else:  # attention_head
            num_heads = 32
            head_size = stress_tensor.shape[0] // num_heads
            target_head_idx = indices[0] // head_size
            start_row = target_head_idx * head_size
            end_row = start_row + head_size
            stress_tensor[start_row:end_row, :] = 0.0
