"""CollapseExecutor class for performing weight transfer operations."""

import torch
import bitsandbytes.functional as bnb_functional
import bitsandbytes as bnb
from utils import set_module_by_name
import config

class CollapseExecutor:
    def __init__(self, model, stress_field):
        """Initializes the executor with access to model and stress field."""
        self.model = model
        self.stress_field = stress_field

    def execute(self, candidates, logger):
        """The main public method to perform a collapse on the highest-priority candidate."""
        if not candidates:
            return
        
        # Target selection - highest stress candidate
        candidate = candidates[0]
        name = candidate['name']
        max_stress = candidate['max_stress']
        
        # Get stress tensor from CPU
        stress_tensor = self.stress_field.l_field[name]
        logical_shape = stress_tensor.shape
        
        # Find max stress location
        flat_index = torch.argmax(stress_tensor)
        unraveled_indices = torch.unravel_index(flat_index, logical_shape)
        
        logger.info(f"Selected target: {name} with stress {max_stress:.4f} at logical indices {unraveled_indices}")
        
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
            logger.info(f"Unknown parameter type for {name} - skipping")
            return
        
        logger.info(f"Intervention type: {intervention_type}")
        
        # Analyze stress in target region
        if intervention_type == "MLP_neuron":
            stress_slice = stress_tensor[:, target_neuron_idx]
        else:  # attention_head
            start_row = target_head_idx * head_size
            end_row = start_row + head_size
            stress_slice = stress_tensor[start_row:end_row, :]
        
        avg_stress = stress_slice.mean().item()
        logger.info(f"Average stress in target region: {avg_stress:.4f}")
        
        if avg_stress > config.SNAP_THRESHOLD:
            logger.info(f"Stress {avg_stress:.4f} exceeds snap threshold {config.SNAP_THRESHOLD} - SNAP not implemented yet")
            return
        else:
            logger.info(f"Performing conservative reroute (stress {avg_stress:.4f} < snap threshold {config.SNAP_THRESHOLD})")
            self._execute_reroute(name, unraveled_indices, intervention_type, logger)

    def _execute_reroute(self, param_name, stress_indices, intervention_type, logger):
        """Conservative weight transfer with original RCG logic - CPU/GPU coordination."""
        logger.info(f"Starting reroute for {param_name} ({intervention_type})")
        
        try:
            # Phase 1: Dequantization (GPU operations)
            logger.info("Phase 1: Dequantizing weights...")
            param_dict = dict(self.model.named_parameters())
            if param_name not in param_dict:
                logger.info(f"ERROR: Parameter {param_name} not found in model")
                return
            
            param = param_dict[param_name]
            logger.info(f"Original param shape: {param.shape}, dtype: {param.dtype}")
            
            # Store original quantization state
            original_quant_state = param.quant_state
            
            # Dequantize the 4-bit weights
            dequantized_weight = bnb_functional.dequantize_4bit(param.data, param.quant_state)
            logger.info(f"Dequantized shape: {dequantized_weight.shape}, dtype: {dequantized_weight.dtype}")
            
            # Reshape to logical 2D shape
            stress_tensor = self.stress_field.l_field[param_name]  # CPU tensor
            logical_shape = stress_tensor.shape
            param_fp = dequantized_weight.view(logical_shape)
            logger.info(f"Successfully reshaped to logical dimensions: {param_fp.shape}")
            
            # Store original weight sum for conservation verification
            original_sum = torch.sum(param_fp).item()
            logger.info(f"Original weight sum: {original_sum:.6f}")
            
            # Phase 2: Execute Reroute - Transfer stress region to GPU temporarily
            logger.info("Phase 2: Executing conservative reroute...")
            
            # Get stress and param slices based on intervention type
            if intervention_type == "MLP_neuron":
                target_neuron_idx = stress_indices[1]
                # Transfer only the target column to GPU
                stress_slice = stress_tensor[:, target_neuron_idx].to('cuda')  # CPU -> GPU transfer
                param_slice = param_fp[:, target_neuron_idx]
                logger.info(f"Target slice: MLP neuron column {target_neuron_idx}")
            else:  # attention_head
                num_heads = 32
                head_size = logical_shape[0] // num_heads
                target_head_idx = stress_indices[0] // head_size
                start_row = target_head_idx * head_size
                end_row = start_row + head_size
                # Transfer only the target head region to GPU
                stress_slice = stress_tensor[start_row:end_row, :].flatten().to('cuda')  # CPU -> GPU transfer
                param_slice = param_fp[start_row:end_row, :].flatten()
                logger.info(f"Target slice: attention head {target_head_idx} (rows {start_row}:{end_row})")
            
            logger.info(f"Working with {stress_slice.numel()} connections in target region")
            
            # Original donor/receptor selection logic
            N = max(1, min(5, stress_slice.numel() // 10))
            donor_indices = torch.topk(stress_slice, k=N).indices
            logger.info(f"Selected {N} donor connections with highest stress")
            
            combined_score = stress_slice + torch.abs(param_slice)
            M = max(1, min(5, stress_slice.numel() // 10))
            receptor_indices = torch.topk(combined_score, k=M, largest=False).indices
            logger.info(f"Selected {M} receptor connections with low stress+magnitude")
            
            # Transfer budget calculation
            delta_budget = 0.0
            donor_deltas = []
            
            for idx in donor_indices:
                delta = config.DELTA_PERCENTAGE * abs(param_slice[idx].item())
                donor_deltas.append(delta)
                delta_budget += delta
            
            receptor_delta = delta_budget / len(receptor_indices)
            logger.info(f"Transfer budget: {delta_budget:.6f}, each receptor gets: {receptor_delta:.6f}")
            
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
            logger.info(f"Final weight sum: {final_sum:.6f}, conservation error: {conservation_error:.8f}")
            
            if conservation_error > 1e-6:
                logger.info(f"WARNING: Conservation error {conservation_error:.8f} exceeds threshold!")
                return
            
            # Phase 3: Create and load new layer
            logger.info("Phase 3: Creating new Linear4bit layer...")
            layer_path = param_name.rsplit('.weight', 1)[0]
            old_layer = self.model.get_submodule(layer_path)
            
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
            set_module_by_name(self.model, layer_path, new_layer)
            logger.info(f"Successfully replaced {layer_path}")
            
            # Phase 4: Reset stress in CPU tensor
            logger.info("Phase 4: Resetting stress in target region...")
            self.stress_field.reset_region(param_name, stress_indices)
            
            logger.info(f"Successfully completed reroute for {param_name}")
            
        except Exception as e:
            logger.info(f"ERROR in execute_reroute for {param_name}: {e}")
            import traceback
            logger.info(traceback.format_exc())
