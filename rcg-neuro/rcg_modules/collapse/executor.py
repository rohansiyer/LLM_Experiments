"""CollapseExecutor class for performing weight transfer operations."""

import torch
import bitsandbytes.functional as bnb_functional
import bitsandbytes as bnb
from collections import defaultdict
from utils import set_module_by_name
import config

class CollapseExecutor:
    def __init__(self, model, stress_field):
        """Initializes the executor with access to model and stress field."""
        self.model = model
        self.stress_field = stress_field
        self.vram_manager = None  # Will be initialized in execute()
        
    def execute(self, candidates, logger):
        """Execute batched collapse operations on all candidates.
        
        This method now handles multiple candidates, grouped by parameter name
        for efficient batch processing.
        """
        if not candidates:
            return
            
        # Initialize VRAM manager if needed
        if self.vram_manager is None:
            from .vram_manager import VRAMManager
            self.vram_manager = VRAMManager(self.model, logger)
        
        # Group candidates by parameter name
        param_groups = defaultdict(list)
        for candidate in candidates:
            param_groups[candidate['name']].append(candidate)
        
        # Log grouping summary
        logger.info(f"\nProcessing {len(candidates)} candidates grouped into {len(param_groups)} parameters")
        
        # Get unique parameter names
        affected_params = list(param_groups.keys())
        
        try:
            # Phase 1: Prepare parameters using VRAM manager
            dequantized_params = self.vram_manager.prepare_for_surgery(affected_params)
            
            # Phase 2: Process each parameter group
            for param_name, group_candidates in param_groups.items():
                # Further group by group_key (neuron or attention head)
                subgroups = defaultdict(list)
                for candidate in group_candidates:
                    subgroups[candidate['group_key']].append(candidate)
                
                # Check for snap condition
                for group_key, candidates_in_group in subgroups.items():
                    if len(candidates_in_group) >= config.SNAP_CANDIDATE_COUNT_THRESHOLD:
                        logger.info(f"SNAP triggered for {group_key} with {len(candidates_in_group)} stress points")
                        self._execute_snap(
                            param_name=param_name,
                            candidates=candidates_in_group,
                            dequantized_param=dequantized_params[param_name],
                            logger=logger
                        )
                    else:
                        logger.info(f"Batched reroute for {group_key} with {len(candidates_in_group)} stress points")
                        self._execute_reroute(
                            param_name=param_name,
                            candidates=candidates_in_group,
                            dequantized_param=dequantized_params[param_name],
                            logger=logger
                        )
            
            # Phase 3: Reintegrate all modified parameters
            self.vram_manager.reintegrate_model(dequantized_params)
            
            # Phase 4: Reset stress for all processed regions
            for param_name, group_candidates in param_groups.items():
                for candidate in group_candidates:
                    self.stress_field.reset_region(param_name, candidate['indices'])
            
            logger.info("Successfully completed all collapse operations")
            
        except Exception as e:
            logger.info(f"ERROR during collapse operations: {e}")
            import traceback
            logger.info(traceback.format_exc())
            raise

    def _execute_reroute(self, param_name, candidates, dequantized_param, logger):
        """Execute conservative weight transfer for a batch of stress points."""
        try:
            # Get stress tensor for this parameter
            stress_tensor = self.stress_field.l_field[param_name]
            
            # Determine intervention type from first candidate
            intervention_type = "MLP_neuron" if 'mlp' in param_name else "attention_head"
            
            # Initialize tracking for conservation
            original_sum = torch.sum(dequantized_param).item()
            total_delta_budget = 0.0
            
            # Process each stress point in the group
            for candidate in candidates:
                indices = candidate['indices']
                
                # Get the appropriate slice based on intervention type
                if intervention_type == "MLP_neuron":
                    target_neuron_idx = indices[1]
                    stress_slice = stress_tensor[:, target_neuron_idx].to('cuda')
                    param_slice = dequantized_param[:, target_neuron_idx]
                else:  # attention_head
                    num_heads = 32
                    head_size = dequantized_param.shape[0] // num_heads
                    target_head_idx = indices[0] // head_size
                    start_row = target_head_idx * head_size
                    end_row = start_row + head_size
                    stress_slice = stress_tensor[start_row:end_row, :].flatten().to('cuda')
                    param_slice = dequantized_param[start_row:end_row, :].flatten()
                
                # Select donors and receptors
                N = max(1, min(5, stress_slice.numel() // 10))
                donor_indices = torch.topk(stress_slice, k=N).indices
                
                combined_score = stress_slice + torch.abs(param_slice)
                M = max(1, min(5, stress_slice.numel() // 10))
                receptor_indices = torch.topk(combined_score, k=M, largest=False).indices
                
                # Calculate transfer budget for this point
                point_budget = 0.0
                donor_deltas = []
                for idx in donor_indices:
                    delta = config.DELTA_PERCENTAGE * abs(param_slice[idx].item())
                    donor_deltas.append(delta)
                    point_budget += delta
                
                receptor_delta = point_budget / len(receptor_indices)
                total_delta_budget += point_budget
                
                # Execute transfers
                if intervention_type == "MLP_neuron":
                    for i, donor_idx in enumerate(donor_indices):
                        dequantized_param[donor_idx, target_neuron_idx] -= donor_deltas[i]
                    
                    for receptor_idx in receptor_indices:
                        dequantized_param[receptor_idx, target_neuron_idx] += receptor_delta
                else:
                    num_cols = dequantized_param.shape[1]
                    for i, flat_idx in enumerate(donor_indices):
                        row_in_slice = flat_idx // num_cols
                        col_in_slice = flat_idx % num_cols
                        dequantized_param[start_row + row_in_slice, col_in_slice] -= donor_deltas[i]
                    
                    for flat_idx in receptor_indices:
                        row_in_slice = flat_idx // num_cols
                        col_in_slice = flat_idx % num_cols
                        dequantized_param[start_row + row_in_slice, col_in_slice] += receptor_delta
            
            # Verify conservation
            final_sum = torch.sum(dequantized_param).item()
            conservation_error = abs(final_sum - original_sum)
            logger.info(f"Reroute complete - transferred {total_delta_budget:.6f} total weight")
            logger.info(f"Conservation error: {conservation_error:.8f}")
            
            if conservation_error > 1e-6:
                raise ValueError(f"Conservation error {conservation_error:.8f} exceeds threshold!")
            
        except Exception as e:
            logger.info(f"ERROR in execute_reroute for {param_name}: {e}")
            raise

    def _execute_snap(self, param_name, candidates, dequantized_param, logger):
        """Execute radical reconfiguration for severely stressed regions."""
        try:
            # Get stress tensor for this parameter
            stress_tensor = self.stress_field.l_field[param_name]
            
            # Determine intervention type and region bounds
            intervention_type = "MLP_neuron" if 'mlp' in param_name else "attention_head"
            
            if intervention_type == "MLP_neuron":
                # For MLP, the snap zone is the entire column for the target neuron
                target_neuron_idx = candidates[0]['indices'][1]
                snap_zone = dequantized_param[:, target_neuron_idx]
                stress_zone = stress_tensor[:, target_neuron_idx]
            else:
                # For attention, the snap zone is the entire head
                num_heads = 32
                head_size = dequantized_param.shape[0] // num_heads
                target_head_idx = candidates[0]['indices'][0] // head_size
                start_row = target_head_idx * head_size
                end_row = start_row + head_size
                snap_zone = dequantized_param[start_row:end_row, :].flatten()
                stress_zone = stress_tensor[start_row:end_row, :].flatten()
            
            # Calculate total influence (weight magnitude) in the snap zone
            delta_budget = torch.sum(torch.abs(snap_zone)).item()
            logger.info(f"Snap zone total influence: {delta_budget:.6f}")
            
            # Store original parameter sum for conservation
            original_sum = torch.sum(dequantized_param).item()
            
            # Zero out the snap zone
            if intervention_type == "MLP_neuron":
                dequantized_param[:, target_neuron_idx] = 0.0
            else:
                dequantized_param[start_row:end_row, :] = 0.0
            
            # Find stable pathways (low stress AND low current magnitude)
            if intervention_type == "MLP_neuron":
                # Exclude the snap zone column
                valid_cols = list(range(0, target_neuron_idx)) + list(range(target_neuron_idx + 1, dequantized_param.shape[1]))
                receptor_region = dequantized_param[:, valid_cols]
                stress_scores = stress_tensor[:, valid_cols]
            else:
                # Exclude the snap zone head
                valid_heads = list(range(0, target_head_idx)) + list(range(target_head_idx + 1, num_heads))
                valid_rows = []
                for head_idx in valid_heads:
                    start = head_idx * head_size
                    end = start + head_size
                    valid_rows.extend(range(start, end))
                receptor_region = dequantized_param[valid_rows, :]
                stress_scores = stress_tensor[valid_rows, :]
            
            # Calculate stability scores
            magnitude_scores = torch.abs(receptor_region)
            stability_scores = 1.0 / (stress_scores + magnitude_scores + 1e-9)
            
            # Select top receptors based on stability
            k = min(100, stability_scores.numel() // 100)  # Select top 1% of stable connections
            receptor_indices = torch.topk(stability_scores.flatten(), k=k, largest=True).indices
            
            # Calculate proportional distribution weights
            receptor_scores = stability_scores.flatten()[receptor_indices]
            distribution_weights = receptor_scores / torch.sum(receptor_scores)
            
            # Distribute the influence budget
            receptor_deltas = delta_budget * distribution_weights
            
            # Apply the redistributed weights
            if intervention_type == "MLP_neuron":
                for i, idx in enumerate(receptor_indices):
                    row = idx // len(valid_cols)
                    col = valid_cols[idx % len(valid_cols)]
                    dequantized_param[row, col] += receptor_deltas[i].item()
            else:
                for i, idx in enumerate(receptor_indices):
                    row = valid_rows[idx // receptor_region.shape[1]]
                    col = idx % receptor_region.shape[1]
                    dequantized_param[row, col] += receptor_deltas[i].item()
            
            # Verify conservation
            final_sum = torch.sum(dequantized_param).item()
            conservation_error = abs(final_sum - original_sum)
            logger.info(f"Snap complete - redistributed {delta_budget:.6f} total weight")
            logger.info(f"Conservation error: {conservation_error:.8f}")
            
            if conservation_error > 1e-6:
                raise ValueError(f"Conservation error {conservation_error:.8f} exceeds threshold!")
            
        except Exception as e:
            logger.info(f"ERROR in execute_snap for {param_name}: {e}")
            raise
