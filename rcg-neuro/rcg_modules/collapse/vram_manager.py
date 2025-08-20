"""VRAMManager class for efficient memory management during weight modifications."""

import torch
import bitsandbytes.functional as bnb_functional

class VRAMManager:
    def __init__(self, model, logger):
        """Initializes the manager with access to model and logger.
        
        Args:
            model: The model whose weights will be modified
            logger: Logger instance for tracking operations
        """
        self.model = model
        self.logger = logger
        self._moved_params = {}  # Track params moved to CPU
        
    def _get_logical_shape(self, name, param):
        """Determines correct logical shape for a parameter."""
        if 'q_proj' in name or 'o_proj' in name:
            return (4096, 4096)
        elif 'k_proj' in name or 'v_proj' in name:
            return (1024, 4096)
        elif 'gate_proj' in name or 'up_proj' in name:
            return (14336, 4096)
        elif 'down_proj' in name:
            return (4096, 14336)
        else:
            return param.shape

    def prepare_for_surgery(self, affected_param_names: list):
        """Prepare model for weight modifications by optimizing VRAM usage.
        
        Parameters to be modified remain in VRAM, while others are moved to CPU.
        
        Args:
            affected_param_names: List of parameter names that will be modified
            
        Returns:
            Dictionary of dequantized parameters ready for modification
        """
        dequantized_params = {}
        moved_to_cpu_count = 0
        
        # First pass: Move unaffected parameters to CPU
        for name, param in self.model.named_parameters():
            if param.requires_grad or param.dtype != torch.uint8:
                continue  # Skip non-quantized or trainable parameters
                
            if name not in affected_param_names:
                # Move unaffected parameters to CPU to free VRAM
                self._moved_params[name] = param.data.clone()
                param.data = param.data.to('cpu')
                moved_to_cpu_count += 1
        
        self.logger.info(f"Moved {moved_to_cpu_count} unaffected parameters to CPU")
        
        # Second pass: Dequantize affected parameters (still in VRAM)
        for name in affected_param_names:
            param = dict(self.model.named_parameters())[name]
            
            # Verify parameter is still on CUDA
            if not param.data.is_cuda:
                param.data = param.data.to('cuda')
                
            # Dequantize the parameter
            dequantized_weight = bnb_functional.dequantize_4bit(param.data, param.quant_state)
            
            # Reshape to logical dimensions
            logical_shape = self._get_logical_shape(name, param)
            dequantized_params[name] = dequantized_weight.view(logical_shape)
            
            self.logger.info(f"Dequantized {name} to shape {logical_shape}")
        
        # Force CUDA synchronization and garbage collection
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        return dequantized_params

    def reintegrate_model(self, modified_params: dict):
        """Reintegrate modified parameters and restore CPU parameters.
        
        Args:
            modified_params: Dictionary of modified parameters to reintegrate
        """
        self.logger.info("Starting model reintegration...")
        
        try:
            # First pass: Reintegrate modified parameters
            for name, modified_param in modified_params.items():
                param = dict(self.model.named_parameters())[name]
                
                # Reshape back to quantized shape
                modified_param_reshaped = modified_param.reshape(param.shape[0], -1)
                
                # Store original quantization state
                original_quant_state = param.quant_state
                
                # Requantize and update
                new_data, new_state = bnb_functional.quantize_4bit(modified_param_reshaped)
                
                # Copy essential metadata from original state
                new_state.shape = original_quant_state.shape
                new_state.dtype = original_quant_state.dtype
                
                with torch.no_grad():
                    param.data.copy_(new_data)
                    param.quant_state = new_state
                
                self.logger.info(f"Successfully reintegrated {name}")
            
            # Second pass: Restore CPU parameters
            for name, cpu_data in self._moved_params.items():
                param = dict(self.model.named_parameters())[name]
                param.data = cpu_data.to('cuda')
            
            self.logger.info(f"Restored {len(self._moved_params)} parameters from CPU to CUDA")
            
            # Clear the tracking dict
            self._moved_params.clear()
            
            # Final cleanup
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.info(f"ERROR during model reintegration: {e}")
            import traceback
            self.logger.info(traceback.format_exc())
            raise
