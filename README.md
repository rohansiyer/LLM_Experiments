# RCG-Neuro: Neural Plasticity Framework for LLMs

RCG-Neuro implements the Rejection-Collapse Governance (RCG) framework, enabling adaptive neuroplasticity in Large Language Models during inference. Based on theoretical work in discrete spacetime physics and universal optimization principles.

## 🧠 Core Concept

RCG-Neuro treats neural pathways like a physical network under stress:
- Incoherent computations create "stress" in specific pathways
- Stress accumulates until reaching collapse thresholds
- Two-tier intervention system (reroute and snap)
- Total weight magnitude is preserved (conservation law)

## 🔬 Key Features

### Incoherence Detection
- H_entropy: Token prediction uncertainty
- H_repetition: N-gram loop detection
- H_attention: Attention weight entropy
- H_dynamism: MLP activation variance

### Performance Optimizations
- Smart VRAM management with CPU streaming
- Batched parameter modifications
- Efficient 4-bit weight modification during inference
- No double forward passes

### Conservation-Aware Weight Modification
- Live modification of 4-bit quantized weights
- Batched reroute operations for efficiency
- Radical snap mechanism for systemic issues
- Strict preservation of total weight magnitude

## 📊 Technical Details

### Memory Efficiency
- VRAMManager for optimized parameter handling
- Parameters to be modified remain in VRAM
- Other parameters streamed to CPU
- Minimal VRAM-CPU transfers

### Weight Modification Process
1. Group stress points by parameter and region
2. Stream unaffected parameters to CPU
3. Dequantize affected parameters in VRAM
4. Process each group with appropriate intervention
5. Reintegrate modified parameters
6. Reset processed stress regions

### Two-Tier Intervention System
**Conservative Reroute**
- For isolated stress points
- Maintains anatomical grouping
- Proportional weight redistribution

**Radical Snap**
- For systemic issues
- Complete region reconfiguration
- Stability-weighted redistribution

## 🚀 Version History

### v0.9 (Current)
- Smart VRAM management system
- Batched parameter operations
- Two-tier intervention (reroute/snap)
- Enhanced logging and monitoring
- Comprehensive error handling

### v1.0 (Planned)
- Softmax-based weighting system
- Threshold calibration
- Enhanced stability measures
- Additional performance optimizations

## 📁 Project Structure

```
rcg-neuro/
├── rcg_modules/           
│   ├── collapse/         # Weight modification system
│   │   ├── __init__.py  # Module exports
│   │   ├── executor.py  # Collapse operations
│   │   └── vram_manager.py  # Memory management
│   ├── incoherence.py   # Signal detection
│   ├── model.py         # Model management
│   └── stress.py        # Stress field system
├── logs/                # Experimental results
├── config.py            # Configuration
├── run_v0_9.py         # Main experiment runner
└── utils.py            # Helper functions
```

## 📈 Results

Current system achievements:
- Efficient VRAM management with CPU streaming
- Parallel processing of stress points
- Two-tier intervention system
- Strict conservation guarantees
- Comprehensive error handling

## 🛠️ Setup

1. Clone the repository
2. Install dependencies (requirements.txt coming soon)
3. Configure HF_HOME_CACHE in config.py
4. Run experiments with run_v0_9.py

## 📝 Logging

The system generates detailed logs for:
- Experimental results
- Collapse and snap events
- Weight modifications
- Memory usage and transfers
- Conservation verification
- Performance metrics

Logs are stored in `rcg-neuro/logs/` with timestamped filenames.

## 🎯 Future Development

1. Threshold calibration
2. Fine-tune snap mechanism
3. Softmax-based weight system
4. Enhanced stability measures
5. Additional performance optimizations

## 📚 References

- RCG Framework Paper (discrete spacetime physics)
- RCG Universal Theory Paper (optimization principles)
