# RCG-Neuro: Neural Plasticity Framework for LLMs

RCG-Neuro enables adaptive neuroplasticity in Large Language Models during inference.

##  Core Concept

RCG-Neuro treats neural pathways like a physical network under stress:
- Incoherent computations create "stress" in specific pathways
- Stress accumulates until reaching collapse thresholds
- Conservative weight transfers occur during collapse events
- Total weight magnitude is preserved (conservation law)

##  Key Features

### Incoherence Detection
- H_entropy: Token prediction uncertainty
- H_repetition: N-gram loop detection
- H_attention: Attention weight entropy
- H_dynamism: MLP activation variance

### Performance Optimizations
- 99% memory reduction via sparse tracing
- Index-based targeting for precise intervention
- Efficient 4-bit weight modification during inference
- No double forward passes

### Conservation-Aware Weight Modification
- Live modification of 4-bit quantized weights
- Conservative weight transfers between pathways
- Strict preservation of total weight magnitude
- Anatomically-aware stress application

##  Technical Details

### Memory Efficiency
- Stores only top 5% problematic MLP neuron indices
- Captures top 25% problematic attention head indices
- Kilobytes instead of megabytes per trace
- Real-time memory usage monitoring

### Weight Modification Process
1. Dequantize target weight matrices
2. Perform conservative weight transfers
3. Requantize with strict conservation
4. Reset local stress fields

##  Version History

### v0.9 (Current)
- Full RCG framework implementation
- Live weight modification capability
- 4-bit quantization support
- Performance optimizations
- Comprehensive logging system

### v1.0 (Planned)
- Softmax-based weighting system
- Snap threshold implementation
- Enhanced stability measures
- Additional performance optimizations

##  Project Structure

```
rcg-neuro/
├── rcg_modules/           # Core RCG components
│   ├── collapse.py       # Weight modification logic
│   ├── incoherence.py    # Signal detection
│   ├── model.py          # Model management
│   └── stress.py         # Stress field system
├── logs/                 # Experimental results
├── config.py             # Configuration
├── run_v0_9.py          # Main experiment runner
└── utils.py             # Helper functions
```

##  Results

Current system achievements:
- Successfully identifies problematic neural pathways
- Applies targeted stress to corresponding weights
- Implements conservation-aware collapse mechanism
- Maintains model stability during modifications

##  Setup

1. Clone the repository
2. Install dependencies (requirements.txt coming soon)
3. Configure HF_HOME_CACHE in config.py
4. Run experiments with run_v0_9.py

##  Logging

The system generates detailed logs for:
- Experimental results
- Collapse events
- Weight modifications
- Memory usage
- Performance metrics

Logs are stored in `rcg-neuro/logs/` with timestamped filenames.

##  Future Development

1. Threshold calibration
2. Implementation of pruning mechanism
3. Softmax-based weight system
4. Enhanced stability measures
5. Additional performance optimizations
