# numpyGPT - JavaScript Implementation

A complete JavaScript/Node.js implementation of GPT (Generative Pre-trained Transformer) built from scratch using only basic matrix operations. This project demonstrates the inner workings of transformer architectures with explicit gradient computation for educational purposes.

## 🎯 Project Overview

This is a **complete JavaScript port** of the original Python numpyGPT implementation, providing:

- **Educational Focus**: Explicit gradient computation without autograd
- **Full Feature Parity**: 100% compatibility with Python version
- **Production Ready**: Comprehensive testing and validation
- **Modern JavaScript**: ES6 modules, TypeScript definitions, browser support
- **Interactive Demo**: Real-time training visualization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/numpyGPT.git
cd numpyGPT

# Install dependencies
npm install
```

### Basic Usage

```javascript
import { GPT } from './src/models/GPT.js';
import { Adam } from './src/optim/adam.js';
import { CharTokenizer } from './src/tokenizer/char_level.js';

// Create tokenizer and build vocabulary
const tokenizer = new CharTokenizer();
const text = "hello world this is a test";
tokenizer.buildVocab(text);

// Create model
const model = new GPT(
  tokenizer.vocabSize, // vocab_size
  64,                  // max_length
  128,                 // d_model
  4,                   // n_heads
  3,                   // n_layers
  256                  // d_ff
);

// Create optimizer
const optimizer = new Adam([model], 0.001);

// Training step
const tokens = tokenizer.encode(text);
const X = new Matrix([tokens.slice(0, -1)]);
const Y = new Matrix([tokens.slice(1)]);

const [logits, loss] = model.forward(X, Y);
model.backward();
optimizer.step();
optimizer.zeroGrad();

console.log(`Loss: ${loss}`);

// Generate text
const prompt = "hello";
const promptTokens = tokenizer.encode(prompt, false, false);
const generated = model.generate(new Matrix([promptTokens]), 20, 1.0);
const generatedText = tokenizer.decode(generated.getRow(0));
console.log(`Generated: ${generatedText}`);
```

## 📚 Features

### Core Components

- **🧠 Neural Network Modules**: Linear, Embedding, LayerNorm, Attention, FeedForward
- **🔄 Transformer Architecture**: Complete GPT implementation with causal masking
- **⚡ Optimizers**: Adam with learning rate scheduling
- **🔤 Tokenizers**: Character-level, Word-level, and BPE tokenization
- **📊 Training Utilities**: Monitoring, visualization, and logging
- **🧪 Comprehensive Testing**: Unit tests and cross-validation

### Advanced Features

- **📝 TypeScript Support**: Complete type definitions for better IDE experience
- **🌐 Browser Compatibility**: Webpack build for web deployment
- **🎮 Interactive Demo**: Real-time training visualization
- **📈 Performance Monitoring**: Benchmarking and profiling tools
- **🔄 Model I/O**: Import/export compatibility with Python version

## 🛠️ Available Scripts

### Training and Inference

```bash
# Generate training data
node datagen.js --output data/simple.txt --size 1000

# Train a model
node train.js --data data/simple.txt --epochs 10 --lr 0.001

# Generate text samples
node sample.js --model checkpoints/model.json --prompt "hello" --length 50

# Plot training curves
node plot.js --metrics out/metrics.json
```

### Testing and Validation

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:nn      # Neural network modules
npm run test:optim   # Optimizers
npm run test:tokenizer # Tokenizers

# Cross-validate with Python version
node validate_parity.js --verbose

# Performance benchmarking
node benchmark.js --verbose
```

### Browser Development

```bash
# Build for browser
npm run build:browser

# Start development server
npm run dev:browser

# Generate TypeScript definitions
npm run build:types
```

## 🏗️ Architecture

### Model Structure

```
GPT Model
├── Token Embedding (vocab_size × d_model)
├── Positional Encoding (max_len × d_model)
├── Transformer Blocks (n_layers)
│   ├── Multi-Head Attention
│   │   ├── Query/Key/Value Projections
│   │   ├── Scaled Dot-Product Attention
│   │   └── Output Projection
│   ├── Feed-Forward Network
│   │   ├── Linear (d_model → d_ff)
│   │   ├── ReLU Activation
│   │   └── Linear (d_ff → d_model)
│   └── Layer Normalization (×2)
├── Final Layer Norm
└── Language Modeling Head (d_model → vocab_size)
```

### Directory Structure

```
numpyGPT/
├── src/
│   ├── nn/
│   │   ├── modules/          # Neural network components
│   │   └── functional.js     # Activation functions and losses
│   ├── models/
│   │   └── GPT.js           # Main GPT implementation
│   ├── optim/               # Optimizers and schedulers
│   ├── tokenizer/           # Tokenization strategies
│   ├── utils/               # Training utilities and visualization
│   └── index.js             # Browser entry point
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation
├── demo/                    # Interactive web demo
├── types/                   # TypeScript definitions
└── benchmark_results/       # Performance benchmarks
```

## 🎓 Educational Value

### Learning Objectives

This implementation helps you understand:

1. **Transformer Architecture**: Complete self-attention mechanism
2. **Gradient Computation**: Manual backpropagation without autograd
3. **Optimization**: Adam optimizer with momentum and bias correction
4. **Tokenization**: Different strategies for text preprocessing
5. **Training Dynamics**: Loss curves, learning rates, and convergence

### Key Educational Features

- **Explicit Mathematics**: All operations clearly implemented
- **Comprehensive Comments**: Detailed explanations throughout
- **Progressive Complexity**: From simple modules to complete GPT
- **Cross-Language Skills**: Python concepts in JavaScript context

## 📊 Performance

### Benchmarks (Node.js v22.14.0, Linux x64)

| Model Size | Parameters | Forward Pass | Training Step | Generation |
|------------|------------|--------------|---------------|------------|
| Tiny       | 72K        | 19.1ms       | 144ms         | 15.0ms     |
| Small      | 552K       | 92.6ms       | 167ms         | 25.6ms     |
| Medium     | 3.3M       | 359ms        | 885 tok/sec   | 1,674 tok/sec |

### Memory Usage

- **Model Size**: ~5MB for medium model (3.3M parameters)
- **Training Memory**: ~2.6MB per training step
- **Peak Usage**: ~27MB during training

## 🌐 Browser Usage

### Quick Setup

```html
<!DOCTYPE html>
<html>
<head>
    <title>numpyGPT in Browser</title>
</head>
<body>
    <script src="dist/numpygpt.bundle.js"></script>
    <script>
        // Create and train a model
        const model = numpyGPT.createModel({
            vocabSize: 100,
            dModel: 64,
            nHeads: 2,
            nLayers: 2
        });
        
        const tokenizer = numpyGPT.createTokenizer('char');
        const optimizer = numpyGPT.createOptimizer(model, 'adam', 0.01);
        
        // Training and generation...
    </script>
</body>
</html>
```

### Interactive Demo

Visit the interactive demo at `demo/index.html` for:

- **Real-time Training**: Watch loss curves update live
- **Text Generation**: Interactive prompt-based generation
- **Architecture Explorer**: Detailed model visualization
- **Parameter Tuning**: Adjust hyperparameters on the fly

## 🔧 Configuration

### Model Configuration

```javascript
const config = {
  vocabSize: 1000,      // Vocabulary size
  maxLen: 128,          // Maximum sequence length
  dModel: 256,          // Model dimension
  nHeads: 8,            // Number of attention heads
  nLayers: 6,           // Number of transformer layers
  dFF: 1024            // Feed-forward dimension
};
```

### Training Configuration

```javascript
const trainingConfig = {
  batchSize: 4,         // Batch size
  lr: 3e-4,            // Learning rate
  maxIters: 1000,      // Training iterations
  evalInterval: 100,    // Evaluation frequency
  gradClip: 1.0        // Gradient clipping
};
```

## 🧪 Testing

### Test Coverage

- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline testing
- **Cross-Validation**: Python vs JavaScript parity
- **Performance Tests**: Benchmarking and profiling

### Running Tests

```bash
# All tests
npm test

# Specific modules
node tests/test_nn/test_attention_simple.js
node tests/test_nn/test_feedforward_simple.js
node tests/test_nn/test_functional_simple.js

# Validation
node validate_parity.js --tolerance 1e-6
```

## 📖 Documentation

### Comprehensive Guides

- **[BACKPROP.md](docs/BACKPROP.md)**: Gradient computation with JavaScript examples
- **[OPTIMIZERS.md](docs/OPTIMIZERS.md)**: Optimizer implementations and usage
- **[TOKENIZERS.md](docs/TOKENIZERS.md)**: Tokenization strategies and examples
- **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**: TypeScript, browser builds, and demos

### API Documentation

All classes and methods include comprehensive JSDoc comments for IDE support.

## 🤝 Contributing

### Development Setup

```bash
# Install dependencies
npm install

# Run tests
npm test

# Check code style
npm run lint
npm run format:check

# Build for production
npm run build:browser
```

### Code Style

- **ESLint**: Enforced code quality rules
- **Prettier**: Consistent code formatting
- **JSDoc**: Comprehensive documentation
- **ES6 Modules**: Modern JavaScript standards

## 📄 License

This project is licensed under the ISC License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Python implementation inspiration
- ml-matrix library for JavaScript matrix operations
- Educational resources on transformer architectures
- Open source community contributions

## 🔗 Related Projects

- **Original Python numpyGPT**: [Link to Python version]
- **PyTorch GPT**: Reference implementation
- **Transformer Papers**: Attention Is All You Need

---

**Built with ❤️ for education and understanding of transformer architectures**
