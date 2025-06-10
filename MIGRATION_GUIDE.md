# Migration Guide: Python to JavaScript numpyGPT

This guide helps you migrate from the Python version of numpyGPT to the JavaScript implementation, covering API differences, code patterns, and best practices.

## 🔄 Quick Migration Overview

The JavaScript implementation maintains **100% feature parity** with the Python version while adapting to JavaScript conventions and modern ES6+ syntax.

### Key Differences Summary

| Aspect | Python | JavaScript |
|--------|--------|------------|
| **Modules** | `import numpy as np` | `import { Matrix } from 'ml-matrix'` |
| **Classes** | `class GPT:` | `export class GPT extends Module` |
| **Arrays** | `np.array([[1, 2]])` | `new Matrix([[1, 2]])` |
| **File I/O** | `np.save()` / `np.load()` | `fs.writeFileSync()` / `JSON.parse()` |
| **Testing** | `pytest` | `node test.js` or Jest |

## 📦 Installation & Setup

### Python Version
```bash
pip install numpy matplotlib
```

### JavaScript Version
```bash
npm install
# Dependencies: ml-matrix, fs-extra, chalk
```

## 🧩 Core API Migration

### 1. Matrix Operations

#### Python (NumPy)
```python
import numpy as np

# Create matrices
X = np.array([[1, 2, 3], [4, 5, 6]])
Y = np.zeros((2, 3))
Z = np.ones((3, 2))

# Operations
result = X @ Z  # Matrix multiplication
mean = np.mean(X, axis=1)
X_T = X.T  # Transpose
```

#### JavaScript (ml-matrix)
```javascript
import { Matrix } from 'ml-matrix';

// Create matrices
const X = new Matrix([[1, 2, 3], [4, 5, 6]]);
const Y = Matrix.zeros(2, 3);
const Z = Matrix.ones(3, 2);

// Operations
const result = X.mmul(Z);  // Matrix multiplication
const mean = X.mean('row');
const X_T = X.transpose();  // Transpose
```

### 2. Model Definition

#### Python
```python
class GPT:
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff):
        self.vocab_size = vocab_size
        self.max_len = max_len
        # ... initialize layers
        
    def forward(self, X, targets=None):
        # Forward pass implementation
        pass
        
    def backward(self):
        # Backward pass implementation
        pass
```

#### JavaScript
```javascript
export class GPT extends Module {
    constructor(vocabSize, maxLen, dModel, nHeads, nLayers, dFF) {
        super();
        this.vocabSize = vocabSize;
        this.maxLen = maxLen;
        // ... initialize layers
    }
    
    forward(X, targets = null) {
        // Forward pass implementation
    }
    
    backward() {
        // Backward pass implementation
    }
}
```

### 3. Training Loop

#### Python
```python
# Training setup
model = GPT(vocab_size=1000, max_len=128, d_model=256, n_heads=8, n_layers=6, d_ff=1024)
optimizer = Adam([model], lr=0.001)

# Training loop
for step in range(max_iters):
    # Get batch
    X, Y = get_batch(data, batch_size, block_size)
    
    # Forward pass
    logits, loss = model.forward(X, Y)
    
    # Backward pass
    model.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

#### JavaScript
```javascript
// Training setup
const model = new GPT(1000, 128, 256, 8, 6, 1024);
const optimizer = new Adam([model], 0.001);

// Training loop
for (let step = 0; step < maxIters; step++) {
    // Get batch
    const { X, Y } = getBatch(data, batchSize, blockSize);
    
    // Forward pass
    const [logits, loss] = model.forward(X, Y);
    
    // Backward pass
    model.backward();
    
    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
}
```

## 🔤 Tokenizer Migration

### Python
```python
from tokenizer.char_level import CharTokenizer

tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

# Encoding/Decoding
tokens = tokenizer.encode("hello world", add_bos=True, add_eos=True)
text = tokenizer.decode(tokens)
```

### JavaScript
```javascript
import { CharTokenizer } from './src/tokenizer/char_level.js';

const tokenizer = new CharTokenizer();
tokenizer.buildVocab(text);

// Encoding/Decoding
const tokens = tokenizer.encode("hello world", true, true);
const text = tokenizer.decode(tokens);
```

## ⚡ Optimizer Migration

### Python
```python
from optim.adam import Adam
from optim.lr_scheduler.warmup_cosine_lr import WarmupCosineLR

optimizer = Adam([model], lr=0.001, betas=(0.9, 0.999))
scheduler = WarmupCosineLR(optimizer, warmup_steps=100, total_steps=1000)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

### JavaScript
```javascript
import { Adam } from './src/optim/adam.js';
import { WarmupCosineLR } from './src/optim/lr_scheduler/warmup_cosine_lr.js';

const optimizer = new Adam([model], 0.001, [0.9, 0.999]);
const scheduler = new WarmupCosineLR(optimizer, 100, 1000);

// Training step
optimizer.zeroGrad();
model.backward();
optimizer.step();
scheduler.step();
```

## 📁 File I/O Migration

### Model Saving/Loading

#### Python
```python
import pickle
import numpy as np

# Save model
model_data = {
    'config': config,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
```

#### JavaScript
```javascript
import fs from 'fs';
import { ModelIO } from './src/utils/model_io.js';

// Save model
const modelData = {
    config: config,
    parameters: ModelIO._serializeParameters(model),
    optimizer: optimizer.getState()
};
fs.writeFileSync('model.json', JSON.stringify(modelData, null, 2));

// Load model
const modelData = JSON.parse(fs.readFileSync('model.json', 'utf-8'));
ModelIO._deserializeParameters(model, modelData.parameters);
```

### Cross-Platform Compatibility

#### Export from JavaScript to Python
```javascript
import { ModelIO } from './src/utils/model_io.js';

// Export to Python-compatible format
ModelIO.exportModel(model, tokenizer, 'model_for_python.json');

// Export to NumPy format
ModelIO.exportToNumPy(model, tokenizer, 'model.npz');
```

#### Import in Python
```python
import json
import numpy as np

# Load from JavaScript
with open('model_for_python.json', 'r') as f:
    js_model = json.load(f)

# Load from NumPy format
model_data = np.load('model.npz', allow_pickle=True)
```

## 🧪 Testing Migration

### Python (pytest)
```python
import pytest
import numpy as np

def test_linear_layer():
    layer = Linear(10, 5)
    X = np.random.randn(3, 10)
    Y = layer.forward(X)
    assert Y.shape == (3, 5)

if __name__ == "__main__":
    pytest.main([__file__])
```

### JavaScript (Node.js)
```javascript
import { Matrix } from 'ml-matrix';
import { Linear } from './src/nn/modules/linear.js';

function testLinearLayer() {
    const layer = new Linear(10, 5);
    const X = Matrix.random(3, 10);
    const Y = layer.forward(X);
    console.assert(Y.rows === 3 && Y.columns === 5);
}

// Run test
testLinearLayer();
console.log('✅ Test passed!');
```

## 🎯 Common Migration Patterns

### 1. Array Indexing

#### Python
```python
# NumPy arrays
X[0, 1]  # Get element
X[:, 1]  # Get column
X[0, :]  # Get row
```

#### JavaScript
```javascript
// ml-matrix
X.get(0, 1)     // Get element
X.getColumn(1)  // Get column
X.getRow(0)     // Get row
```

### 2. Mathematical Operations

#### Python
```python
# Element-wise operations
result = np.exp(X)
result = np.log(X + 1e-8)
result = np.maximum(X, 0)  # ReLU
```

#### JavaScript
```javascript
// Element-wise operations
const result = X.apply((value) => Math.exp(value));
const result = X.apply((value) => Math.log(value + 1e-8));
const result = X.apply((value) => Math.max(value, 0));  // ReLU
```

### 3. Random Number Generation

#### Python
```python
import numpy as np

# Set seed
np.random.seed(42)

# Generate random arrays
X = np.random.randn(3, 4)
Y = np.random.uniform(0, 1, (2, 3))
```

#### JavaScript
```javascript
import { Matrix } from 'ml-matrix';

// Set seed (if using a seeded random library)
// Math.seedrandom(42);  // requires seedrandom library

// Generate random matrices
const X = Matrix.random(3, 4, { random: () => Math.random() * 2 - 1 });  // Normal-like
const Y = Matrix.random(2, 3);  // Uniform [0, 1]
```

## 🔧 Configuration Migration

### Python Configuration
```python
# config.py
CONFIG = {
    'vocab_size': 1000,
    'max_len': 128,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 1024,
    'lr': 0.001,
    'batch_size': 4
}
```

### JavaScript Configuration
```javascript
// config.js
export const CONFIG = {
    vocabSize: 1000,
    maxLen: 128,
    dModel: 256,
    nHeads: 8,
    nLayers: 6,
    dFF: 1024,
    lr: 0.001,
    batchSize: 4
};
```

## 🚀 Performance Considerations

### Memory Management

#### Python
```python
# Automatic garbage collection
# Use del for explicit cleanup
del large_array
```

#### JavaScript
```javascript
// Automatic garbage collection
// Set to null for explicit cleanup
largeMatrix = null;
```

### Optimization Tips

1. **Use TypeScript**: Add type safety with the provided `.d.ts` files
2. **Browser Optimization**: Use the webpack build for browser deployment
3. **Memory Monitoring**: Use the built-in performance benchmarking tools
4. **Batch Processing**: Optimize batch sizes for your target environment

## 🌐 Browser-Specific Migration

### Python (Not applicable)
```python
# Python runs server-side only
```

### JavaScript (Browser support)
```html
<!DOCTYPE html>
<html>
<head>
    <title>numpyGPT in Browser</title>
</head>
<body>
    <script src="dist/numpygpt.bundle.js"></script>
    <script>
        // Use the global numpyGPT object
        const model = numpyGPT.createModel({
            vocabSize: 100,
            dModel: 64,
            nHeads: 2,
            nLayers: 2
        });
        
        // Training and inference in the browser
        const result = numpyGPT.trainStep(model, optimizer, X, Y);
        const text = numpyGPT.generateText(model, tokenizer, "hello", 20);
    </script>
</body>
</html>
```

## 📊 Validation & Testing

### Cross-Platform Validation

```javascript
// Validate JavaScript vs Python outputs
node validate_parity.js --verbose --tolerance 1e-6

// Expected output:
// ✅ Forward pass validation: Loss 2.995496
// ✅ Gradient computation validation: Loss 2.705712
// ✅ Training step validation: 3.221241 → 2.987654
// ✅ Text generation validation: "hello world"
```

## 🎓 Learning Path

### For Python Developers

1. **Start with Core Concepts**: Understand Matrix operations in ml-matrix
2. **Module System**: Learn ES6 import/export syntax
3. **Async Patterns**: Understand JavaScript promises and async/await
4. **Browser Development**: Explore webpack and browser compatibility
5. **TypeScript**: Add type safety for better development experience

### Recommended Migration Order

1. **Tokenizers**: Start with simple character-level tokenization
2. **Neural Network Modules**: Migrate layer by layer
3. **Model Architecture**: Implement the complete GPT model
4. **Training Loop**: Set up optimizers and training procedures
5. **Validation**: Cross-validate with Python implementation
6. **Advanced Features**: Add browser support and visualization

## 🔍 Debugging & Troubleshooting

### Common Issues

#### Matrix Shape Mismatches
```javascript
// Check matrix dimensions
console.log(`Matrix shape: ${matrix.rows} x ${matrix.columns}`);

// Debug matrix operations
console.log('Input shape:', X.rows, 'x', X.columns);
console.log('Weight shape:', W.rows, 'x', W.columns);
```

#### Import/Export Errors
```javascript
// Ensure proper ES6 module syntax
import { GPT } from './src/models/GPT.js';  // Include .js extension

// Check file paths
import { Linear } from '../nn/modules/linear.js';  // Relative paths
```

#### Browser Compatibility
```javascript
// Check for browser-specific issues
if (typeof window !== 'undefined') {
    // Browser-specific code
    console.log('Running in browser');
} else {
    // Node.js-specific code
    console.log('Running in Node.js');
}
```

## 📚 Additional Resources

### Documentation
- **[README.md](README.md)**: Complete JavaScript documentation
- **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**: TypeScript, browser builds, demos
- **[docs/](docs/)**: Detailed technical documentation

### Examples
- **[demo/index.html](demo/index.html)**: Interactive browser demo
- **[tests/](tests/)**: Comprehensive test suite
- **[benchmark.js](benchmark.js)**: Performance benchmarking

### Tools
- **[validate_parity.js](validate_parity.js)**: Cross-platform validation
- **[benchmark.js](benchmark.js)**: Performance measurement
- **[webpack.config.js](webpack.config.js)**: Browser build configuration

---

**Happy migrating! 🚀**

For questions or issues, please refer to the comprehensive test suite and validation tools to ensure your migration maintains the same behavior as the Python version.