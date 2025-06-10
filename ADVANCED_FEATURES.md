# Advanced Features Guide

This document describes the advanced features available in the numpyGPT JavaScript implementation, including TypeScript support, browser compatibility, and interactive demos.

## 🎯 TypeScript Support

### Type Definitions

The project includes comprehensive TypeScript definitions in [`types/index.d.ts`](types/index.d.ts) that provide:

- **Complete API Coverage**: Type definitions for all classes, functions, and interfaces
- **IDE Support**: Enhanced autocomplete, error checking, and documentation
- **Type Safety**: Compile-time type checking for better code quality

### Usage in TypeScript Projects

```typescript
import { GPT, Adam, CharTokenizer } from 'numpygpt-js';
import { Matrix } from 'ml-matrix';

// TypeScript will provide full type checking and autocomplete
const model: GPT = new GPT(1000, 128, 256, 8, 6, 1024);
const optimizer: Adam = new Adam([model], 0.001);
const tokenizer: CharTokenizer = new CharTokenizer();

// Type-safe training configuration
interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  maxIters: number;
}

const config: TrainingConfig = {
  batchSize: 4,
  learningRate: 0.001,
  maxIters: 100
};
```

### Building Type Definitions

```bash
# Generate TypeScript declarations
npm run build:types
```

## 🌐 Browser Compatibility

### Browser Build System

The project uses Webpack to create browser-compatible bundles:

- **Universal Module Definition (UMD)**: Works in browsers, Node.js, and AMD environments
- **Polyfills**: Includes necessary polyfills for browser compatibility
- **Optimized Bundle**: Minified and optimized for production use

### Building for Browser

```bash
# Build production bundle
npm run build:browser

# Start development server with hot reload
npm run dev:browser
```

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
    <title>numpyGPT in Browser</title>
</head>
<body>
    <!-- Load the bundled library -->
    <script src="dist/numpygpt.bundle.js"></script>
    
    <script>
        // numpyGPT is available globally
        const model = numpyGPT.createModel({
            vocabSize: 1000,
            dModel: 256,
            nHeads: 8,
            nLayers: 6
        });
        
        const tokenizer = numpyGPT.createTokenizer('char');
        const optimizer = numpyGPT.createOptimizer(model, 'adam', 0.001);
        
        // Train and generate text in the browser
        async function trainAndGenerate() {
            // Training data
            const text = "Hello world, this is a test.";
            tokenizer.buildVocab(text);
            
            // Simple training loop
            for (let i = 0; i < 10; i++) {
                const tokens = tokenizer.encode(text);
                const X = new Matrix([tokens.slice(0, -1)]);
                const Y = new Matrix([tokens.slice(1)]);
                
                const result = await numpyGPT.trainStep(model, optimizer, X, Y);
                console.log(`Step ${i}: Loss = ${result.loss}`);
            }
            
            // Generate text
            const generated = numpyGPT.generateText(model, tokenizer, "Hello", 20);
            console.log("Generated:", generated);
        }
        
        trainAndGenerate();
    </script>
</body>
</html>
```

## 🎮 Interactive Web Demo

### Demo Features

The interactive web demo ([`demo/index.html`](demo/index.html)) provides:

- **Real-time Training**: Watch the model train with live loss updates
- **Text Generation**: Generate text with adjustable parameters
- **Architecture Visualization**: Explore model architecture and parameters
- **Responsive Design**: Works on desktop and mobile devices

### Demo Sections

#### 1. Training Tab
- **Model Configuration**: Adjust vocabulary size, dimensions, layers, and heads
- **Training Parameters**: Set learning rate, training steps, and tokenizer type
- **Live Metrics**: Monitor loss, learning rate, and training time
- **Progress Tracking**: Visual progress bar and detailed logging

#### 2. Text Generation Tab
- **Prompt Input**: Enter custom prompts for text generation
- **Generation Parameters**: Control temperature, max tokens, and number of samples
- **Multiple Samples**: Generate multiple variations with different randomness

#### 3. Architecture Tab
- **Model Information**: Detailed architecture breakdown
- **Parameter Count**: Total parameters and memory usage
- **Layer Structure**: Complete model layer information

### Running the Demo

```bash
# Start development server
npm run dev:browser

# Or serve the demo directory with any static server
# The demo will be available at http://localhost:8080
```

### Demo API

The demo exposes a simplified API for browser usage:

```javascript
// Quick model creation
const model = numpyGPT.createModel({
    vocabSize: 1000,
    maxLen: 128,
    dModel: 256,
    nHeads: 8,
    nLayers: 6
});

// Quick tokenizer creation
const tokenizer = numpyGPT.createTokenizer('char'); // 'char', 'word', or 'bpe'

// Quick optimizer creation
const optimizer = numpyGPT.createOptimizer(model, 'adam', 0.001);

// Browser-friendly training step
const result = await numpyGPT.trainStep(model, optimizer, X, Y, {
    gradClip: 1.0,
    scheduler: null
});

// Easy text generation
const generated = numpyGPT.generateText(model, tokenizer, "Hello", 50, 1.0);
```

## 🔧 Configuration Files

### Webpack Configuration

[`webpack.config.js`](webpack.config.js) provides:

- **Entry Point**: [`src/index.js`](src/index.js) - Browser-optimized entry
- **Output**: UMD bundle in `dist/numpygpt.bundle.js`
- **Babel Transpilation**: ES6+ to ES5 for broader browser support
- **Development Server**: Hot reload for development
- **Polyfills**: Node.js modules for browser compatibility

### Package.json Scripts

```json
{
  "scripts": {
    "build:browser": "webpack --config webpack.config.js",
    "build:types": "tsc --declaration --emitDeclarationOnly --outDir types",
    "dev:browser": "webpack serve --config webpack.config.js --mode development"
  }
}
```

## 🚀 Performance Considerations

### Browser Optimizations

- **Bundle Size**: Optimized for minimal bundle size while maintaining functionality
- **Memory Management**: Efficient matrix operations using ml-matrix
- **Async Operations**: Non-blocking training and generation
- **Progressive Loading**: Lazy loading of non-essential features

### Limitations in Browser

- **File System**: No direct file I/O (use localStorage or IndexedDB)
- **Memory Constraints**: Limited by browser memory allocation
- **Performance**: Slower than Node.js due to browser overhead
- **Model Size**: Recommended to use smaller models for browser deployment

## 📱 Mobile Compatibility

### Responsive Design

The demo is fully responsive and works on:

- **Desktop**: Full feature set with optimal layout
- **Tablet**: Adapted layout with touch-friendly controls
- **Mobile**: Simplified interface optimized for small screens

### Mobile Considerations

- **Memory Usage**: Use smaller models on mobile devices
- **Battery Life**: Training can be battery-intensive
- **Performance**: Expect slower training on mobile hardware
- **Touch Interface**: All controls are touch-optimized

## 🔮 Future Enhancements

### Planned Features

- **WebGL Acceleration**: GPU-accelerated matrix operations
- **Web Workers**: Background training without blocking UI
- **Progressive Web App**: Offline capability and app-like experience
- **Model Sharing**: Save/load models to/from cloud storage
- **Advanced Visualizations**: Real-time training visualizations
- **Multi-language Support**: Internationalization for global users

### Contributing

To contribute to the advanced features:

1. **TypeScript**: Add type definitions for new features
2. **Browser Compatibility**: Test across different browsers
3. **Demo Enhancements**: Improve the interactive demo
4. **Performance**: Optimize for better browser performance
5. **Documentation**: Update this guide with new features

## 📚 Examples

### Complete Browser Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>numpyGPT Complete Example</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .output { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; }
    </style>
</head>
<body>
    <h1>numpyGPT Browser Example</h1>
    
    <button onclick="runExample()">Run Training Example</button>
    <div id="output" class="output">Click the button to start...</div>
    
    <script src="dist/numpygpt.bundle.js"></script>
    <script>
        async function runExample() {
            const output = document.getElementById('output');
            output.innerHTML = 'Starting training...<br>';
            
            try {
                // Create components
                const tokenizer = numpyGPT.createTokenizer('char');
                const text = "hello world this is a simple example";
                tokenizer.buildVocab(text);
                
                const model = numpyGPT.createModel({
                    vocabSize: tokenizer.vocabSize,
                    maxLen: 32,
                    dModel: 64,
                    nHeads: 2,
                    nLayers: 2
                });
                
                const optimizer = numpyGPT.createOptimizer(model, 'adam', 0.01);
                
                output.innerHTML += `Model created with ${tokenizer.vocabSize} vocab size<br>`;
                
                // Training data
                const tokens = tokenizer.encode(text, false, false);
                const seqLen = 8;
                
                // Training loop
                for (let step = 0; step < 20; step++) {
                    // Create batch
                    const startIdx = Math.floor(Math.random() * (tokens.length - seqLen - 1));
                    const X = new Matrix([[tokens.slice(startIdx, startIdx + seqLen)]]);
                    const Y = new Matrix([[tokens.slice(startIdx + 1, startIdx + seqLen + 1)]]);
                    
                    // Training step
                    const result = await numpyGPT.trainStep(model, optimizer, X, Y);
                    
                    if (step % 5 === 0) {
                        output.innerHTML += `Step ${step}: Loss = ${result.loss.toFixed(4)}<br>`;
                    }
                }
                
                // Generate text
                output.innerHTML += '<br>Generating text...<br>';
                const generated = numpyGPT.generateText(model, tokenizer, "hello", 15, 1.0);
                output.innerHTML += `Generated: "${generated}"<br>`;
                
                output.innerHTML += '<br>Training completed successfully!';
                
            } catch (error) {
                output.innerHTML += `<br>Error: ${error.message}`;
                console.error(error);
            }
        }
    </script>
</body>
</html>
```

This example demonstrates the complete workflow from model creation to text generation in a browser environment.