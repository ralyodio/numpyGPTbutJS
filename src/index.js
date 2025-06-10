/**
 * Browser entry point for numpyGPT JavaScript Implementation
 * Exports all major components for browser usage
 */

// Core Neural Network Modules
export { Module } from './nn/modules/module.js';
export { Linear } from './nn/modules/linear.js';
export { Embedding } from './nn/modules/embedding.js';
export { LayerNorm } from './nn/modules/layerNorm.js';
export { ReLU, Softmax } from './nn/modules/activation.js';
export { PositionalEncoding } from './nn/modules/positional.js';
export { MultiHeadAttention } from './nn/modules/attention.js';
export { FeedForward } from './nn/modules/feedforward.js';
export { TransformerBlock } from './nn/modules/transformer.js';

// Neural Network Functional
export { crossEntropyLoss, softmax } from './nn/functional.js';

// GPT Model
export { GPT } from './models/GPT.js';

// Optimizers
export { Optimizer } from './optim/optimizer.js';
export { Adam } from './optim/adam.js';

// Learning Rate Schedulers
export { LRScheduler } from './optim/lr_scheduler/lr_scheduler.js';
export { StepLR } from './optim/lr_scheduler/step_lr.js';
export { WarmupCosineLR } from './optim/lr_scheduler/warmup_cosine_lr.js';

// Tokenizers
export { CharTokenizer } from './tokenizer/char_level.js';
export { WordTokenizer } from './tokenizer/word_level.js';
export { BPETokenizer } from './tokenizer/bpe.js';

// Utilities (browser-compatible subset)
export { 
  TrainingMonitor, 
  EarlyStopping, 
  LRFinder,
  setupLogger,
  clipGradNorm,
  getLr 
} from './utils/training.js';

export { MetricsLogger } from './utils/vis.js';

// Browser-specific utilities
export const numpyGPT = {
  // Version info
  version: '1.0.0',
  
  // Quick model creation helper
  createModel: (config) => {
    const {
      vocabSize = 1000,
      maxLen = 128,
      dModel = 256,
      nHeads = 8,
      nLayers = 6,
      dFF = null
    } = config;
    
    const actualDFF = dFF || 4 * dModel;
    return new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, actualDFF);
  },
  
  // Quick tokenizer creation
  createTokenizer: (type = 'char') => {
    switch (type) {
      case 'char':
        return new CharTokenizer();
      case 'word':
        return new WordTokenizer();
      case 'bpe':
        return new BPETokenizer();
      default:
        throw new Error(`Unknown tokenizer type: ${type}`);
    }
  },
  
  // Quick optimizer creation
  createOptimizer: (model, type = 'adam', lr = 0.001) => {
    switch (type) {
      case 'adam':
        return new Adam([model], lr);
      default:
        throw new Error(`Unknown optimizer type: ${type}`);
    }
  },
  
  // Browser-friendly training loop
  async trainStep(model, optimizer, X, Y, options = {}) {
    const { gradClip = null, scheduler = null } = options;
    
    // Forward pass
    const [logits, loss] = model.forward(X, Y);
    
    // Backward pass
    model.backward();
    
    // Gradient clipping
    let gradNorm = null;
    if (gradClip) {
      gradNorm = clipGradNorm(model, gradClip);
    }
    
    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
    
    // Scheduler step
    if (scheduler) {
      scheduler.step();
    }
    
    return { loss, gradNorm, lr: optimizer.lr };
  },
  
  // Text generation helper
  generateText(model, tokenizer, prompt, maxTokens = 50, temperature = 1.0) {
    // Encode prompt
    const promptTokens = tokenizer.encode(prompt, false, false);
    const promptMatrix = new Matrix([promptTokens]);
    
    // Generate
    const generated = model.generate(promptMatrix, maxTokens, temperature, tokenizer.eosTokenId);
    
    // Decode
    const generatedTokens = generated.getRow(0);
    return tokenizer.decode(generatedTokens);
  }
};

// Default export for convenience
export default numpyGPT;