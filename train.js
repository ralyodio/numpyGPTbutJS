#!/usr/bin/env node

/**
 * Training Script for numpyGPT JavaScript Implementation
 * JavaScript equivalent of train.py
 * 
 * This script:
 * 1. Loads preprocessed data and tokenizer
 * 2. Initializes GPT model, optimizer, and scheduler
 * 3. Runs training loop with evaluation and checkpointing
 * 4. Saves model checkpoints and training metrics
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Matrix } from 'ml-matrix';
import { GPT } from './src/models/GPT.js';
import { Adam } from './src/optim/adam.js';
import { WarmupCosineLR } from './src/optim/lr_scheduler/warmup_cosine_lr.js';
import { loadTokenizer, loadData } from './datagen.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Training configuration
const config = {
  // Data and output paths
  dataDir: 'data/test_char',
  outDir: 'out/char',
  alwaysSaveCheckpoint: true,
  resume: true,
  
  // Training hyperparameters
  batchSize: 4,  // Smaller for testing
  blockSize: 32, // Smaller for testing
  lr: 3e-4,
  minLr: 3e-5,
  
  // Model architecture
  nLayer: 2,     // Smaller for testing
  nHead: 2,      // Smaller for testing
  nEmbd: 64,     // Smaller for testing
  
  // Training schedule
  maxIters: 100,   // Much smaller for testing
  warmupIters: 10,
  lrDecayIters: 100,
  evalInterval: 20,
  evalIters: 5,
  logInterval: 5,
  gradClip: 1.0
};

// Derived config
config.dFF = 4 * config.nEmbd;

/**
 * Simple DataLoader class for batch generation
 */
class DataLoader {
  constructor(dataDir, split, batchSize, blockSize) {
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    
    // Load data
    const dataPath = path.join(dataDir, `${split}.bin`);
    this.data = loadData(dataPath);
    
    // Load tokenizer to get vocab size
    const tokenizerPath = path.join(dataDir, 'tokenizer.json');
    const tokenizer = loadTokenizer(tokenizerPath);
    this.vocabSize = tokenizer.vocabSize;
    
    console.log(`Loaded ${split} data: ${this.data.length} tokens, vocab_size: ${this.vocabSize}`);
  }
  
  /**
   * Get a batch of data
   * @returns {Object} - {X: Matrix, Y: Matrix}
   */
  getBatch() {
    const B = this.batchSize;
    const T = this.blockSize;
    
    // Sample random starting positions
    const X = Matrix.zeros(B, T); // (B, T) format
    const Y = Matrix.zeros(B, T); // (B, T) format
    
    for (let b = 0; b < B; b++) {
      // Random start position (ensure we have enough tokens for sequence + 1)
      const startIdx = Math.floor(Math.random() * (this.data.length - T - 1));
      
      for (let t = 0; t < T; t++) {
        X.set(b, t, this.data[startIdx + t]);
        Y.set(b, t, this.data[startIdx + t + 1]);
      }
    }
    
    return { X, Y };
  }
}

/**
 * Simple training monitor for logging
 */
class TrainingMonitor {
  constructor(logInterval) {
    this.logInterval = logInterval;
    this.losses = [];
    this.startTime = Date.now();
  }
  
  logStep(iterNum, loss, lr, gradNorm) {
    this.losses.push(loss);
    
    if (iterNum % this.logInterval === 0) {
      const avgLoss = this.losses.slice(-this.logInterval).reduce((a, b) => a + b, 0) / Math.min(this.logInterval, this.losses.length);
      const elapsed = (Date.now() - this.startTime) / 1000;
      const gradNormStr = gradNorm !== null ? `, grad_norm: ${gradNorm.toFixed(4)}` : '';
      return `iter ${iterNum}: loss ${avgLoss.toFixed(4)}, lr: ${lr.toExponential(2)}${gradNormStr}, time: ${elapsed.toFixed(1)}s`;
    }
    
    return null;
  }
}

/**
 * Simple metrics logger
 */
class MetricsLogger {
  constructor(filepath) {
    this.filepath = filepath;
    this.metrics = [];
  }
  
  log(iterNum, trainLoss = null, valLoss = null, lr = null, gradNorm = null) {
    const entry = { iter: iterNum, timestamp: Date.now() };
    if (trainLoss !== null) entry.train_loss = trainLoss;
    if (valLoss !== null) entry.val_loss = valLoss;
    if (lr !== null) entry.lr = lr;
    if (gradNorm !== null) entry.grad_norm = gradNorm;
    
    this.metrics.push(entry);
    
    // Save to file
    fs.writeFileSync(this.filepath, JSON.stringify(this.metrics, null, 2));
  }
}

/**
 * Clip gradient norm
 * @param {GPT} model - Model to clip gradients for
 * @param {number} maxNorm - Maximum gradient norm
 * @returns {number} - Actual gradient norm
 */
function clipGradNorm(model, maxNorm) {
  const grads = model.grads();
  let totalNorm = 0;
  
  // Calculate total gradient norm
  for (const grad of Object.values(grads)) {
    for (let i = 0; i < grad.rows; i++) {
      for (let j = 0; j < grad.columns; j++) {
        const val = grad.get(i, j);
        totalNorm += val * val;
      }
    }
  }
  
  totalNorm = Math.sqrt(totalNorm);
  
  // Clip if necessary
  if (totalNorm > maxNorm) {
    const clipCoeff = maxNorm / totalNorm;
    for (const grad of Object.values(grads)) {
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          grad.set(i, j, grad.get(i, j) * clipCoeff);
        }
      }
    }
  }
  
  return totalNorm;
}

/**
 * Get current learning rate from optimizer
 * @param {Adam} optimizer - Optimizer instance
 * @returns {number} - Current learning rate
 */
function getLr(optimizer) {
  return optimizer.lr;
}

/**
 * Save model checkpoint
 * @param {string} filepath - Path to save checkpoint
 * @param {GPT} model - Model to save
 * @param {number} iterNum - Current iteration
 * @param {number} valLoss - Validation loss (optional)
 * @param {Object} optimizerState - Optimizer state (optional)
 */
function saveModel(filepath, model, iterNum, valLoss = null, optimizerState = null) {
  const modelData = {
    model: {},
    iterNum,
    config: {
      vocabSize: model.vocabSize,
      maxLen: model.maxLen,
      dModel: model.dModel,
      nHeads: model.nHeads,
      nLayers: model.nLayers,
      dFF: model.dFF
    }
  };
  
  // Convert Matrix objects to arrays for JSON serialization
  const params = model.params();
  for (const [name, param] of Object.entries(params)) {
    modelData.model[name] = param.to2DArray();
  }
  
  if (valLoss !== null) {
    modelData.valLoss = valLoss;
  }
  
  if (optimizerState !== null) {
    modelData.optimizerState = optimizerState;
  }
  
  fs.writeFileSync(filepath, JSON.stringify(modelData, null, 2));
}

/**
 * Load model checkpoint
 * @param {string} filepath - Path to checkpoint
 * @param {GPT} model - Model to load into
 * @returns {Object} - Checkpoint data
 */
function loadModel(filepath, model) {
  const data = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
  
  // Load parameters
  const params = model.params();
  for (const [name, paramArray] of Object.entries(data.model)) {
    if (params[name]) {
      const matrix = new Matrix(paramArray);
      // Copy values
      for (let i = 0; i < matrix.rows; i++) {
        for (let j = 0; j < matrix.columns; j++) {
          params[name].set(i, j, matrix.get(i, j));
        }
      }
    }
  }
  
  return data;
}

/**
 * Estimate loss on train/val sets
 * @param {GPT} model - Model to evaluate
 * @param {DataLoader} trainLoader - Training data loader
 * @param {DataLoader} valLoader - Validation data loader
 * @param {number} evalIters - Number of evaluation iterations
 * @returns {Object} - {train: number, val: number}
 */
function estimateLoss(model, trainLoader, valLoader, evalIters) {
  const out = {};
  
  for (const split of ['train', 'val']) {
    const loader = split === 'train' ? trainLoader : valLoader;
    const losses = [];
    
    for (let k = 0; k < evalIters; k++) {
      const { X, Y } = loader.getBatch();
      const [logits, loss] = model.forward(X, Y);
      losses.push(loss);
    }
    
    out[split] = losses.reduce((a, b) => a + b, 0) / losses.length;
  }
  
  return out;
}

/**
 * Main training function
 */
async function train() {
  console.log('Starting numpyGPT training...');
  console.log('Configuration:', JSON.stringify(config, null, 2));
  
  // Create output directory
  if (!fs.existsSync(config.outDir)) {
    fs.mkdirSync(config.outDir, { recursive: true });
  }
  
  // Save config
  fs.writeFileSync(
    path.join(config.outDir, 'config.json'),
    JSON.stringify(config, null, 2)
  );
  
  // Create data loaders
  const trainLoader = new DataLoader(config.dataDir, 'train', config.batchSize, config.blockSize);
  const valLoader = new DataLoader(config.dataDir, 'val', config.batchSize, config.blockSize);
  
  const vocabSize = trainLoader.vocabSize;
  console.log(`Vocabulary size: ${vocabSize}`);
  
  // Create model
  const model = new GPT(
    vocabSize,
    config.blockSize,
    config.nEmbd,
    config.nHead,
    config.nLayer,
    config.dFF
  );
  
  // Create optimizer and scheduler
  const optimizer = new Adam([model], config.lr);
  const scheduler = new WarmupCosineLR(
    optimizer,
    config.warmupIters,
    config.lrDecayIters,
    config.minLr
  );
  
  // Create monitoring
  const monitor = new TrainingMonitor(config.logInterval);
  const metrics = new MetricsLogger(path.join(config.outDir, 'metrics.json'));
  
  // Training state
  let iterNum = 0;
  let bestValLoss = Infinity;
  let resumeFromCheckpoint = false;
  
  // Resume from checkpoint if exists
  const ckptPath = path.join(config.outDir, 'ckpt.json');
  if (config.resume && fs.existsSync(ckptPath)) {
    console.log(`Resuming training from ${ckptPath}`);
    const checkpoint = loadModel(ckptPath, model);
    
    if (checkpoint.optimizerState) {
      // Note: In a full implementation, we'd restore optimizer state
      console.log('Optimizer state restoration not fully implemented');
    }
    
    iterNum = checkpoint.iterNum;
    bestValLoss = checkpoint.bestValLoss || Infinity;
    
    // Advance scheduler
    for (let i = 0; i < iterNum; i++) {
      scheduler.step();
    }
    
    resumeFromCheckpoint = true;
    console.log(`Resumed from iteration ${iterNum}, best_val_loss=${bestValLoss.toFixed(4)}`);
  }
  
  // Count parameters
  const params = model.params();
  let numParams = 0;
  for (const param of Object.values(params)) {
    numParams += param.rows * param.columns;
  }
  console.log(`Number of parameters: ${(numParams / 1e6).toFixed(2)}M`);
  
  if (!resumeFromCheckpoint) {
    console.log('Starting training from scratch');
  }
  
  const startTime = Date.now();
  
  // Training loop
  while (iterNum < config.maxIters) {
    // Evaluation
    if (iterNum % config.evalInterval === 0) {
      const losses = estimateLoss(model, trainLoader, valLoader, config.evalIters);
      console.log(`Step ${iterNum}: train loss ${losses.train.toFixed(4)}, val loss ${losses.val.toFixed(4)}`);
      
      metrics.log(iterNum, null, losses.val, getLr(optimizer));
      
      // Save checkpoint
      if (losses.val < bestValLoss || config.alwaysSaveCheckpoint) {
        if (losses.val < bestValLoss) {
          bestValLoss = losses.val;
        }
        
        if (iterNum > 0) {
          console.log(`Saving checkpoint to ${config.outDir}`);
          saveModel(ckptPath, model, iterNum, losses.val, null);
          
          if (losses.val < bestValLoss) {
            saveModel(path.join(config.outDir, 'best_model.json'), model, iterNum, losses.val);
          }
        }
      }
    }
    
    // Training step
    optimizer.zeroGrad();
    
    const { X, Y } = trainLoader.getBatch();
    const [logits, loss] = model.forward(X, Y);
    model.backward();
    
    // Gradient clipping
    const gradNorm = config.gradClip > 0 ? clipGradNorm(model, config.gradClip) : null;
    
    optimizer.step();
    scheduler.step();
    
    // Logging
    metrics.log(iterNum, loss, null, getLr(optimizer), gradNorm);
    
    const logMsg = monitor.logStep(iterNum, loss, getLr(optimizer), gradNorm);
    if (logMsg) {
      console.log(logMsg);
    }
    
    iterNum++;
  }
  
  const endTime = Date.now();
  const totalTime = (endTime - startTime) / 1000;
  console.log(`Training finished in ${totalTime.toFixed(2)}s`);
  
  console.log(`Training metrics saved to ${path.join(config.outDir, 'metrics.json')}`);
}

/**
 * Parse command line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key === 'data_dir') config.dataDir = value;
    else if (key === 'out_dir') config.outDir = value;
    else if (key === 'batch_size') config.batchSize = parseInt(value);
    else if (key === 'block_size') config.blockSize = parseInt(value);
    else if (key === 'lr') config.lr = parseFloat(value);
    else if (key === 'max_iters') config.maxIters = parseInt(value);
    else if (key === 'n_layer') config.nLayer = parseInt(value);
    else if (key === 'n_head') config.nHead = parseInt(value);
    else if (key === 'n_embd') config.nEmbd = parseInt(value);
    else if (key === 'help') {
      console.log(`
Usage: node train.js [options]

Options:
  --data_dir <path>         Data directory (default: data/test_char)
  --out_dir <path>          Output directory (default: out/char)
  --batch_size <int>        Batch size (default: 4)
  --block_size <int>        Block size (default: 32)
  --lr <float>              Learning rate (default: 3e-4)
  --max_iters <int>         Maximum iterations (default: 100)
  --n_layer <int>           Number of layers (default: 2)
  --n_head <int>            Number of heads (default: 2)
  --n_embd <int>            Embedding dimension (default: 64)
  --help                    Show this help message

Examples:
  node train.js --data_dir data/test_char --max_iters 200
  node train.js --data_dir data/test_bpe --n_layer 4 --n_head 4
      `);
      process.exit(0);
    }
  }
  
  // Update derived config
  config.dFF = 4 * config.nEmbd;
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    parseArgs();
    await train();
  } catch (error) {
    console.error('Training failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Export for use as module
export { train, DataLoader, TrainingMonitor, MetricsLogger, clipGradNorm, getLr, saveModel, loadModel };