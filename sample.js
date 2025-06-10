#!/usr/bin/env node

/**
 * Sampling Script for numpyGPT JavaScript Implementation
 * JavaScript equivalent of sample.py
 * 
 * This script:
 * 1. Loads a trained GPT model from checkpoint
 * 2. Loads the corresponding tokenizer
 * 3. Generates text samples using the model
 * 4. Decodes and displays the generated text
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Matrix } from 'ml-matrix';
import { GPT } from './src/models/GPT.js';
import { loadTokenizer } from './datagen.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Sample text from a trained model
 * @param {string} modelPath - Path to model checkpoint
 * @param {string} dataDir - Data directory containing tokenizer
 * @param {number} numSamples - Number of samples to generate
 * @param {number} maxNewTokens - Maximum new tokens to generate
 * @param {number} temperature - Sampling temperature
 * @param {string} startText - Starting text prompt
 * @param {number} seed - Random seed (for reproducibility)
 */
function sampleFromModel(modelPath, dataDir, numSamples = 1, maxNewTokens = 100, 
                        temperature = 0.8, startText = "\n", seed = 1337) {
  console.log(`Loading model from ${modelPath}`);
  
  // Check if model file exists
  if (!fs.existsSync(modelPath)) {
    throw new Error(`Model checkpoint not found: ${modelPath}`);
  }
  
  // Load model checkpoint
  const checkpoint = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  const config = checkpoint.config;
  
  // Create model with same configuration
  const model = new GPT(
    config.vocabSize,
    config.maxLen,
    config.dModel,
    config.nHeads,
    config.nLayers,
    config.dFF
  );
  
  // Load model parameters
  const params = model.params();
  for (const [name, paramArray] of Object.entries(checkpoint.model)) {
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
  
  console.log('Model loaded');
  
  // Load tokenizer
  const tokenizerPath = path.join(dataDir, 'tokenizer.json');
  if (!fs.existsSync(tokenizerPath)) {
    throw new Error(`Tokenizer not found: ${tokenizerPath}`);
  }
  
  const tokenizer = loadTokenizer(tokenizerPath);
  console.log(`Vocab size: ${tokenizer.vocabSize}`);
  
  // Encode starting text
  const startIds = tokenizer.encode(startText);
  const x = new Matrix([startIds]); // (1, start_len)
  
  const eosTokenId = tokenizer.eosTokenId;
  
  console.log(`Generating ${numSamples} samples...`);
  
  // Set seed for reproducibility (simple implementation)
  let seedState = seed;
  Math.random = function() {
    seedState = (seedState * 9301 + 49297) % 233280;
    return seedState / 233280;
  };
  
  // Generate samples
  for (let k = 0; k < numSamples; k++) {
    console.log('\n' + '='.repeat(60));
    console.log(`SAMPLE ${k + 1}/${numSamples}`);
    console.log('='.repeat(60));
    
    try {
      // Generate text
      const y = model.generate(x, maxNewTokens, temperature, eosTokenId);
      
      // Convert to array for decoding
      const generatedIds = [];
      for (let i = 0; i < y.columns; i++) {
        generatedIds.push(y.get(0, i));
      }
      
      // Decode and print
      const decodedText = tokenizer.decode(generatedIds);
      console.log(decodedText);
      
    } catch (error) {
      console.error(`Error generating sample ${k + 1}:`, error.message);
      console.log('Trying with simpler generation...');
      
      // Fallback: just show the starting text
      const decodedText = tokenizer.decode(startIds);
      console.log(`[Starting text only]: ${decodedText}`);
    }
    
    console.log('='.repeat(60) + '\n');
  }
}

/**
 * Parse command line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    modelPath: 'out/char/best_model.json',
    dataDir: 'data/test_char',
    numSamples: 1,
    maxNewTokens: 100,
    temperature: 0.8,
    start: 'Hello',
    seed: 1337
  };
  
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key === 'model_path') options.modelPath = value;
    else if (key === 'data_dir') options.dataDir = value;
    else if (key === 'num_samples') options.numSamples = parseInt(value);
    else if (key === 'max_new_tokens') options.maxNewTokens = parseInt(value);
    else if (key === 'temperature') options.temperature = parseFloat(value);
    else if (key === 'start') options.start = value;
    else if (key === 'seed') options.seed = parseInt(value);
    else if (key === 'help') {
      console.log(`
Usage: node sample.js [options]

Options:
  --model_path <path>       Path to model checkpoint (default: out/char/best_model.json)
  --data_dir <path>         Data directory with tokenizer (default: data/test_char)
  --num_samples <int>       Number of samples to generate (default: 1)
  --max_new_tokens <int>    Maximum new tokens to generate (default: 100)
  --temperature <float>     Sampling temperature (default: 0.8)
  --start <text>            Starting text prompt (default: "Hello")
  --seed <int>              Random seed (default: 1337)
  --help                    Show this help message

Examples:
  node sample.js --start "Hello world" --num_samples 3
  node sample.js --model_path out/char/ckpt.json --temperature 1.0
  node sample.js --start "The quick brown" --max_new_tokens 50
      `);
      process.exit(0);
    }
  }
  
  return options;
}

/**
 * Quick test function to verify model loading
 */
function testModelLoading(modelPath, dataDir) {
  console.log('Testing model loading...');
  
  try {
    // Load tokenizer
    const tokenizerPath = path.join(dataDir, 'tokenizer.json');
    const tokenizer = loadTokenizer(tokenizerPath);
    console.log(`✓ Tokenizer loaded: vocab_size=${tokenizer.vocabSize}`);
    
    // Test encoding/decoding
    const testText = 'Hello';
    const encoded = tokenizer.encode(testText);
    const decoded = tokenizer.decode(encoded);
    console.log(`✓ Tokenizer test: "${testText}" -> [${encoded.join(',')}] -> "${decoded}"`);
    
    // Check if model exists
    if (fs.existsSync(modelPath)) {
      const checkpoint = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
      console.log(`✓ Model checkpoint found: ${Object.keys(checkpoint.model).length} parameter groups`);
      console.log(`✓ Model config:`, checkpoint.config);
    } else {
      console.log(`⚠ Model checkpoint not found: ${modelPath}`);
      console.log('Available files in output directory:');
      const outDir = path.dirname(modelPath);
      if (fs.existsSync(outDir)) {
        const files = fs.readdirSync(outDir);
        files.forEach(file => console.log(`  - ${file}`));
      }
    }
    
  } catch (error) {
    console.error('✗ Test failed:', error.message);
  }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const options = parseArgs();
    
    console.log('Sampling Configuration:');
    console.log(`  Model path: ${options.modelPath}`);
    console.log(`  Data directory: ${options.dataDir}`);
    console.log(`  Number of samples: ${options.numSamples}`);
    console.log(`  Max new tokens: ${options.maxNewTokens}`);
    console.log(`  Temperature: ${options.temperature}`);
    console.log(`  Starting text: "${options.start}"`);
    console.log(`  Seed: ${options.seed}`);
    console.log('');
    
    // Check if data directory exists
    if (!fs.existsSync(options.dataDir)) {
      console.error(`Data directory not found: ${options.dataDir}`);
      process.exit(1);
    }
    
    // If model doesn't exist, run a test to show what's available
    if (!fs.existsSync(options.modelPath)) {
      console.log(`Model checkpoint not found: ${options.modelPath}`);
      testModelLoading(options.modelPath, options.dataDir);
      console.log('\nTip: Train a model first using: node train.js');
      process.exit(1);
    }
    
    // Generate samples
    sampleFromModel(
      options.modelPath,
      options.dataDir,
      options.numSamples,
      options.maxNewTokens,
      options.temperature,
      options.start,
      options.seed
    );
    
  } catch (error) {
    console.error('Sampling failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Export for use as module
export { sampleFromModel, testModelLoading };