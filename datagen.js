#!/usr/bin/env node

/**
 * Data Generation Script for numpyGPT JavaScript Implementation
 * JavaScript equivalent of datagen.py
 * 
 * This script:
 * 1. Reads input text file
 * 2. Creates and trains a tokenizer (char/word/BPE)
 * 3. Encodes the text into token indices
 * 4. Splits data into train/validation sets
 * 5. Saves binary data files and tokenizer state
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { CharTokenizer } from './src/tokenizer/char_level.js';
import { WordTokenizer } from './src/tokenizer/word_level.js';
import { BPETokenizer } from './src/tokenizer/bpe.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Prepare training data from text file
 * @param {string} inputFile - Path to input text file
 * @param {string} outputDir - Output directory for processed data
 * @param {string} tokenizerType - Type of tokenizer ('char', 'word', 'bpe')
 * @param {number} trainSplit - Fraction of data for training (0.0-1.0)
 * @param {number} minFreq - Minimum frequency for word tokenizer
 * @param {number} maxVocabSize - Maximum vocabulary size
 * @returns {Object} - Trained tokenizer
 */
function prepareData(inputFile, outputDir, tokenizerType = 'char', trainSplit = 0.9, 
                    minFreq = 1, maxVocabSize = 1000) {
  console.log(`Reading ${inputFile}...`);
  
  // Read input text file
  if (!fs.existsSync(inputFile)) {
    throw new Error(`Input file not found: ${inputFile}`);
  }
  
  const text = fs.readFileSync(inputFile, 'utf-8');
  console.log(`Data length: ${text.length.toLocaleString()} characters`);
  
  // Create tokenizer based on type
  let tokenizer;
  if (tokenizerType === 'char') {
    tokenizer = new CharTokenizer();
  } else if (tokenizerType === 'word') {
    tokenizer = new WordTokenizer(minFreq, maxVocabSize);
  } else if (tokenizerType === 'bpe') {
    tokenizer = new BPETokenizer(maxVocabSize);
  } else {
    throw new Error(`Unknown tokenizer type: ${tokenizerType}`);
  }
  
  // Build vocabulary
  console.log('Building vocabulary...');
  tokenizer.buildVocab(text);
  console.log(`Vocab size: ${tokenizer.vocabSize}`);
  
  // Encode text to token indices
  console.log('Encoding text...');
  const encoded = tokenizer.encode(text);
  
  // Convert to Uint16Array for efficient storage
  const data = new Uint16Array(encoded);
  
  // Split into train/validation sets
  const splitIdx = Math.floor(data.length * trainSplit);
  const trainData = data.slice(0, splitIdx);
  const valData = data.slice(splitIdx);
  
  console.log(`Train: ${trainData.length.toLocaleString()}, Val: ${valData.length.toLocaleString()}`);
  
  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Save binary data files
  console.log('Saving data files...');
  fs.writeFileSync(path.join(outputDir, 'train.bin'), Buffer.from(trainData.buffer));
  fs.writeFileSync(path.join(outputDir, 'val.bin'), Buffer.from(valData.buffer));
  
  // Save tokenizer state as JSON
  const tokenizerState = {
    type: tokenizerType,
    state: tokenizer.getState()
  };
  fs.writeFileSync(
    path.join(outputDir, 'tokenizer.json'), 
    JSON.stringify(tokenizerState, null, 2)
  );
  
  // Save metadata
  const metadata = {
    inputFile,
    tokenizerType,
    vocabSize: tokenizer.vocabSize,
    trainSplit,
    trainSize: trainData.length,
    valSize: valData.length,
    totalSize: data.length,
    textLength: text.length,
    createdAt: new Date().toISOString()
  };
  fs.writeFileSync(
    path.join(outputDir, 'metadata.json'),
    JSON.stringify(metadata, null, 2)
  );
  
  console.log('Done.');
  return tokenizer;
}

/**
 * Load tokenizer from saved state
 * @param {string} tokenizerPath - Path to tokenizer.json file
 * @returns {Object} - Loaded tokenizer
 */
function loadTokenizer(tokenizerPath) {
  if (!fs.existsSync(tokenizerPath)) {
    throw new Error(`Tokenizer file not found: ${tokenizerPath}`);
  }
  
  const tokenizerData = JSON.parse(fs.readFileSync(tokenizerPath, 'utf-8'));
  const { type, state } = tokenizerData;
  
  let tokenizer;
  if (type === 'char') {
    tokenizer = new CharTokenizer();
  } else if (type === 'word') {
    tokenizer = new WordTokenizer();
  } else if (type === 'bpe') {
    tokenizer = new BPETokenizer();
  } else {
    throw new Error(`Unknown tokenizer type: ${type}`);
  }
  
  tokenizer.loadState(state);
  return tokenizer;
}

/**
 * Load binary data file
 * @param {string} dataPath - Path to .bin file
 * @returns {Uint16Array} - Loaded data
 */
function loadData(dataPath) {
  if (!fs.existsSync(dataPath)) {
    throw new Error(`Data file not found: ${dataPath}`);
  }
  
  const buffer = fs.readFileSync(dataPath);
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
}

/**
 * Parse command line arguments
 * @returns {Object} - Parsed arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    inputFile: 'data/shakespeare.txt',
    outputDir: 'data/shakespeare_char',
    tokenizerType: 'char',
    trainSplit: 0.9,
    minFreq: 1,
    maxVocabSize: 1000
  };
  
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key === 'input_file') options.inputFile = value;
    else if (key === 'output_dir') options.outputDir = value;
    else if (key === 'tokenizer_type') options.tokenizerType = value;
    else if (key === 'train_split') options.trainSplit = parseFloat(value);
    else if (key === 'min_freq') options.minFreq = parseInt(value);
    else if (key === 'max_vocab_size') options.maxVocabSize = parseInt(value);
    else if (key === 'help') {
      console.log(`
Usage: node datagen.js [options]

Options:
  --input_file <path>        Input text file (default: data/shakespeare.txt)
  --output_dir <path>        Output directory (default: data/shakespeare_char)
  --tokenizer_type <type>    Tokenizer type: char, word, bpe (default: char)
  --train_split <float>      Train/val split ratio (default: 0.9)
  --min_freq <int>           Minimum word frequency (default: 1)
  --max_vocab_size <int>     Maximum vocabulary size (default: 1000)
  --help                     Show this help message

Examples:
  node datagen.js --input_file data/shakespeare.txt --tokenizer_type char
  node datagen.js --input_file data/shakespeare.txt --tokenizer_type bpe --max_vocab_size 2000
  node datagen.js --input_file data/shakespeare.txt --tokenizer_type word --min_freq 2
      `);
      process.exit(0);
    }
  }
  
  // Validate tokenizer type
  if (!['char', 'word', 'bpe'].includes(options.tokenizerType)) {
    throw new Error(`Invalid tokenizer type: ${options.tokenizerType}. Must be one of: char, word, bpe`);
  }
  
  // Validate train split
  if (options.trainSplit < 0 || options.trainSplit > 1) {
    throw new Error(`Invalid train split: ${options.trainSplit}. Must be between 0 and 1`);
  }
  
  return options;
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const options = parseArgs();
    
    console.log('Data Generation Configuration:');
    console.log(`  Input file: ${options.inputFile}`);
    console.log(`  Output directory: ${options.outputDir}`);
    console.log(`  Tokenizer type: ${options.tokenizerType}`);
    console.log(`  Train split: ${options.trainSplit}`);
    console.log(`  Min frequency: ${options.minFreq}`);
    console.log(`  Max vocab size: ${options.maxVocabSize}`);
    console.log('');
    
    const tokenizer = prepareData(
      options.inputFile,
      options.outputDir,
      options.tokenizerType,
      options.trainSplit,
      options.minFreq,
      options.maxVocabSize
    );
    
    console.log('\nData preparation completed successfully!');
    console.log(`Files saved to: ${options.outputDir}`);
    console.log('  - train.bin (training data)');
    console.log('  - val.bin (validation data)');
    console.log('  - tokenizer.json (tokenizer state)');
    console.log('  - metadata.json (dataset metadata)');
    
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Export functions for use as module
export { prepareData, loadTokenizer, loadData };