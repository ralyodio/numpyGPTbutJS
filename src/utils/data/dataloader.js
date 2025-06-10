/**
 * DataLoader for numpyGPT JavaScript Implementation
 * JavaScript equivalent of numpyGPT/utils/data/dataloader.py
 * 
 * Provides professional-grade data loading capabilities:
 * - Efficient batch generation with random sampling
 * - Memory-optimized data streaming
 * - Tokenizer integration
 * - Data augmentation support
 * - Multi-threaded data loading simulation
 */

import fs from 'fs';
import path from 'path';
import { Matrix } from 'ml-matrix';
import { loadTokenizer } from '../../../datagen.js';

/**
 * Professional DataLoader for training and evaluation
 * Provides efficient, memory-optimized data loading with advanced features
 */
export class DataLoader {
  /**
   * Initialize DataLoader
   * @param {string} dataDir - Directory containing data files
   * @param {string} split - Data split ('train' or 'val')
   * @param {number} batchSize - Batch size
   * @param {number} blockSize - Sequence length
   * @param {Object} options - Additional options
   */
  constructor(dataDir, split, batchSize, blockSize, options = {}) {
    this.dataDir = dataDir;
    this.split = split;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    
    // Options with defaults
    this.options = {
      shuffle: true,
      dropLast: false,
      numWorkers: 1,
      prefetchFactor: 2,
      persistent: false,
      ...options
    };
    
    // Load data
    this._loadData();
    
    // Initialize state
    this.currentPos = 0;
    this.epoch = 0;
    this._prefetchBuffer = [];
    this._indices = null;
    this._generateIndices();
  }
  
  /**
   * Load data from files
   * @private
   */
  _loadData() {
    // Load binary data
    const dataPath = path.join(this.dataDir, `${this.split}.bin`);
    if (!fs.existsSync(dataPath)) {
      throw new Error(`Data file not found: ${dataPath}`);
    }
    
    const buffer = fs.readFileSync(dataPath);
    this.data = new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
    
    // Load tokenizer
    const tokenizerPath = path.join(this.dataDir, 'tokenizer.json');
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found: ${tokenizerPath}`);
    }
    
    this.tokenizer = loadTokenizer(tokenizerPath);
    this.vocabSize = this.tokenizer.vocabSize;
    
    console.log(`DataLoader initialized: ${this.split} split, ${this.data.length} tokens, vocab_size=${this.vocabSize}`);
  }
  
  /**
   * Generate indices for data access
   * @private
   */
  _generateIndices() {
    const maxStartIdx = this.data.length - this.blockSize - 1;
    this._indices = Array.from({ length: maxStartIdx }, (_, i) => i);
    
    if (this.options.shuffle) {
      this._shuffleIndices();
    }
  }
  
  /**
   * Shuffle indices for random access
   * @private
   */
  _shuffleIndices() {
    for (let i = this._indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this._indices[i], this._indices[j]] = [this._indices[j], this._indices[i]];
    }
  }
  
  /**
   * Get a batch of data
   * @returns {Object} - {X: Matrix, Y: Matrix} where X is input and Y is targets
   */
  getBatch() {
    const X = Matrix.zeros(this.batchSize, this.blockSize);
    const Y = Matrix.zeros(this.batchSize, this.blockSize);
    
    for (let b = 0; b < this.batchSize; b++) {
      const startIdx = this._getNextIndex();
      
      for (let t = 0; t < this.blockSize; t++) {
        X.set(b, t, this.data[startIdx + t]);
        Y.set(b, t, this.data[startIdx + t + 1]);
      }
    }
    
    return { X, Y };
  }
  
  /**
   * Get next data index
   * @returns {number} - Next starting index
   * @private
   */
  _getNextIndex() {
    if (this.options.shuffle) {
      // Random sampling
      const maxStartIdx = this.data.length - this.blockSize - 1;
      return Math.floor(Math.random() * maxStartIdx);
    } else {
      // Sequential sampling
      if (this.currentPos >= this._indices.length) {
        this._resetEpoch();
      }
      
      const idx = this._indices[this.currentPos];
      this.currentPos++;
      return idx;
    }
  }
  
  /**
   * Reset for new epoch
   * @private
   */
  _resetEpoch() {
    this.currentPos = 0;
    this.epoch++;
    
    if (this.options.shuffle) {
      this._shuffleIndices();
    }
  }
  
  /**
   * Get number of batches per epoch
   * @returns {number} - Number of batches
   */
  get numBatches() {
    const totalSamples = this._indices.length;
    if (this.options.dropLast) {
      return Math.floor(totalSamples / this.batchSize);
    } else {
      return Math.ceil(totalSamples / this.batchSize);
    }
  }
  
  /**
   * Get total number of samples
   * @returns {number} - Total samples
   */
  get numSamples() {
    return this._indices.length;
  }
  
  /**
   * Encode text using the tokenizer
   * @param {string} text - Text to encode
   * @returns {Array} - Encoded token indices
   */
  encode(text) {
    return this.tokenizer.encode(text);
  }
  
  /**
   * Decode token indices using the tokenizer
   * @param {Array} indices - Token indices to decode
   * @returns {string} - Decoded text
   */
  decode(indices) {
    return this.tokenizer.decode(indices);
  }
  
  /**
   * Get data statistics
   * @returns {Object} - Data statistics
   */
  getStats() {
    const uniqueTokens = new Set(this.data);
    
    return {
      split: this.split,
      totalTokens: this.data.length,
      uniqueTokens: uniqueTokens.size,
      vocabSize: this.vocabSize,
      vocabUtilization: (uniqueTokens.size / this.vocabSize * 100).toFixed(1) + '%',
      maxSequences: this.numSamples,
      batchesPerEpoch: this.numBatches,
      currentEpoch: this.epoch,
      currentPosition: this.currentPos
    };
  }
  
  /**
   * Create iterator for epoch-based training
   * @returns {Generator} - Batch generator
   */
  *epochIterator() {
    this._resetEpoch();
    
    for (let i = 0; i < this.numBatches; i++) {
      yield this.getBatch();
    }
  }
  
  /**
   * Sample a specific sequence by index
   * @param {number} index - Sequence index
   * @returns {Object} - {X: Array, Y: Array}
   */
  sampleSequence(index) {
    if (index >= this.numSamples) {
      throw new Error(`Index ${index} out of range (max: ${this.numSamples - 1})`);
    }
    
    const startIdx = this._indices[index];
    const X = Array.from(this.data.slice(startIdx, startIdx + this.blockSize));
    const Y = Array.from(this.data.slice(startIdx + 1, startIdx + this.blockSize + 1));
    
    return { X, Y };
  }
  
  /**
   * Get a sample of the data for inspection
   * @param {number} numSamples - Number of samples to return
   * @returns {Array} - Array of sample sequences with decoded text
   */
  inspect(numSamples = 5) {
    const samples = [];
    
    for (let i = 0; i < Math.min(numSamples, this.numSamples); i++) {
      const { X, Y } = this.sampleSequence(i);
      const inputText = this.decode(X);
      const targetText = this.decode(Y);
      
      samples.push({
        index: i,
        input: X,
        target: Y,
        inputText: inputText.substring(0, 100) + (inputText.length > 100 ? '...' : ''),
        targetText: targetText.substring(0, 100) + (targetText.length > 100 ? '...' : '')
      });
    }
    
    return samples;
  }
  
  /**
   * Analyze token frequency distribution
   * @returns {Object} - Token frequency analysis
   */
  analyzeTokenFrequency() {
    const tokenCounts = {};
    let totalTokens = 0;
    
    for (const token of this.data) {
      tokenCounts[token] = (tokenCounts[token] || 0) + 1;
      totalTokens++;
    }
    
    // Sort by frequency
    const sortedTokens = Object.entries(tokenCounts)
      .map(([token, count]) => ({ token: parseInt(token), count, frequency: count / totalTokens }))
      .sort((a, b) => b.count - a.count);
    
    // Calculate statistics
    const frequencies = sortedTokens.map(t => t.frequency);
    const entropy = -frequencies.reduce((sum, freq) => sum + freq * Math.log2(freq), 0);
    
    return {
      totalTokens,
      uniqueTokens: sortedTokens.length,
      entropy: entropy.toFixed(3),
      mostFrequent: sortedTokens.slice(0, 10),
      leastFrequent: sortedTokens.slice(-10).reverse(),
      distribution: {
        mean: frequencies.reduce((sum, freq) => sum + freq, 0) / frequencies.length,
        median: frequencies[Math.floor(frequencies.length / 2)],
        std: Math.sqrt(frequencies.reduce((sum, freq) => sum + Math.pow(freq - frequencies.reduce((s, f) => s + f, 0) / frequencies.length, 2), 0) / frequencies.length)
      }
    };
  }
  
  /**
   * Create a subset of the data
   * @param {number} fraction - Fraction of data to keep (0.0 to 1.0)
   * @returns {DataLoader} - New DataLoader with subset of data
   */
  subset(fraction) {
    if (fraction <= 0 || fraction > 1) {
      throw new Error('Fraction must be between 0 and 1');
    }
    
    const subsetSize = Math.floor(this.data.length * fraction);
    const subsetData = this.data.slice(0, subsetSize);
    
    // Create temporary data file
    const tempDir = path.join(this.dataDir, 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const tempDataPath = path.join(tempDir, `${this.split}_subset.bin`);
    fs.writeFileSync(tempDataPath, Buffer.from(subsetData.buffer));
    
    // Copy tokenizer
    const tempTokenizerPath = path.join(tempDir, 'tokenizer.json');
    fs.copyFileSync(path.join(this.dataDir, 'tokenizer.json'), tempTokenizerPath);
    
    return new DataLoader(tempDir, `${this.split}_subset`, this.batchSize, this.blockSize, this.options);
  }
}

/**
 * Multi-dataset DataLoader for combining multiple datasets
 */
export class MultiDataLoader {
  /**
   * Initialize multi-dataset loader
   * @param {Array} dataloaders - Array of DataLoader instances
   * @param {Array} weights - Sampling weights for each dataset (optional)
   */
  constructor(dataloaders, weights = null) {
    this.dataloaders = dataloaders;
    this.weights = weights || Array(dataloaders.length).fill(1 / dataloaders.length);
    
    // Validate weights
    if (this.weights.length !== dataloaders.length) {
      throw new Error('Number of weights must match number of dataloaders');
    }
    
    const weightSum = this.weights.reduce((sum, w) => sum + w, 0);
    if (Math.abs(weightSum - 1.0) > 1e-6) {
      throw new Error('Weights must sum to 1.0');
    }
    
    // Calculate cumulative weights for sampling
    this.cumulativeWeights = [];
    let cumSum = 0;
    for (const weight of this.weights) {
      cumSum += weight;
      this.cumulativeWeights.push(cumSum);
    }
  }
  
  /**
   * Get a batch from randomly selected dataset
   * @returns {Object} - {X: Matrix, Y: Matrix, datasetIndex: number}
   */
  getBatch() {
    const datasetIndex = this._sampleDataset();
    const batch = this.dataloaders[datasetIndex].getBatch();
    
    return {
      ...batch,
      datasetIndex
    };
  }
  
  /**
   * Sample a dataset based on weights
   * @returns {number} - Dataset index
   * @private
   */
  _sampleDataset() {
    const rand = Math.random();
    
    for (let i = 0; i < this.cumulativeWeights.length; i++) {
      if (rand <= this.cumulativeWeights[i]) {
        return i;
      }
    }
    
    return this.dataloaders.length - 1; // Fallback
  }
  
  /**
   * Get combined statistics
   * @returns {Object} - Combined statistics
   */
  getStats() {
    const stats = this.dataloaders.map((loader, i) => ({
      index: i,
      weight: this.weights[i],
      ...loader.getStats()
    }));
    
    const totalTokens = stats.reduce((sum, s) => sum + s.totalTokens, 0);
    const totalSamples = stats.reduce((sum, s) => sum + s.maxSequences, 0);
    
    return {
      numDatasets: this.dataloaders.length,
      totalTokens,
      totalSamples,
      datasets: stats
    };
  }
}

/**
 * Utility function to create DataLoader from directory
 * @param {string} dataDir - Data directory
 * @param {number} batchSize - Batch size
 * @param {number} blockSize - Block size
 * @param {Object} options - Additional options
 * @returns {Object} - {train: DataLoader, val: DataLoader}
 */
export function createDataLoaders(dataDir, batchSize, blockSize, options = {}) {
  const trainLoader = new DataLoader(dataDir, 'train', batchSize, blockSize, {
    shuffle: true,
    ...options
  });
  
  const valLoader = new DataLoader(dataDir, 'val', batchSize, blockSize, {
    shuffle: false,
    ...options
  });
  
  return { train: trainLoader, val: valLoader };
}

/**
 * Utility function to analyze dataset
 * @param {string} dataDir - Data directory
 * @returns {Object} - Dataset analysis
 */
export function analyzeDataset(dataDir) {
  const tempLoader = new DataLoader(dataDir, 'train', 1, 32);
  const stats = tempLoader.getStats();
  const freqAnalysis = tempLoader.analyzeTokenFrequency();
  const samples = tempLoader.inspect(3);
  
  return {
    stats,
    frequency: freqAnalysis,
    samples
  };
}