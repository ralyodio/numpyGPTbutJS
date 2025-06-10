/**
 * Test suite for utility data functions and DataLoader
 * Tests data loading, preprocessing, and batch generation
 */

import { DataLoader, MultiDataLoader, createDataLoaders } from '../../src/utils/data/dataloader.js';
import { Matrix } from 'ml-matrix';
import fs from 'fs';
import path from 'path';
import { CharTokenizer } from '../../src/tokenizer/char_level.js';

function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

function assertClose(actual, expected, tolerance = 1e-6, message = '') {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Assertion failed: ${message}. Expected ${expected}, got ${actual}`);
  }
}

function createTestData() {
  // Create temporary test directory
  const testDir = path.join(process.cwd(), 'test_data_temp');
  if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
  }
  
  // Create sample text data
  const sampleText = "Hello world! This is a test. The quick brown fox jumps over the lazy dog. " +
                    "Machine learning is fascinating. Neural networks can learn complex patterns. " +
                    "JavaScript is a versatile programming language. Testing is important for quality.";
  
  // Create tokenizer
  const tokenizer = new CharTokenizer();
  tokenizer.buildVocab(sampleText);
  
  // Encode text
  const tokens = tokenizer.encode(sampleText, false, false);
  
  // Create binary data files
  const trainTokens = tokens.slice(0, Math.floor(tokens.length * 0.8));
  const valTokens = tokens.slice(Math.floor(tokens.length * 0.8));
  
  // Write binary files
  const trainBuffer = Buffer.from(new Uint16Array(trainTokens).buffer);
  const valBuffer = Buffer.from(new Uint16Array(valTokens).buffer);
  
  fs.writeFileSync(path.join(testDir, 'train.bin'), trainBuffer);
  fs.writeFileSync(path.join(testDir, 'val.bin'), valBuffer);
  
  // Write tokenizer file
  const tokenizerData = {
    type: 'char',
    state: tokenizer.getState()
  };
  fs.writeFileSync(path.join(testDir, 'tokenizer.json'), JSON.stringify(tokenizerData, null, 2));
  
  return { testDir, tokenizer, trainTokens, valTokens };
}

function cleanupTestData(testDir) {
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true, force: true });
  }
}

function testDataLoaderBasic() {
  console.log('Testing DataLoader basic functionality...');
  
  const { testDir, tokenizer } = createTestData();
  
  try {
    const batchSize = 4;
    const blockSize = 8;
    const dataLoader = new DataLoader(testDir, 'train', batchSize, blockSize);
    
    // Test initialization
    assert(dataLoader.batchSize === batchSize, 'DataLoader should store batch size');
    assert(dataLoader.blockSize === blockSize, 'DataLoader should store block size');
    assert(dataLoader.numBatches > 0, 'DataLoader should calculate number of batches');
    assert(dataLoader.vocabSize > 0, 'DataLoader should have vocabulary size');
    
    // Test batch generation
    const batch = dataLoader.getBatch();
    assert(batch.X instanceof Matrix, 'Batch X should be a Matrix');
    assert(batch.Y instanceof Matrix, 'Batch Y should be a Matrix');
    assert(batch.X.rows === batchSize, 'Batch X should have correct batch size');
    assert(batch.X.columns === blockSize, 'Batch X should have correct block size');
    assert(batch.Y.rows === batchSize, 'Batch Y should have correct batch size');
    assert(batch.Y.columns === blockSize, 'Batch Y should have correct block size');
    
    // Test that all values are valid token indices
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < blockSize; j++) {
        const xVal = batch.X.get(i, j);
        const yVal = batch.Y.get(i, j);
        assert(xVal >= 0 && xVal < dataLoader.vocabSize, `X value should be valid token index: ${xVal}`);
        assert(yVal >= 0 && yVal < dataLoader.vocabSize, `Y value should be valid token index: ${yVal}`);
      }
    }
    
    console.log('✓ DataLoader basic functionality tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

function testDataLoaderIteration() {
  console.log('Testing DataLoader iteration...');
  
  const { testDir } = createTestData();
  
  try {
    const batchSize = 2;
    const blockSize = 6;
    const dataLoader = new DataLoader(testDir, 'train', batchSize, blockSize);
    
    // Test multiple batch generation
    const batches = [];
    const numBatchesToTest = Math.min(3, dataLoader.numBatches);
    
    for (let i = 0; i < numBatchesToTest; i++) {
      const batch = dataLoader.getBatch();
      batches.push(batch);
      
      assert(batch.X instanceof Matrix, `Batch ${i} X should be a Matrix`);
      assert(batch.Y instanceof Matrix, `Batch ${i} Y should be a Matrix`);
      assert(batch.X.rows === batchSize, `Batch ${i} should have correct batch size`);
      assert(batch.X.columns === blockSize, `Batch ${i} should have correct block size`);
    }
    
    // Test epoch iterator
    let epochBatchCount = 0;
    for (const batch of dataLoader.epochIterator()) {
      assert(batch.X instanceof Matrix, 'Epoch batch X should be a Matrix');
      assert(batch.Y instanceof Matrix, 'Epoch batch Y should be a Matrix');
      epochBatchCount++;
      if (epochBatchCount >= 2) break; // Don't test all batches
    }
    
    assert(epochBatchCount > 0, 'Epoch iterator should produce batches');
    
    console.log('✓ DataLoader iteration tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

function testDataLoaderEdgeCases() {
  console.log('Testing DataLoader edge cases...');
  
  const { testDir } = createTestData();
  
  try {
    // Test with small batch size
    const smallBatchLoader = new DataLoader(testDir, 'train', 1, 4);
    assert(smallBatchLoader.numBatches > 0, 'Small batch should still produce batches');
    
    const smallBatch = smallBatchLoader.getBatch();
    assert(smallBatch.X instanceof Matrix, 'Small batch X should be a Matrix');
    assert(smallBatch.Y instanceof Matrix, 'Small batch Y should be a Matrix');
    assert(smallBatch.X.rows === 1, 'Small batch should have batch size 1');
    
    // Test with larger block size
    const largeBlockLoader = new DataLoader(testDir, 'train', 2, 16);
    const largeBatch = largeBlockLoader.getBatch();
    assert(largeBatch.X instanceof Matrix, 'Large block batch should work');
    assert(largeBatch.Y instanceof Matrix, 'Large block batch should work');
    assert(largeBatch.X.columns === 16, 'Large block should have correct size');
    
    console.log('✓ DataLoader edge cases tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

function testDataLoaderStatistics() {
  console.log('Testing DataLoader statistics...');
  
  const { testDir } = createTestData();
  
  try {
    const dataLoader = new DataLoader(testDir, 'train', 4, 8);
    
    // Test statistics calculation
    const stats = dataLoader.getStats();
    assert(typeof stats === 'object', 'Stats should be an object');
    assert(typeof stats.totalTokens === 'number', 'Stats should include total tokens');
    assert(typeof stats.vocabSize === 'number', 'Stats should include vocabulary size');
    assert(typeof stats.batchesPerEpoch === 'number', 'Stats should include batches per epoch');
    assert(typeof stats.split === 'string', 'Stats should include split name');
    
    assert(stats.totalTokens > 0, 'Total tokens should be positive');
    assert(stats.vocabSize > 0, 'Vocabulary size should be positive');
    assert(stats.batchesPerEpoch > 0, 'Batches per epoch should be positive');
    assert(stats.split === 'train', 'Split should be train');
    
    // Test frequency analysis
    const freqAnalysis = dataLoader.analyzeTokenFrequency();
    assert(typeof freqAnalysis === 'object', 'Frequency analysis should be an object');
    assert(typeof freqAnalysis.totalTokens === 'number', 'Should include total tokens');
    assert(typeof freqAnalysis.uniqueTokens === 'number', 'Should include unique tokens');
    assert(Array.isArray(freqAnalysis.mostFrequent), 'Should include most frequent tokens');
    
    console.log('✓ DataLoader statistics tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

function testMultiDataLoader() {
  console.log('Testing MultiDataLoader...');
  
  const { testDir: testDir1 } = createTestData();
  const { testDir: testDir2 } = createTestData();
  
  try {
    // Create multiple dataloaders
    const loader1 = new DataLoader(testDir1, 'train', 2, 4);
    const loader2 = new DataLoader(testDir2, 'train', 2, 4);
    
    const multiLoader = new MultiDataLoader([loader1, loader2]);
    
    // Test initialization
    assert(multiLoader.dataloaders.length === 2, 'MultiDataLoader should store all dataloaders');
    assert(Array.isArray(multiLoader.weights), 'MultiDataLoader should have weights');
    assert(multiLoader.weights.length === 2, 'Should have weights for each dataloader');
    
    // Test batch generation
    const batch = multiLoader.getBatch();
    assert(batch.X instanceof Matrix, 'Multi batch X should be a Matrix');
    assert(batch.Y instanceof Matrix, 'Multi batch Y should be a Matrix');
    assert(typeof batch.datasetIndex === 'number', 'Batch should include dataset index');
    assert(batch.datasetIndex >= 0 && batch.datasetIndex < 2, 'Dataset index should be valid');
    
    // Test statistics
    const stats = multiLoader.getStats();
    assert(typeof stats === 'object', 'Multi stats should be an object');
    assert(stats.numDatasets === 2, 'Should report correct number of datasets');
    assert(Array.isArray(stats.datasets), 'Should include individual dataset stats');
    
    console.log('✓ MultiDataLoader tests passed');
  } finally {
    cleanupTestData(testDir1);
    cleanupTestData(testDir2);
  }
}

function testDataLoaderMemoryEfficiency() {
  console.log('Testing DataLoader memory efficiency...');
  
  const { testDir } = createTestData();
  
  try {
    const dataLoader = new DataLoader(testDir, 'train', 8, 12);
    
    // Test that we can generate multiple batches without memory issues
    const startTime = Date.now();
    const numBatches = 20; // Reduced for test efficiency
    
    for (let i = 0; i < numBatches; i++) {
      const batch = dataLoader.getBatch();
      assert(batch.X instanceof Matrix, `Batch ${i} should be valid`);
      assert(batch.Y instanceof Matrix, `Batch ${i} should be valid`);
      
      // Verify batch dimensions
      assert(batch.X.rows === 8, `Batch ${i} should have correct batch size`);
      assert(batch.X.columns === 12, `Batch ${i} should have correct block size`);
    }
    
    const endTime = Date.now();
    const timePerBatch = (endTime - startTime) / numBatches;
    
    // Should be reasonably fast (less than 50ms per batch for file-based loading)
    assert(timePerBatch < 50, 'Batch generation should be efficient');
    
    console.log(`  Generated ${numBatches} batches in ${endTime - startTime}ms (${timePerBatch.toFixed(2)}ms per batch)`);
    console.log('✓ DataLoader memory efficiency tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

function testDataLoaderValidation() {
  console.log('Testing DataLoader validation and utilities...');
  
  const { testDir } = createTestData();
  
  try {
    // Test createDataLoaders utility
    const { train, val } = createDataLoaders(testDir, 4, 8);
    assert(train instanceof DataLoader, 'Train loader should be DataLoader instance');
    assert(val instanceof DataLoader, 'Val loader should be DataLoader instance');
    assert(train.split === 'train', 'Train loader should have train split');
    assert(val.split === 'val', 'Val loader should have val split');
    
    // Test data inspection
    const samples = train.inspect(2);
    assert(Array.isArray(samples), 'Inspect should return array');
    assert(samples.length <= 2, 'Should return requested number of samples');
    
    if (samples.length > 0) {
      const sample = samples[0];
      assert(typeof sample === 'object', 'Sample should be object');
      assert(Array.isArray(sample.input), 'Sample should have input array');
      assert(Array.isArray(sample.target), 'Sample should have target array');
      assert(typeof sample.inputText === 'string', 'Sample should have input text');
      assert(typeof sample.targetText === 'string', 'Sample should have target text');
    }
    
    // Test sequence sampling
    if (train.numSamples > 0) {
      const sequence = train.sampleSequence(0);
      assert(Array.isArray(sequence.X), 'Sequence X should be array');
      assert(Array.isArray(sequence.Y), 'Sequence Y should be array');
      assert(sequence.X.length === train.blockSize, 'Sequence X should have correct length');
      assert(sequence.Y.length === train.blockSize, 'Sequence Y should have correct length');
    }
    
    console.log('✓ DataLoader validation tests passed');
  } finally {
    cleanupTestData(testDir);
  }
}

// Run all tests
function runTests() {
  console.log('=== Testing Utility Data Functions ===\n');
  
  try {
    testDataLoaderBasic();
    testDataLoaderIteration();
    testDataLoaderEdgeCases();
    testDataLoaderStatistics();
    testMultiDataLoader();
    testDataLoaderMemoryEfficiency();
    testDataLoaderValidation();
    
    console.log('\n🎉 All utility data tests passed!');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests();
}

export { runTests };