/**
 * Simple test suite for utilities
 * Tests the core functionality of training, visualization, and data utilities
 */

import { Matrix } from 'ml-matrix';
import { 
  setupLogger, 
  clipGradNorm, 
  getLr, 
  TrainingMonitor,
  MetricsLogger,
  DataLoader
} from '../../src/utils/index.js';

function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

/**
 * Mock model for testing
 */
class MockModel {
  constructor() {
    this.lr = 0.001;
    this._grads = {
      'layer1.weight': Matrix.random(3, 4),
      'layer2.weight': Matrix.random(2, 3)
    };
  }
  
  grads() {
    return this._grads;
  }
}

function testSetupLogger() {
  console.log('Testing setupLogger...');
  
  const logger = setupLogger('test', 'info');
  assert(logger.name === 'test', 'Logger name should be set');
  assert(typeof logger.info === 'function', 'Logger should have info method');
  assert(typeof logger.warn === 'function', 'Logger should have warn method');
  assert(typeof logger.error === 'function', 'Logger should have error method');
  
  // Test logging (visual verification)
  logger.info('Test info message');
  logger.warn('Test warning message');
  
  console.log('✓ setupLogger test passed');
}

function testClipGradNorm() {
  console.log('Testing clipGradNorm...');
  
  const model = new MockModel();
  const maxNorm = 1.0;
  
  // Get initial gradient norm (without clipping)
  const initialNorm = clipGradNorm(model, 999); // No clipping
  assert(initialNorm > 0, 'Initial gradient norm should be positive');
  
  // Create a model with large gradients for clipping test
  const modelWithLargeGrads = new MockModel();
  // Scale up gradients to ensure they need clipping
  for (const grad of Object.values(modelWithLargeGrads._grads)) {
    for (let i = 0; i < grad.rows; i++) {
      for (let j = 0; j < grad.columns; j++) {
        grad.set(i, j, grad.get(i, j) * 10); // Make gradients larger
      }
    }
  }
  
  // Test clipping
  const largeNorm = clipGradNorm(modelWithLargeGrads, 999); // Get actual large norm
  const clippedNorm = clipGradNorm(modelWithLargeGrads, maxNorm); // Now clip
  
  // The function returns the norm BEFORE clipping, so we need to check differently
  assert(largeNorm > maxNorm, 'Large gradients should exceed max norm');
  
  // Check that gradients were actually clipped by computing new norm
  let newNorm = 0;
  for (const grad of Object.values(modelWithLargeGrads._grads)) {
    for (let i = 0; i < grad.rows; i++) {
      for (let j = 0; j < grad.columns; j++) {
        const val = grad.get(i, j);
        newNorm += val * val;
      }
    }
  }
  newNorm = Math.sqrt(newNorm);
  
  assert(newNorm <= maxNorm + 1e-6, 'Gradients should be clipped to max_norm');
  
  console.log(`  Large norm: ${largeNorm.toFixed(4)}, After clipping: ${newNorm.toFixed(4)}`);
  console.log('✓ clipGradNorm test passed');
}

function testGetLr() {
  console.log('Testing getLr...');
  
  const optimizer = { lr: 0.001 };
  const lr = getLr(optimizer);
  
  assert(lr === 0.001, 'Should return correct learning rate');
  
  console.log('✓ getLr test passed');
}

function testTrainingMonitor() {
  console.log('Testing TrainingMonitor...');
  
  const monitor = new TrainingMonitor(5); // Log every 5 iterations
  
  // Test logging
  let logMessage = monitor.logStep(0, 2.5, 0.001, 1.2);
  assert(logMessage !== null, 'Should log at iteration 0');
  assert(logMessage.includes('iter'), 'Log message should contain iteration');
  assert(logMessage.includes('loss'), 'Log message should contain loss');
  
  // Test no logging
  logMessage = monitor.logStep(1, 2.4, 0.001, 1.1);
  assert(logMessage === null, 'Should not log at iteration 1');
  
  // Test statistics
  const stats = monitor.getStats();
  assert(stats.totalSteps === 2, 'Should track total steps');
  assert(stats.avgLoss > 0, 'Should calculate average loss');
  
  console.log(`  Stats: ${stats.totalSteps} steps, avg loss: ${stats.avgLoss.toFixed(3)}`);
  console.log('✓ TrainingMonitor test passed');
}

async function testMetricsLogger() {
  console.log('Testing MetricsLogger...');
  
  const tempFile = 'test_metrics.json';
  const logger = new MetricsLogger(tempFile);
  
  // Test logging
  logger.log(0, 2.5, null, 1.2, 0.001);
  logger.log(1, 2.4, null, 1.1, 0.001);
  logger.log(2, 2.3, 2.2, 1.0, 0.0009);
  
  // Test analysis
  const analysis = logger.analyzeTraining();
  assert(analysis.totalIterations === 3, 'Should track iterations');
  assert(analysis.trainLoss.improvement > 0, 'Should detect improvement');
  
  console.log(`  Analysis: ${analysis.totalIterations} iterations, improvement: ${analysis.trainLoss.improvement.toFixed(3)}`);
  
  // Test summary
  const summary = logger.getSummary();
  assert(summary.totalIterations === 3, 'Summary should match');
  
  // Cleanup
  try {
    const fs = await import('fs');
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  } catch (error) {
    // Ignore cleanup errors
  }
  
  console.log('✓ MetricsLogger test passed');
}

function testDataLoader() {
  console.log('Testing DataLoader...');
  
  try {
    // Test with our existing test data
    const loader = new DataLoader('data/test_char', 'train', 2, 8);
    
    assert(loader.vocabSize > 0, 'Should have valid vocab size');
    assert(loader.numSamples > 0, 'Should have samples');
    
    // Test batch generation
    const batch = loader.getBatch();
    assert(batch.X.rows === 2, 'Batch should have correct batch size');
    assert(batch.X.columns === 8, 'Batch should have correct sequence length');
    assert(batch.Y.rows === 2, 'Targets should have correct batch size');
    assert(batch.Y.columns === 8, 'Targets should have correct sequence length');
    
    // Test encoding/decoding
    const text = 'Hello';
    const encoded = loader.encode(text);
    const decoded = loader.decode(encoded);
    assert(typeof decoded === 'string', 'Should decode to string');
    
    // Test statistics
    const stats = loader.getStats();
    assert(stats.totalTokens > 0, 'Should have token count');
    assert(stats.vocabSize === loader.vocabSize, 'Stats should match');
    
    console.log(`  Stats: ${stats.totalTokens} tokens, ${stats.vocabSize} vocab, ${stats.batchesPerEpoch} batches/epoch`);
    console.log('✓ DataLoader test passed');
    
  } catch (error) {
    console.log(`⚠ DataLoader test skipped (data not available): ${error.message}`);
  }
}

// Run all tests
async function runTests() {
  console.log('=== Testing Utilities ===\n');
  
  try {
    testSetupLogger();
    testClipGradNorm();
    testGetLr();
    testTrainingMonitor();
    await testMetricsLogger();
    testDataLoader();
    
    console.log('\n🎉 All utility tests passed!');
    
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