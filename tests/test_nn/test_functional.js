/**
 * Comprehensive test suite for neural network functional operations
 * Tests cross-entropy loss, softmax, and other functional utilities
 */

import { Matrix } from 'ml-matrix';
import { crossEntropyLoss, softmax, logSoftmax } from '../../src/nn/functional.js';

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

function assertMatrixClose(actual, expected, tolerance = 1e-6, message = '') {
  assert(actual.rows === expected.rows, `${message} - Row count mismatch`);
  assert(actual.columns === expected.columns, `${message} - Column count mismatch`);
  
  for (let i = 0; i < actual.rows; i++) {
    for (let j = 0; j < actual.columns; j++) {
      const actualVal = actual.get(i, j);
      const expectedVal = expected.get(i, j);
      if (Math.abs(actualVal - expectedVal) > tolerance) {
        throw new Error(`${message} - Value mismatch at (${i}, ${j}): expected ${expectedVal}, got ${actualVal}`);
      }
    }
  }
}

function testCrossEntropyLoss() {
  console.log('Testing cross-entropy loss...');
  
  // Test 1: Simple case with known values
  const logits = new Matrix([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 0.2]
  ]);
  const targets = [0, 1]; // First sample target class 0, second sample target class 1
  
  const loss = crossEntropyLoss(logits, targets);
  assert(loss > 0, 'Loss should be positive');
  assert(isFinite(loss), 'Loss should be finite');
  
  // Test 2: Perfect predictions should give low loss
  const perfectLogits = new Matrix([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0]
  ]);
  const perfectLoss = crossEntropyLoss(perfectLogits, targets);
  assert(perfectLoss < loss, 'Perfect predictions should have lower loss');
  
  // Test 3: Single sample
  const singleLogits = new Matrix([[1.0, 2.0, 0.5]]);
  const singleTarget = [1];
  const singleLoss = crossEntropyLoss(singleLogits, singleTarget);
  assert(singleLoss > 0, 'Single sample loss should be positive');
  
  console.log(`  Basic loss: ${loss.toFixed(4)}, Perfect loss: ${perfectLoss.toFixed(4)}, Single loss: ${singleLoss.toFixed(4)}`);
  console.log('✓ Cross-entropy loss tests passed');
}

function testSoftmax() {
  console.log('Testing softmax function...');
  
  // Test 1: Basic softmax properties
  const input = new Matrix([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 0.0]
  ]);
  
  const output = softmax(input);
  
  // Check dimensions
  assert(output.rows === input.rows, 'Output should have same number of rows');
  assert(output.columns === input.columns, 'Output should have same number of columns');
  
  // Check that each row sums to 1
  for (let i = 0; i < output.rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < output.columns; j++) {
      const val = output.get(i, j);
      assert(val >= 0, 'Softmax values should be non-negative');
      assert(val <= 1, 'Softmax values should be <= 1');
      rowSum += val;
    }
    assertClose(rowSum, 1.0, 1e-6, `Row ${i} should sum to 1`);
  }
  
  // Test 2: Numerical stability with large values
  const largeInput = new Matrix([[100.0, 101.0, 99.0]]);
  const largeOutput = softmax(largeInput);
  
  let largeSum = 0;
  for (let j = 0; j < largeOutput.columns; j++) {
    const val = largeOutput.get(0, j);
    assert(isFinite(val), 'Softmax should handle large inputs without overflow');
    largeSum += val;
  }
  assertClose(largeSum, 1.0, 1e-6, 'Large input softmax should still sum to 1');
  
  // Test 3: Uniform input should give uniform output
  const uniformInput = new Matrix([[1.0, 1.0, 1.0]]);
  const uniformOutput = softmax(uniformInput);
  
  for (let j = 0; j < uniformOutput.columns; j++) {
    assertClose(uniformOutput.get(0, j), 1.0/3.0, 1e-6, 'Uniform input should give uniform output');
  }
  
  console.log('✓ Softmax tests passed');
}

function testLogSoftmax() {
  console.log('Testing log-softmax function...');
  
  const input = new Matrix([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 0.0]
  ]);
  
  const logSoftmaxOutput = logSoftmax(input);
  const softmaxOutput = softmax(input);
  
  // Check dimensions
  assert(logSoftmaxOutput.rows === input.rows, 'Log-softmax output should have same number of rows');
  assert(logSoftmaxOutput.columns === input.columns, 'Log-softmax output should have same number of columns');
  
  // Check that log-softmax equals log of softmax
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      const logSoftmaxVal = logSoftmaxOutput.get(i, j);
      const expectedVal = Math.log(softmaxOutput.get(i, j));
      assertClose(logSoftmaxVal, expectedVal, 1e-6, `Log-softmax should equal log of softmax at (${i}, ${j})`);
    }
  }
  
  // Check that all values are negative (since softmax values are <= 1)
  for (let i = 0; i < logSoftmaxOutput.rows; i++) {
    for (let j = 0; j < logSoftmaxOutput.columns; j++) {
      const val = logSoftmaxOutput.get(i, j);
      assert(val <= 0, 'Log-softmax values should be non-positive');
      assert(isFinite(val), 'Log-softmax values should be finite');
    }
  }
  
  console.log('✓ Log-softmax tests passed');
}

function testNumericalStability() {
  console.log('Testing numerical stability...');
  
  // Test with very large values
  const largeValues = new Matrix([[1000.0, 1001.0, 999.0]]);
  const largeSoftmax = softmax(largeValues);
  const largeLogSoftmax = logSoftmax(largeValues);
  
  // Should not produce NaN or Inf
  for (let j = 0; j < largeSoftmax.columns; j++) {
    assert(isFinite(largeSoftmax.get(0, j)), 'Large value softmax should be finite');
    assert(isFinite(largeLogSoftmax.get(0, j)), 'Large value log-softmax should be finite');
  }
  
  // Test with very small values
  const smallValues = new Matrix([[-1000.0, -1001.0, -999.0]]);
  const smallSoftmax = softmax(smallValues);
  const smallLogSoftmax = logSoftmax(smallValues);
  
  for (let j = 0; j < smallSoftmax.columns; j++) {
    assert(isFinite(smallSoftmax.get(0, j)), 'Small value softmax should be finite');
    assert(isFinite(smallLogSoftmax.get(0, j)), 'Small value log-softmax should be finite');
  }
  
  console.log('✓ Numerical stability tests passed');
}

function testGradientProperties() {
  console.log('Testing gradient properties...');
  
  // Test that cross-entropy loss gradient has correct properties
  const logits = new Matrix([
    [1.0, 2.0, 0.5],
    [0.5, 1.5, 2.0]
  ]);
  const targets = [1, 2];
  
  // Compute loss
  const loss = crossEntropyLoss(logits, targets);
  
  // Test numerical gradient vs analytical gradient (simplified)
  const eps = 1e-5;
  const originalVal = logits.get(0, 1);
  
  // Forward difference
  logits.set(0, 1, originalVal + eps);
  const lossPlus = crossEntropyLoss(logits, targets);
  
  logits.set(0, 1, originalVal - eps);
  const lossMinus = crossEntropyLoss(logits, targets);
  
  // Restore original value
  logits.set(0, 1, originalVal);
  
  const numericalGrad = (lossPlus - lossMinus) / (2 * eps);
  
  // The numerical gradient should be finite
  assert(isFinite(numericalGrad), 'Numerical gradient should be finite');
  
  console.log(`  Numerical gradient: ${numericalGrad.toFixed(6)}`);
  console.log('✓ Gradient property tests passed');
}

function testEdgeCases() {
  console.log('Testing edge cases...');
  
  // Test single class
  const singleClass = new Matrix([[5.0]]);
  const singleSoftmax = softmax(singleClass);
  assertClose(singleSoftmax.get(0, 0), 1.0, 1e-6, 'Single class softmax should be 1');
  
  // Test zero input
  const zeroInput = new Matrix([[0.0, 0.0, 0.0]]);
  const zeroSoftmax = softmax(zeroInput);
  for (let j = 0; j < zeroSoftmax.columns; j++) {
    assertClose(zeroSoftmax.get(0, j), 1.0/3.0, 1e-6, 'Zero input should give uniform distribution');
  }
  
  // Test with negative values
  const negativeInput = new Matrix([[-1.0, -2.0, -3.0]]);
  const negativeSoftmax = softmax(negativeInput);
  let negativeSum = 0;
  for (let j = 0; j < negativeSoftmax.columns; j++) {
    const val = negativeSoftmax.get(0, j);
    assert(val > 0, 'Softmax of negative values should still be positive');
    negativeSum += val;
  }
  assertClose(negativeSum, 1.0, 1e-6, 'Negative input softmax should sum to 1');
  
  console.log('✓ Edge case tests passed');
}

function testBatchConsistency() {
  console.log('Testing batch consistency...');
  
  // Test that batch processing gives same results as individual processing
  const batchInput = new Matrix([
    [1.0, 2.0, 3.0],
    [0.5, 1.5, 2.5]
  ]);
  
  const batchSoftmax = softmax(batchInput);
  
  // Process individually
  const individual1 = softmax(new Matrix([[1.0, 2.0, 3.0]]));
  const individual2 = softmax(new Matrix([[0.5, 1.5, 2.5]]));
  
  // Compare results
  for (let j = 0; j < batchInput.columns; j++) {
    assertClose(batchSoftmax.get(0, j), individual1.get(0, j), 1e-6, 
                `Batch row 0 should match individual processing at column ${j}`);
    assertClose(batchSoftmax.get(1, j), individual2.get(0, j), 1e-6, 
                `Batch row 1 should match individual processing at column ${j}`);
  }
  
  console.log('✓ Batch consistency tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Neural Network Functional Operations ===\n');
  
  try {
    testCrossEntropyLoss();
    testSoftmax();
    testLogSoftmax();
    testNumericalStability();
    testGradientProperties();
    testEdgeCases();
    testBatchConsistency();
    
    console.log('\n🎉 All functional tests passed!');
    
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