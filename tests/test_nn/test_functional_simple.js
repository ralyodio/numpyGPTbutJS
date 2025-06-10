import { Matrix } from 'ml-matrix';
import { 
  crossEntropyLoss, 
  softmax, 
  logSoftmax, 
  relu, 
  sigmoid, 
  tanh 
} from '../../src/nn/functional.js';

/**
 * Test suite for Neural Network Functional Operations
 */

// Test utilities
function assertClose(actual, expected, tolerance = 1e-6, message = '') {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(`${message} Expected ${expected}, got ${actual}, diff: ${diff}`);
  }
}

function assertMatrixClose(actual, expected, tolerance = 1e-6, message = '') {
  if (actual.rows !== expected.rows || actual.columns !== expected.columns) {
    throw new Error(`${message} Matrix dimensions don't match`);
  }
  
  for (let i = 0; i < actual.rows; i++) {
    for (let j = 0; j < actual.columns; j++) {
      const actualVal = actual.get(i, j);
      const expectedVal = expected.get(i, j);
      const diff = Math.abs(actualVal - expectedVal);
      if (diff > tolerance) {
        throw new Error(`${message} At position [${i},${j}]: expected ${expectedVal}, got ${actualVal}, diff: ${diff}`);
      }
    }
  }
}

// Test Cross-Entropy Loss
function testCrossEntropyBasic() {
  console.log('Testing cross-entropy loss - basic functionality...');
  
  // Simple 2-class case
  const logits = new Matrix([[1.0, 2.0], [3.0, 1.0]]);
  const targets = [1, 0]; // Second class for first sample, first class for second sample
  
  const loss = crossEntropyLoss(logits, targets);
  
  // Manual calculation:
  // Sample 1: logits=[1,2], target=1, softmax=[0.269, 0.731], loss=-log(0.731)=0.313
  // Sample 2: logits=[3,1], target=0, softmax=[0.881, 0.119], loss=-log(0.881)=0.127
  // Average: (0.313 + 0.127) / 2 = 0.220
  assertClose(loss, 0.220, 1e-3, 'Cross-entropy loss basic test');
  
  console.log('✓ Cross-entropy basic test passed');
}

function testCrossEntropyNumericalStability() {
  console.log('Testing cross-entropy loss - numerical stability...');
  
  // Large logits that could cause overflow
  const logits = new Matrix([[100.0, 101.0], [50.0, 49.0]]);
  const targets = [1, 0];
  
  const loss = crossEntropyLoss(logits, targets);
  
  // Should not be NaN or Infinity
  if (isNaN(loss) || !isFinite(loss)) {
    throw new Error('Cross-entropy loss should be finite with large logits');
  }
  
  // Should be reasonable values - with large logits, the model is very confident
  // so the loss should be small (close to 0 for correct predictions)
  if (loss < 0 || loss > 2.0) {
    throw new Error(`Cross-entropy loss should be reasonable, got ${loss}`);
  }
  
  console.log('✓ Cross-entropy numerical stability test passed');
}

function testCrossEntropyMultiClass() {
  console.log('Testing cross-entropy loss - multi-class...');
  
  // 3-class case
  const logits = new Matrix([
    [1.0, 2.0, 0.5],
    [0.1, 0.2, 2.5],
    [1.5, 1.0, 1.2]
  ]);
  const targets = [1, 2, 0]; // Classes 1, 2, 0 respectively
  
  const loss = crossEntropyLoss(logits, targets);
  
  // Should be a reasonable positive value
  if (loss <= 0 || isNaN(loss)) {
    throw new Error('Multi-class cross-entropy should be positive and finite');
  }
  
  console.log('✓ Cross-entropy multi-class test passed');
}

// Test Softmax
function testSoftmaxBasic() {
  console.log('Testing softmax - basic functionality...');
  
  const X = new Matrix([[1.0, 2.0, 3.0]]);
  const result = softmax(X);
  
  // Check probabilities sum to 1
  const rowSum = result.getRow(0).reduce((sum, val) => sum + val, 0);
  assertClose(rowSum, 1.0, 1e-6, 'Softmax probabilities should sum to 1');
  
  // Check values are positive
  for (let j = 0; j < result.columns; j++) {
    const val = result.get(0, j);
    if (val <= 0) {
      throw new Error('Softmax values should be positive');
    }
  }
  
  // Check monotonicity (larger input -> larger probability)
  const probs = result.getRow(0);
  if (!(probs[0] < probs[1] && probs[1] < probs[2])) {
    throw new Error('Softmax should preserve order');
  }
  
  console.log('✓ Softmax basic test passed');
}

function testSoftmaxMultiRow() {
  console.log('Testing softmax - multiple rows...');
  
  const X = new Matrix([
    [1.0, 2.0],
    [3.0, 1.0]
  ]);
  const result = softmax(X);
  
  // Check each row sums to 1
  for (let i = 0; i < result.rows; i++) {
    const rowSum = result.getRow(i).reduce((sum, val) => sum + val, 0);
    assertClose(rowSum, 1.0, 1e-6, `Row ${i} should sum to 1`);
  }
  
  console.log('✓ Softmax multi-row test passed');
}

function testSoftmaxNumericalStability() {
  console.log('Testing softmax - numerical stability...');
  
  // Large values that could cause overflow
  const X = new Matrix([[100.0, 101.0, 99.0]]);
  const result = softmax(X);
  
  // Should not contain NaN or Infinity
  for (let i = 0; i < result.rows; i++) {
    for (let j = 0; j < result.columns; j++) {
      const val = result.get(i, j);
      if (isNaN(val) || !isFinite(val)) {
        throw new Error('Softmax should handle large inputs without overflow');
      }
    }
  }
  
  // Should still sum to 1
  const rowSum = result.getRow(0).reduce((sum, val) => sum + val, 0);
  assertClose(rowSum, 1.0, 1e-6, 'Softmax with large inputs should sum to 1');
  
  console.log('✓ Softmax numerical stability test passed');
}

// Test Log-Softmax
function testLogSoftmaxBasic() {
  console.log('Testing log-softmax - basic functionality...');
  
  const X = new Matrix([[1.0, 2.0, 3.0]]);
  const logResult = logSoftmax(X);
  const softmaxResult = softmax(X);
  
  // Check that exp(log_softmax) ≈ softmax
  for (let j = 0; j < X.columns; j++) {
    const logVal = logResult.get(0, j);
    const softmaxVal = softmaxResult.get(0, j);
    const expLogVal = Math.exp(logVal);
    assertClose(expLogVal, softmaxVal, 1e-6, `exp(log_softmax) should equal softmax at position ${j}`);
  }
  
  console.log('✓ Log-softmax basic test passed');
}

// Test ReLU
function testReluBasic() {
  console.log('Testing ReLU - basic functionality...');
  
  const X = new Matrix([[-1.0, 0.0, 1.0, 2.0]]);
  const result = relu(X);
  
  const expected = new Matrix([[0.0, 0.0, 1.0, 2.0]]);
  assertMatrixClose(result, expected, 1e-6, 'ReLU basic test');
  
  console.log('✓ ReLU basic test passed');
}

function testReluMultiDimensional() {
  console.log('Testing ReLU - multi-dimensional...');
  
  const X = new Matrix([
    [-2.0, 1.0],
    [0.0, -1.0],
    [3.0, 4.0]
  ]);
  const result = relu(X);
  
  const expected = new Matrix([
    [0.0, 1.0],
    [0.0, 0.0],
    [3.0, 4.0]
  ]);
  assertMatrixClose(result, expected, 1e-6, 'ReLU multi-dimensional test');
  
  console.log('✓ ReLU multi-dimensional test passed');
}

// Test Sigmoid
function testSigmoidBasic() {
  console.log('Testing sigmoid - basic functionality...');
  
  const X = new Matrix([[0.0, 1.0, -1.0]]);
  const result = sigmoid(X);
  
  // Check known values
  assertClose(result.get(0, 0), 0.5, 1e-6, 'Sigmoid(0) should be 0.5');
  assertClose(result.get(0, 1), 1/(1+Math.exp(-1)), 1e-6, 'Sigmoid(1)');
  assertClose(result.get(0, 2), 1/(1+Math.exp(1)), 1e-6, 'Sigmoid(-1)');
  
  // Check range (0, 1)
  for (let j = 0; j < result.columns; j++) {
    const val = result.get(0, j);
    if (val <= 0 || val >= 1) {
      throw new Error('Sigmoid values should be in (0, 1)');
    }
  }
  
  console.log('✓ Sigmoid basic test passed');
}

// Test Tanh
function testTanhBasic() {
  console.log('Testing tanh - basic functionality...');
  
  const X = new Matrix([[0.0, 1.0, -1.0]]);
  const result = tanh(X);
  
  // Check known values
  assertClose(result.get(0, 0), 0.0, 1e-6, 'Tanh(0) should be 0');
  assertClose(result.get(0, 1), Math.tanh(1), 1e-6, 'Tanh(1)');
  assertClose(result.get(0, 2), Math.tanh(-1), 1e-6, 'Tanh(-1)');
  
  // Check range (-1, 1)
  for (let j = 0; j < result.columns; j++) {
    const val = result.get(0, j);
    if (val <= -1 || val >= 1) {
      throw new Error('Tanh values should be in (-1, 1)');
    }
  }
  
  console.log('✓ Tanh basic test passed');
}

// Run all tests
function runAllTests() {
  console.log('=== Testing Neural Network Functional Operations ===\n');
  
  try {
    // Cross-entropy tests
    testCrossEntropyBasic();
    testCrossEntropyNumericalStability();
    testCrossEntropyMultiClass();
    
    // Softmax tests
    testSoftmaxBasic();
    testSoftmaxMultiRow();
    testSoftmaxNumericalStability();
    
    // Log-softmax tests
    testLogSoftmaxBasic();
    
    // Activation function tests
    testReluBasic();
    testReluMultiDimensional();
    testSigmoidBasic();
    testTanhBasic();
    
    console.log('\n🎉 All functional tests passed!');
    return true;
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    return false;
  }
}

// Export for use in other test files
export { runAllTests };

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests();
}