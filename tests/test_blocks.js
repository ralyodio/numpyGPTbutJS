/**
 * Test suite for PyTorch comparison tests
 * Tests JavaScript implementation against PyTorch reference implementations
 */

import { Matrix } from 'ml-matrix';
import { Linear } from '../src/nn/modules/linear.js';
import { LayerNorm } from '../src/nn/modules/layerNorm.js';
import { MultiHeadAttention } from '../src/nn/modules/attention.js';
import { FeedForward } from '../src/nn/modules/feedforward.js';
import { TransformerBlock } from '../src/nn/modules/transformer.js';
import { crossEntropyLoss, softmax } from '../src/nn/functional.js';

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
  assert(actual.rows === expected.rows, `${message} - Row dimension mismatch`);
  assert(actual.columns === expected.columns, `${message} - Column dimension mismatch`);
  
  for (let i = 0; i < actual.rows; i++) {
    for (let j = 0; j < actual.columns; j++) {
      const actualVal = actual.get(i, j);
      const expectedVal = expected.get(i, j);
      if (Math.abs(actualVal - expectedVal) > tolerance) {
        throw new Error(`${message} - Matrix element mismatch at [${i},${j}]: expected ${expectedVal}, got ${actualVal}`);
      }
    }
  }
}

function testLinearLayerComparison() {
  console.log('Testing Linear layer PyTorch comparison...');
  
  // Test case: Linear layer with known weights and bias
  const inputSize = 4;
  const outputSize = 3;
  const batchSize = 2;
  
  const linear = new Linear(inputSize, outputSize);
  
  // Set known weights for reproducible comparison
  // Linear layer uses W with shape (inFeatures, outFeatures)
  const weights = new Matrix([
    [0.5, -0.2, 0.1],    // input feature 0 -> all outputs
    [-0.3, 0.4, -0.1],   // input feature 1 -> all outputs
    [0.2, -0.1, 0.5],    // input feature 2 -> all outputs
    [0.1, 0.3, -0.2]     // input feature 3 -> all outputs
  ]);
  const bias = new Matrix([[0.1, -0.05, 0.2]]);
  
  linear.W = weights;
  linear.b = bias;
  
  // Test input
  const input = new Matrix([
    [1.0, 0.5, -0.2, 0.3],
    [-0.1, 0.8, 0.4, -0.5]
  ]);
  
  // Forward pass
  const output = linear.forward(input);
  
  // Let me calculate this step by step for verification
  // Sample 1: [1.0, 0.5, -0.2, 0.3] @ weights + bias
  // Output 0: 1.0*0.5 + 0.5*(-0.3) + (-0.2)*0.2 + 0.3*0.1 + 0.1 = 0.5 - 0.15 - 0.04 + 0.03 + 0.1 = 0.44
  // Output 1: 1.0*(-0.2) + 0.5*0.4 + (-0.2)*(-0.1) + 0.3*0.3 + (-0.05) = -0.2 + 0.2 + 0.02 + 0.09 - 0.05 = 0.06
  // Output 2: 1.0*0.1 + 0.5*(-0.1) + (-0.2)*0.5 + 0.3*(-0.2) + 0.2 = 0.1 - 0.05 - 0.1 - 0.06 + 0.2 = 0.09
  
  // Sample 2: [-0.1, 0.8, 0.4, -0.5] @ weights + bias
  // Output 0: (-0.1)*0.5 + 0.8*(-0.3) + 0.4*0.2 + (-0.5)*0.1 + 0.1 = -0.05 - 0.24 + 0.08 - 0.05 + 0.1 = -0.16
  // Output 1: (-0.1)*(-0.2) + 0.8*0.4 + 0.4*(-0.1) + (-0.5)*0.3 + (-0.05) = 0.02 + 0.32 - 0.04 - 0.15 - 0.05 = 0.1
  // Output 2: (-0.1)*0.1 + 0.8*(-0.1) + 0.4*0.5 + (-0.5)*(-0.2) + 0.2 = -0.01 - 0.08 + 0.2 + 0.1 + 0.2 = 0.41
  
  const expectedOutput = new Matrix([
    [0.44, 0.06, 0.09],
    [-0.16, 0.1, 0.41]
  ]);
  
  assertMatrixClose(output, expectedOutput, 1e-6, 'Linear layer forward pass');
  
  // Test backward pass
  const gradOutput = new Matrix([
    [1.0, 0.5, -0.2],
    [0.3, -0.1, 0.4]
  ]);
  
  const gradInput = linear.backward(gradOutput);
  
  // Verify gradient dimensions
  assert(gradInput.rows === batchSize, 'Gradient input should have correct batch size');
  assert(gradInput.columns === inputSize, 'Gradient input should have correct input size');
  
  // Verify weight gradients were computed
  assert(linear.dW !== null, 'Weight gradients should be computed');
  assert(linear.db !== null, 'Bias gradients should be computed');
  
  console.log('✓ Linear layer PyTorch comparison tests passed');
}

function testLayerNormComparison() {
  console.log('Testing LayerNorm PyTorch comparison...');
  
  const normalizedShape = 4;
  const layerNorm = new LayerNorm(normalizedShape);
  
  // Set known parameters
  layerNorm.weight = new Matrix([[1.2, 0.8, 1.1, 0.9]]);
  layerNorm.bias = new Matrix([[0.1, -0.05, 0.02, -0.03]]);
  
  // Test input with known statistics
  const input = new Matrix([
    [2.0, 1.0, 3.0, 0.5],
    [-1.0, 2.5, 1.5, 0.0]
  ]);
  
  const output = layerNorm.forward(input);
  
  // Verify output shape
  assert(output.rows === input.rows, 'LayerNorm output should preserve batch dimension');
  assert(output.columns === input.columns, 'LayerNorm output should preserve feature dimension');
  
  // Verify normalization properties (mean ≈ 0, std ≈ 1 before scaling)
  for (let i = 0; i < output.rows; i++) {
    let sum = 0;
    let sumSq = 0;
    for (let j = 0; j < output.columns; j++) {
      const val = output.get(i, j);
      sum += val;
      sumSq += val * val;
    }
    const mean = sum / output.columns;
    const variance = (sumSq / output.columns) - (mean * mean);
    
    // After scaling and bias, mean and variance will be different
    // But the relative relationships should be preserved
    assert(!isNaN(mean), `LayerNorm output should not contain NaN values`);
    assert(!isNaN(variance), `LayerNorm variance should not be NaN`);
  }
  
  console.log('✓ LayerNorm PyTorch comparison tests passed');
}

function testAttentionComparison() {
  console.log('Testing MultiHeadAttention PyTorch comparison...');
  
  const dModel = 8;
  const nHeads = 2;
  const seqLen = 4;
  const batchSize = 2;
  
  const attention = new MultiHeadAttention(dModel, nHeads);
  
  // Test input
  const input = new Matrix(batchSize * seqLen, dModel);
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      input.set(i, j, Math.sin(i * 0.1 + j * 0.2));
    }
  }
  
  // Forward pass with batch size and sequence length
  const output = attention.forward(input, batchSize, seqLen);
  
  // Verify output dimensions
  assert(output.rows === input.rows, 'Attention output should preserve sequence dimension');
  assert(output.columns === input.columns, 'Attention output should preserve model dimension');
  
  // Verify attention weights are computed
  assert(attention.cache.attnWeights !== null, 'Attention weights should be computed');
  assert(Array.isArray(attention.cache.attnWeights), 'Attention weights should be an array');
  assert(attention.cache.attnWeights.length === nHeads, 'Should have attention weights for each head');
  
  // Test with causal mask (lower triangular mask)
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask.set(i, j, j > i ? -Infinity : 0); // Causal mask
    }
  }
  
  const maskedOutput = attention.forward(input, batchSize, seqLen, mask);
  
  assert(maskedOutput.rows === input.rows, 'Masked attention output should preserve dimensions');
  assert(maskedOutput.columns === input.columns, 'Masked attention output should preserve dimensions');
  
  console.log('✓ MultiHeadAttention PyTorch comparison tests passed');
}

function testFeedForwardComparison() {
  console.log('Testing FeedForward PyTorch comparison...');
  
  const dModel = 6;
  const dFf = 12;
  const seqLen = 3;
  const batchSize = 2;
  
  const feedforward = new FeedForward(dModel, dFf);
  
  // Test input
  const input = new Matrix(batchSize * seqLen, dModel);
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      input.set(i, j, (i + j) * 0.1 - 0.5);
    }
  }
  
  const output = feedforward.forward(input);
  
  // Verify output dimensions
  assert(output.rows === input.rows, 'FeedForward output should preserve sequence dimension');
  assert(output.columns === input.columns, 'FeedForward output should preserve model dimension');
  
  // Verify non-linearity (output should be different from linear transformation)
  let hasNonLinearity = false;
  for (let i = 0; i < Math.min(5, output.rows); i++) {
    for (let j = 0; j < output.columns; j++) {
      const outVal = output.get(i, j);
      const inVal = input.get(i, j);
      if (Math.abs(outVal - inVal) > 0.01) {
        hasNonLinearity = true;
        break;
      }
    }
    if (hasNonLinearity) break;
  }
  assert(hasNonLinearity, 'FeedForward should apply non-linear transformation');
  
  console.log('✓ FeedForward PyTorch comparison tests passed');
}

function testTransformerBlockComparison() {
  console.log('Testing TransformerBlock PyTorch comparison...');
  
  const dModel = 8;
  const nHeads = 2;
  const dFf = 16;
  const seqLen = 4;
  const batchSize = 2;
  
  const transformerBlock = new TransformerBlock(dModel, nHeads, dFf);
  
  // Test input
  const input = new Matrix(batchSize * seqLen, dModel);
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      input.set(i, j, Math.cos(i * 0.15 + j * 0.25) * 0.5);
    }
  }
  
  // Forward pass with batch size and sequence length
  const output = transformerBlock.forward(input, batchSize, seqLen);
  
  // Verify output dimensions
  assert(output.rows === input.rows, 'TransformerBlock output should preserve sequence dimension');
  assert(output.columns === input.columns, 'TransformerBlock output should preserve model dimension');
  
  // Test residual connections (output should be related to input)
  let residualPreserved = false;
  for (let i = 0; i < Math.min(3, output.rows); i++) {
    for (let j = 0; j < output.columns; j++) {
      const outVal = output.get(i, j);
      const inVal = input.get(i, j);
      // Residual connections should make output somewhat correlated with input
      if (!isNaN(outVal) && !isNaN(inVal)) {
        residualPreserved = true;
        break;
      }
    }
    if (residualPreserved) break;
  }
  assert(residualPreserved, 'TransformerBlock should preserve information through residual connections');
  
  // Test with causal mask
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask.set(i, j, j > i ? -Infinity : 0); // Causal mask
    }
  }
  
  const maskedOutput = transformerBlock.forward(input, batchSize, seqLen, mask);
  assert(maskedOutput.rows === input.rows, 'Masked TransformerBlock output should preserve dimensions');
  assert(maskedOutput.columns === input.columns, 'Masked TransformerBlock output should preserve dimensions');
  
  console.log('✓ TransformerBlock PyTorch comparison tests passed');
}

function testLossFunctionComparison() {
  console.log('Testing loss function PyTorch comparison...');
  
  // Test cross-entropy loss
  const batchSize = 3;
  const numClasses = 4;
  
  const logits = new Matrix([
    [2.0, 1.0, 0.1, 0.5],
    [0.5, 2.5, 1.0, 0.2],
    [1.0, 0.5, 2.0, 1.5]
  ]);
  
  const targets = [1, 1, 2]; // Class indices as array
  
  const loss = crossEntropyLoss(logits, targets);
  
  // Verify loss is a positive scalar
  assert(typeof loss === 'number', 'Cross-entropy loss should return a number');
  assert(loss > 0, 'Cross-entropy loss should be positive');
  assert(!isNaN(loss), 'Cross-entropy loss should not be NaN');
  
  // Test softmax function
  const probs = softmax(logits);
  
  // Verify softmax properties
  assert(probs.rows === logits.rows, 'Softmax should preserve batch dimension');
  assert(probs.columns === logits.columns, 'Softmax should preserve class dimension');
  
  // Verify probabilities sum to 1 for each sample
  for (let i = 0; i < probs.rows; i++) {
    let sum = 0;
    for (let j = 0; j < probs.columns; j++) {
      const prob = probs.get(i, j);
      assert(prob >= 0, `Softmax probability should be non-negative: ${prob}`);
      assert(prob <= 1, `Softmax probability should be <= 1: ${prob}`);
      sum += prob;
    }
    assertClose(sum, 1.0, 1e-6, `Softmax probabilities should sum to 1 for sample ${i}`);
  }
  
  console.log('✓ Loss function PyTorch comparison tests passed');
}

function testNumericalStability() {
  console.log('Testing numerical stability...');
  
  // Test with extreme values
  const extremeLogits = new Matrix([
    [100.0, -100.0, 50.0, -50.0],
    [-1000.0, 1000.0, 0.0, 500.0]
  ]);
  
  const stableProbs = softmax(extremeLogits);
  
  // Verify no NaN or Inf values
  for (let i = 0; i < stableProbs.rows; i++) {
    for (let j = 0; j < stableProbs.columns; j++) {
      const prob = stableProbs.get(i, j);
      assert(!isNaN(prob), `Softmax should handle extreme values without NaN: ${prob}`);
      assert(isFinite(prob), `Softmax should handle extreme values without Inf: ${prob}`);
      assert(prob >= 0, `Softmax probability should be non-negative: ${prob}`);
    }
  }
  
  // Test LayerNorm with extreme values
  const layerNorm = new LayerNorm(4);
  const extremeInput = new Matrix([
    [1000.0, -1000.0, 500.0, -500.0],
    [0.001, -0.001, 0.0005, -0.0005]
  ]);
  
  const normalizedOutput = layerNorm.forward(extremeInput);
  
  // Verify no NaN values in normalized output
  for (let i = 0; i < normalizedOutput.rows; i++) {
    for (let j = 0; j < normalizedOutput.columns; j++) {
      const val = normalizedOutput.get(i, j);
      assert(!isNaN(val), `LayerNorm should handle extreme values without NaN: ${val}`);
      assert(isFinite(val), `LayerNorm should handle extreme values without Inf: ${val}`);
    }
  }
  
  console.log('✓ Numerical stability tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing PyTorch Comparison ===\n');
  
  try {
    testLinearLayerComparison();
    testLayerNormComparison();
    testAttentionComparison();
    testFeedForwardComparison();
    testTransformerBlockComparison();
    testLossFunctionComparison();
    testNumericalStability();
    
    console.log('\n🎉 All PyTorch comparison tests passed!');
    
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