/**
 * Comprehensive test suite for neural network modules
 * Tests all individual modules with extensive coverage
 */

import { Matrix } from 'ml-matrix';
import { Module } from '../../src/nn/modules/module.js';
import { Linear } from '../../src/nn/modules/linear.js';
import { Embedding } from '../../src/nn/modules/embedding.js';
import { LayerNorm } from '../../src/nn/modules/layerNorm.js';
import { ReLU, Softmax, LeakyReLU } from '../../src/nn/modules/activation.js';
import { PositionalEncoding } from '../../src/nn/modules/positional.js';
import { MultiHeadAttention } from '../../src/nn/modules/attention.js';
import { FeedForward } from '../../src/nn/modules/feedforward.js';
import { TransformerBlock } from '../../src/nn/modules/transformer.js';

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

function testModuleBase() {
  console.log('Testing Module base class...');
  
  // Test that Module is abstract
  try {
    new Module();
    assert(false, 'Module should be abstract and not instantiable');
  } catch (error) {
    assert(error.message.includes('abstract'), 'Should throw abstract class error');
  }
  
  // Test with concrete implementation
  class TestModule extends Module {
    forward(x) {
      return x;
    }
    
    backward(dZ) {
      return dZ;
    }
  }
  
  const testModule = new TestModule();
  assert(testModule instanceof Module, 'Concrete module should be instance of Module');
  
  console.log('✓ Module base class tests passed');
}

function testLinearLayer() {
  console.log('Testing Linear layer...');
  
  const inFeatures = 10;
  const outFeatures = 5;
  const linear = new Linear(inFeatures, outFeatures);
  
  // Test initialization
  assert(linear.inFeatures === inFeatures, 'Input features should be set');
  assert(linear.outFeatures === outFeatures, 'Output features should be set');
  assert(linear.W.rows === inFeatures, 'Weight matrix should have correct rows');
  assert(linear.W.columns === outFeatures, 'Weight matrix should have correct columns');
  assert(linear.b.rows === 1, 'Bias should have correct rows');
  assert(linear.b.columns === outFeatures, 'Bias should have correct columns');
  
  // Test forward pass
  const batchSize = 3;
  const input = Matrix.random(batchSize, inFeatures);
  const output = linear.forward(input);
  
  assert(output.rows === batchSize, 'Output should have correct batch size');
  assert(output.columns === outFeatures, 'Output should have correct feature size');
  
  // Test that output is finite
  for (let i = 0; i < output.rows; i++) {
    for (let j = 0; j < output.columns; j++) {
      assert(isFinite(output.get(i, j)), 'Output values should be finite');
    }
  }
  
  // Test backward pass
  const dZ = Matrix.random(batchSize, outFeatures);
  const dX = linear.backward(dZ);
  
  assert(dX.rows === batchSize, 'Input gradient should have correct batch size');
  assert(dX.columns === inFeatures, 'Input gradient should have correct feature size');
  
  // Test parameter access
  const params = linear.params();
  assert(params.W instanceof Matrix, 'Should have weight parameters');
  assert(params.b instanceof Matrix, 'Should have bias parameters');
  
  const grads = linear.grads();
  assert(grads.W instanceof Matrix, 'Should have weight gradients');
  assert(grads.b instanceof Matrix, 'Should have bias gradients');
  
  console.log('✓ Linear layer tests passed');
}

function testEmbeddingLayer() {
  console.log('Testing Embedding layer...');
  
  const vocabSize = 100;
  const embeddingDim = 64;
  const embedding = new Embedding(vocabSize, embeddingDim);
  
  // Test initialization
  assert(embedding.vocabSize === vocabSize, 'Vocab size should be set');
  assert(embedding.embedDim === embeddingDim, 'Embedding dim should be set');
  assert(embedding.W.rows === vocabSize, 'Weight matrix should have correct rows');
  assert(embedding.W.columns === embeddingDim, 'Weight matrix should have correct columns');
  
  // Test forward pass
  const batchSize = 2;
  const seqLen = 5;
  const input = Matrix.zeros(batchSize, seqLen);
  
  // Fill with valid token indices
  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < seqLen; j++) {
      input.set(i, j, Math.floor(Math.random() * vocabSize));
    }
  }
  
  const output = embedding.forward(input);
  
  assert(output.rows === batchSize * seqLen, 'Output should have correct batch*seq dimension');
  assert(output.columns === embeddingDim, 'Output should have correct embedding dimension');
  
  // Test that embeddings are finite
  for (let i = 0; i < output.rows; i++) {
    for (let j = 0; j < output.columns; j++) {
      assert(isFinite(output.get(i, j)), 'Embedding values should be finite');
    }
  }
  
  console.log('✓ Embedding layer tests passed');
}

function testLayerNorm() {
  console.log('Testing LayerNorm...');
  
  const normalizedShape = 64;
  const layerNorm = new LayerNorm(normalizedShape);
  
  // Test initialization
  assert(layerNorm.dModel === normalizedShape, 'Normalized shape should be set');
  assert(layerNorm.gamma.columns === normalizedShape, 'Gamma should have correct size');
  assert(layerNorm.beta.columns === normalizedShape, 'Beta should have correct size');
  
  // Test forward pass
  const batchSize = 3;
  const input = Matrix.random(batchSize, normalizedShape);
  const output = layerNorm.forward(input);
  
  assert(output.rows === batchSize, 'Output should have correct batch size');
  assert(output.columns === normalizedShape, 'Output should have correct feature size');
  
  // Test normalization properties (approximately zero mean, unit variance)
  for (let i = 0; i < batchSize; i++) {
    let sum = 0;
    let sumSq = 0;
    
    for (let j = 0; j < normalizedShape; j++) {
      const val = output.get(i, j);
      sum += val;
      sumSq += val * val;
    }
    
    const mean = sum / normalizedShape;
    const variance = (sumSq / normalizedShape) - (mean * mean);
    
    assertClose(mean, 0, 1e-4, `Row ${i} should have approximately zero mean`);
    assertClose(variance, 1, 1e-3, `Row ${i} should have approximately unit variance`);
  }
  
  console.log('✓ LayerNorm tests passed');
}

function testActivationFunctions() {
  console.log('Testing activation functions...');
  
  // Test ReLU
  const relu = new ReLU();
  const reluInput = new Matrix([[-1, 0, 1, 2]]);
  const reluOutput = relu.forward(reluInput);
  
  assert(reluOutput.get(0, 0) === 0, 'ReLU should zero negative values');
  assert(reluOutput.get(0, 1) === 0, 'ReLU should zero zero values');
  assert(reluOutput.get(0, 2) === 1, 'ReLU should preserve positive values');
  assert(reluOutput.get(0, 3) === 2, 'ReLU should preserve positive values');
  
  // Test LeakyReLU
  const leakyRelu = new LeakyReLU(0.1);
  const leakyOutput = leakyRelu.forward(reluInput);
  
  assert(leakyOutput.get(0, 0) === -0.1, 'LeakyReLU should scale negative values');
  assert(leakyOutput.get(0, 1) === 0, 'LeakyReLU should preserve zero');
  assert(leakyOutput.get(0, 2) === 1, 'LeakyReLU should preserve positive values');
  
  // Test Softmax
  const softmax = new Softmax();
  const softmaxInput = new Matrix([[1, 2, 3]]);
  const softmaxOutput = softmax.forward(softmaxInput);
  
  // Check that output sums to 1
  let sum = 0;
  for (let j = 0; j < softmaxOutput.columns; j++) {
    const val = softmaxOutput.get(0, j);
    assert(val >= 0 && val <= 1, 'Softmax values should be in [0, 1]');
    sum += val;
  }
  assertClose(sum, 1.0, 1e-6, 'Softmax should sum to 1');
  
  console.log('✓ Activation function tests passed');
}

function testPositionalEncoding() {
  console.log('Testing PositionalEncoding...');
  
  const maxLen = 100;
  const dModel = 64;
  const posEnc = new PositionalEncoding(maxLen, dModel);
  
  // Test initialization
  assert(posEnc.maxLen === maxLen, 'Max length should be set');
  assert(posEnc.dModel === dModel, 'Model dimension should be set');
  assert(posEnc.W.rows === maxLen, 'PE matrix should have correct rows');
  assert(posEnc.W.columns === dModel, 'PE matrix should have correct columns');
  
  // Test forward pass
  const seqLen = 10;
  const input = Matrix.random(seqLen, dModel);
  const output = posEnc.forward(input);
  
  assert(output.rows === seqLen, 'Output should have correct sequence length');
  assert(output.columns === dModel, 'Output should have correct model dimension');
  
  // Test that positional encoding is deterministic
  const output2 = posEnc.forward(input);
  for (let i = 0; i < output.rows; i++) {
    for (let j = 0; j < output.columns; j++) {
      assertClose(output.get(i, j), output2.get(i, j), 1e-10, 
                 'Positional encoding should be deterministic');
    }
  }
  
  console.log('✓ PositionalEncoding tests passed');
}

function testMultiHeadAttention() {
  console.log('Testing MultiHeadAttention...');
  
  const dModel = 64;
  const nHeads = 4;
  const attention = new MultiHeadAttention(dModel, nHeads);
  
  // Test initialization
  assert(attention.dModel === dModel, 'Model dimension should be set');
  assert(attention.nHeads === nHeads, 'Number of heads should be set');
  assert(attention.dK === dModel / nHeads, 'Head dimension should be correct');
  
  // Test forward pass
  const B = 2;
  const T = 8;
  const input = Matrix.random(B * T, dModel);
  const output = attention.forward(input, B, T);
  
  assert(output.rows === B * T, 'Output should have correct batch*seq dimension');
  assert(output.columns === dModel, 'Output should have correct model dimension');
  
  // Test with causal mask
  const mask = Matrix.zeros(T, T);
  for (let i = 0; i < T; i++) {
    for (let j = i + 1; j < T; j++) {
      mask.set(i, j, -1e9);
    }
  }
  
  const maskedOutput = attention.forward(input, B, T, mask);
  assert(maskedOutput.rows === B * T, 'Masked output should have correct dimensions');
  assert(maskedOutput.columns === dModel, 'Masked output should have correct model dimension');
  
  console.log('✓ MultiHeadAttention tests passed');
}

function testFeedForward() {
  console.log('Testing FeedForward...');
  
  const dModel = 64;
  const dFF = 256;
  const ff = new FeedForward(dModel, dFF);
  
  // Test initialization
  assert(ff.dModel === dModel, 'Model dimension should be set');
  assert(ff.dFF === dFF, 'Feed-forward dimension should be set');
  
  // Test forward pass
  const batchSize = 3;
  const seqLen = 5;
  const input = Matrix.random(batchSize * seqLen, dModel);
  const output = ff.forward(input);
  
  assert(output.rows === batchSize * seqLen, 'Output should have correct batch*seq dimension');
  assert(output.columns === dModel, 'Output should have correct model dimension');
  
  // Test that output is finite
  for (let i = 0; i < output.rows; i++) {
    for (let j = 0; j < output.columns; j++) {
      assert(isFinite(output.get(i, j)), 'FeedForward output should be finite');
    }
  }
  
  console.log('✓ FeedForward tests passed');
}

function testTransformerBlock() {
  console.log('Testing TransformerBlock...');
  
  const dModel = 64;
  const nHeads = 4;
  const dFF = 256;
  const transformer = new TransformerBlock(dModel, nHeads, dFF);
  
  // Test initialization
  assert(transformer.dModel === dModel, 'Model dimension should be set');
  assert(transformer.nHeads === nHeads, 'Number of heads should be set');
  assert(transformer.dFF === dFF, 'Feed-forward dimension should be set');
  
  // Test forward pass
  const B = 2;
  const T = 6;
  const input = Matrix.random(B * T, dModel);
  const output = transformer.forward(input, B, T);
  
  assert(output.rows === B * T, 'Output should have correct batch*seq dimension');
  assert(output.columns === dModel, 'Output should have correct model dimension');
  
  // Test residual connection property (output should be different from input)
  let isDifferent = false;
  for (let i = 0; i < Math.min(input.rows, output.rows) && !isDifferent; i++) {
    for (let j = 0; j < Math.min(input.columns, output.columns) && !isDifferent; j++) {
      if (Math.abs(input.get(i, j) - output.get(i, j)) > 1e-6) {
        isDifferent = true;
      }
    }
  }
  assert(isDifferent, 'Transformer should modify the input');
  
  console.log('✓ TransformerBlock tests passed');
}

function testParameterSharing() {
  console.log('Testing parameter sharing and isolation...');
  
  // Test that different instances have different parameters
  const linear1 = new Linear(10, 5);
  const linear2 = new Linear(10, 5);
  
  // Modify one instance
  linear1.W.set(0, 0, 999);
  
  // Check that the other instance is unaffected
  assert(linear2.W.get(0, 0) !== 999, 'Different instances should have isolated parameters');
  
  // Test parameter access consistency
  const params1 = linear1.params();
  const params2 = linear1.params();
  
  // Should reference the same underlying matrices
  params1.W.set(1, 1, 888);
  assert(params2.W.get(1, 1) === 888, 'Parameter access should be consistent');
  
  console.log('✓ Parameter sharing tests passed');
}

function testGradientFlow() {
  console.log('Testing gradient flow...');
  
  // Test that gradients flow through a simple network
  const linear1 = new Linear(5, 3);
  const relu = new ReLU();
  const linear2 = new Linear(3, 2);
  
  const input = Matrix.random(2, 5);
  
  // Forward pass
  const h1 = linear1.forward(input);
  const h2 = relu.forward(h1);
  const output = linear2.forward(h2);
  
  // Backward pass
  const dOutput = Matrix.ones(2, 2);
  const dH2 = linear2.backward(dOutput);
  const dH1 = relu.backward(dH2);
  const dInput = linear1.backward(dH1);
  
  // Check that gradients exist and are finite
  const grads1 = linear1.grads();
  const grads2 = linear2.grads();
  
  assert(grads1.W instanceof Matrix, 'Linear1 should have weight gradients');
  assert(grads2.W instanceof Matrix, 'Linear2 should have weight gradients');
  
  // Check that gradients are non-zero (assuming non-pathological case)
  let hasNonZeroGrad = false;
  for (let i = 0; i < grads1.W.rows && !hasNonZeroGrad; i++) {
    for (let j = 0; j < grads1.W.columns && !hasNonZeroGrad; j++) {
      if (Math.abs(grads1.W.get(i, j)) > 1e-10) {
        hasNonZeroGrad = true;
      }
    }
  }
  assert(hasNonZeroGrad, 'Should have non-zero gradients');
  
  console.log('✓ Gradient flow tests passed');
}

function testModuleComposition() {
  console.log('Testing module composition...');
  
  // Test that modules can be composed correctly
  const embedding = new Embedding(100, 32);
  const layerNorm = new LayerNorm(32);
  const linear = new Linear(32, 10);
  
  // Create input
  const batchSize = 2;
  const seqLen = 5;
  const input = Matrix.zeros(batchSize, seqLen);
  
  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < seqLen; j++) {
      input.set(i, j, Math.floor(Math.random() * 100));
    }
  }
  
  // Forward pass through composition
  const embedded = embedding.forward(input);
  const normalized = layerNorm.forward(embedded);
  const output = linear.forward(normalized);
  
  assert(output.rows === batchSize * seqLen, 'Composed output should have correct batch*seq dimension');
  assert(output.columns === 10, 'Composed output should have correct feature dimension');
  
  console.log('✓ Module composition tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Neural Network Modules ===\n');
  
  try {
    testModuleBase();
    testLinearLayer();
    testEmbeddingLayer();
    testLayerNorm();
    testActivationFunctions();
    testPositionalEncoding();
    testMultiHeadAttention();
    testFeedForward();
    testTransformerBlock();
    testParameterSharing();
    testGradientFlow();
    testModuleComposition();
    
    console.log('\n🎉 All neural network module tests passed!');
    
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