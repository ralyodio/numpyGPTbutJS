/**
 * Test suite for component comparison tests
 * Tests individual components and their interactions
 */

import { Matrix } from 'ml-matrix';
import { Linear } from '../src/nn/modules/linear.js';
import { LayerNorm } from '../src/nn/modules/layerNorm.js';
import { MultiHeadAttention } from '../src/nn/modules/attention.js';
import { FeedForward } from '../src/nn/modules/feedforward.js';
import { TransformerBlock } from '../src/nn/modules/transformer.js';
import { Embedding } from '../src/nn/modules/embedding.js';
import { PositionalEncoding } from '../src/nn/modules/positional.js';
import { ReLU, Softmax } from '../src/nn/modules/activation.js';
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

function testEmbeddingComponent() {
  console.log('Testing Embedding component...');
  
  const vocabSize = 10;
  const embeddingDim = 6;
  const embedding = new Embedding(vocabSize, embeddingDim);
  
  // Test single token embedding
  const tokenIds = new Matrix([[2], [5], [8]]);
  const embeddings = embedding.forward(tokenIds);
  
  // Verify output dimensions
  assert(embeddings.rows === 3, 'Embedding should output correct number of tokens');
  assert(embeddings.columns === embeddingDim, 'Embedding should have correct embedding dimension');
  
  // Test that different tokens have different embeddings
  const emb1 = embeddings.getRow(0);
  const emb2 = embeddings.getRow(1);
  let different = false;
  for (let i = 0; i < embeddingDim; i++) {
    if (Math.abs(emb1[i] - emb2[i]) > 1e-6) {
      different = true;
      break;
    }
  }
  assert(different, 'Different tokens should have different embeddings');
  
  // Test batch processing
  const batchTokens = new Matrix([[1, 3, 7], [2, 4, 9]]);
  const batchEmbeddings = embedding.forward(batchTokens);
  assert(batchEmbeddings.rows === 6, 'Batch embedding should flatten correctly');
  assert(batchEmbeddings.columns === embeddingDim, 'Batch embedding should preserve dimension');
  
  console.log('✓ Embedding component tests passed');
}

function testPositionalEncodingComponent() {
  console.log('Testing PositionalEncoding component...');
  
  const dModel = 8;
  const maxLen = 16;
  const posEncoding = new PositionalEncoding(dModel, maxLen);
  
  // Test positional encoding generation
  const seqLen = 4;
  const batchSize = 2;
  const input = new Matrix(batchSize * seqLen, dModel);
  
  // Fill with ones to test additive property
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      input.set(i, j, 1.0);
    }
  }
  
  const output = posEncoding.forward(input, batchSize, seqLen);
  
  // Verify output dimensions
  assert(output.rows === input.rows, 'PositionalEncoding should preserve sequence dimension');
  assert(output.columns === input.columns, 'PositionalEncoding should preserve model dimension');
  
  // Test that positional encoding was added (output should be different from input)
  let hasPositionalInfo = false;
  for (let i = 0; i < Math.min(4, output.rows); i++) {
    for (let j = 0; j < output.columns; j++) {
      if (Math.abs(output.get(i, j) - 1.0) > 1e-6) {
        hasPositionalInfo = true;
        break;
      }
    }
    if (hasPositionalInfo) break;
  }
  assert(hasPositionalInfo, 'PositionalEncoding should modify input with positional information');
  
  // Test that different positions have different encodings
  if (output.rows >= 2) {
    const pos1 = output.getRow(0);
    const pos2 = output.getRow(1);
    let positionsDiffer = false;
    for (let j = 0; j < dModel; j++) {
      if (Math.abs(pos1[j] - pos2[j]) > 1e-6) {
        positionsDiffer = true;
        break;
      }
    }
    assert(positionsDiffer, 'Different positions should have different encodings');
  }
  
  console.log('✓ PositionalEncoding component tests passed');
}

function testActivationComponents() {
  console.log('Testing Activation components...');
  
  // Test ReLU
  const relu = new ReLU();
  const reluInput = new Matrix([
    [1.0, -2.0, 3.0, -4.0],
    [0.5, -0.5, 0.0, 2.0]
  ]);
  
  const reluOutput = relu.forward(reluInput);
  
  // Verify ReLU properties
  for (let i = 0; i < reluOutput.rows; i++) {
    for (let j = 0; j < reluOutput.columns; j++) {
      const input = reluInput.get(i, j);
      const output = reluOutput.get(i, j);
      const expected = Math.max(0, input);
      assertClose(output, expected, 1e-6, `ReLU output at [${i},${j}]`);
    }
  }
  
  // Test Softmax
  const softmaxFunc = new Softmax();
  const softmaxInput = new Matrix([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 1.0]
  ]);
  
  const softmaxOutput = softmaxFunc.forward(softmaxInput);
  
  // Verify softmax properties
  for (let i = 0; i < softmaxOutput.rows; i++) {
    let sum = 0;
    for (let j = 0; j < softmaxOutput.columns; j++) {
      const prob = softmaxOutput.get(i, j);
      assert(prob >= 0, `Softmax probability should be non-negative: ${prob}`);
      assert(prob <= 1, `Softmax probability should be <= 1: ${prob}`);
      sum += prob;
    }
    assertClose(sum, 1.0, 1e-6, `Softmax probabilities should sum to 1 for row ${i}`);
  }
  
  console.log('✓ Activation component tests passed');
}

function testComponentComposition() {
  console.log('Testing component composition...');
  
  // Test embedding + positional encoding pipeline
  const vocabSize = 20;
  const dModel = 12;
  const seqLen = 6;
  const batchSize = 2;
  
  const embedding = new Embedding(vocabSize, dModel);
  const posEncoding = new PositionalEncoding(dModel, 32);
  
  // Create token sequence
  const tokens = new Matrix(batchSize, seqLen);
  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < seqLen; j++) {
      tokens.set(i, j, (i * seqLen + j) % vocabSize);
    }
  }
  
  // Forward through embedding
  const embeddings = embedding.forward(tokens);
  assert(embeddings.rows === batchSize * seqLen, 'Embedding should flatten batch and sequence');
  assert(embeddings.columns === dModel, 'Embedding should have correct model dimension');
  
  // Forward through positional encoding
  const withPositions = posEncoding.forward(embeddings, batchSize, seqLen);
  assert(withPositions.rows === embeddings.rows, 'PositionalEncoding should preserve dimensions');
  assert(withPositions.columns === embeddings.columns, 'PositionalEncoding should preserve dimensions');
  
  // Test transformer block on the result
  const transformerBlock = new TransformerBlock(dModel, 3, dModel * 2);
  const transformed = transformerBlock.forward(withPositions, batchSize, seqLen);
  
  assert(transformed.rows === withPositions.rows, 'TransformerBlock should preserve dimensions');
  assert(transformed.columns === withPositions.columns, 'TransformerBlock should preserve dimensions');
  
  console.log('✓ Component composition tests passed');
}

function testGradientFlow() {
  console.log('Testing gradient flow through components...');
  
  const dModel = 8;
  const seqLen = 4;
  const batchSize = 2;
  
  // Create a simple pipeline: Linear -> LayerNorm -> ReLU -> Linear
  const linear1 = new Linear(dModel, dModel);
  const layerNorm = new LayerNorm(dModel);
  const relu = new ReLU();
  const linear2 = new Linear(dModel, dModel);
  
  // Forward pass
  const input = new Matrix(batchSize * seqLen, dModel);
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      input.set(i, j, (i + j) * 0.1);
    }
  }
  
  const out1 = linear1.forward(input);
  const out2 = layerNorm.forward(out1);
  const out3 = relu.forward(out2);
  const output = linear2.forward(out3);
  
  // Create gradient for backward pass
  const gradOutput = new Matrix(output.rows, output.columns);
  for (let i = 0; i < gradOutput.rows; i++) {
    for (let j = 0; j < gradOutput.columns; j++) {
      gradOutput.set(i, j, 1.0); // Uniform gradient
    }
  }
  
  // Backward pass
  const grad3 = linear2.backward(gradOutput);
  const grad2 = relu.backward(grad3);
  const grad1 = layerNorm.backward(grad2);
  const gradInput = linear1.backward(grad1);
  
  // Verify gradient dimensions
  assert(gradInput.rows === input.rows, 'Gradient should have same dimensions as input');
  assert(gradInput.columns === input.columns, 'Gradient should have same dimensions as input');
  
  // Verify gradients were computed for all parameters
  assert(linear1.dW !== null, 'Linear1 weight gradients should be computed');
  assert(linear1.db !== null, 'Linear1 bias gradients should be computed');
  assert(linear2.dW !== null, 'Linear2 weight gradients should be computed');
  assert(linear2.db !== null, 'Linear2 bias gradients should be computed');
  assert(layerNorm.dWeight !== null, 'LayerNorm weight gradients should be computed');
  assert(layerNorm.dBias !== null, 'LayerNorm bias gradients should be computed');
  
  console.log('✓ Gradient flow tests passed');
}

function testParameterSharing() {
  console.log('Testing parameter sharing and collection...');
  
  const dModel = 6;
  const nHeads = 2;
  const dFF = 12;
  
  // Create transformer block
  const transformerBlock = new TransformerBlock(dModel, nHeads, dFF);
  
  // Get all parameters
  const params = transformerBlock.params();
  
  // Verify parameter structure
  assert(typeof params === 'object', 'Parameters should be returned as object');
  
  // Check that attention parameters are included
  const attnParamKeys = Object.keys(params).filter(key => key.startsWith('attn.'));
  assert(attnParamKeys.length > 0, 'Should include attention parameters');
  
  // Check that layer norm parameters are included
  const ln1ParamKeys = Object.keys(params).filter(key => key.startsWith('ln1.'));
  const ln2ParamKeys = Object.keys(params).filter(key => key.startsWith('ln2.'));
  assert(ln1ParamKeys.length > 0, 'Should include first layer norm parameters');
  assert(ln2ParamKeys.length > 0, 'Should include second layer norm parameters');
  
  // Check that feedforward parameters are included
  const ffnParamKeys = Object.keys(params).filter(key => key.startsWith('ffn.'));
  assert(ffnParamKeys.length > 0, 'Should include feedforward parameters');
  
  // Test parameter modification
  const originalParam = params['ln1.gamma'];
  assert(originalParam instanceof Matrix, 'Parameters should be Matrix objects');
  
  // Modify parameter and verify it affects the module
  const newValue = 2.0;
  originalParam.set(0, 0, newValue);
  const retrievedParams = transformerBlock.params();
  assertClose(retrievedParams['ln1.gamma'].get(0, 0), newValue, 1e-6, 'Parameter modification should be reflected');
  
  console.log('✓ Parameter sharing tests passed');
}

function testMemoryEfficiency() {
  console.log('Testing memory efficiency...');
  
  const dModel = 16;
  const seqLen = 32;
  const batchSize = 4;
  
  // Create multiple components
  const components = [
    new Linear(dModel, dModel),
    new LayerNorm(dModel),
    new MultiHeadAttention(dModel, 4),
    new FeedForward(dModel, dModel * 2),
    new TransformerBlock(dModel, 4, dModel * 2)
  ];
  
  // Test multiple forward passes
  const startTime = Date.now();
  const numPasses = 10;
  
  for (let pass = 0; pass < numPasses; pass++) {
    const input = new Matrix(batchSize * seqLen, dModel);
    
    // Fill with random-ish data
    for (let i = 0; i < input.rows; i++) {
      for (let j = 0; j < input.columns; j++) {
        input.set(i, j, Math.sin(i * 0.1 + j * 0.2 + pass * 0.05));
      }
    }
    
    // Forward through components
    let current = input;
    for (const component of components) {
      if (component instanceof MultiHeadAttention || component instanceof TransformerBlock) {
        current = component.forward(current, batchSize, seqLen);
      } else {
        current = component.forward(current);
      }
      
      // Verify output is valid
      assert(current instanceof Matrix, 'Component output should be Matrix');
      assert(!isNaN(current.get(0, 0)), 'Component output should not contain NaN');
    }
  }
  
  const endTime = Date.now();
  const timePerPass = (endTime - startTime) / numPasses;
  
  // Should be reasonably fast (less than 100ms per pass for this size)
  assert(timePerPass < 100, `Forward passes should be efficient: ${timePerPass}ms per pass`);
  
  console.log(`  Completed ${numPasses} forward passes in ${endTime - startTime}ms (${timePerPass.toFixed(2)}ms per pass)`);
  console.log('✓ Memory efficiency tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Component Comparisons ===\n');
  
  try {
    testEmbeddingComponent();
    testPositionalEncodingComponent();
    testActivationComponents();
    testComponentComposition();
    testGradientFlow();
    testParameterSharing();
    testMemoryEfficiency();
    
    console.log('\n🎉 All component comparison tests passed!');
    
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