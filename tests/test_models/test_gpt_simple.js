import { Matrix } from 'ml-matrix';
import { GPT } from '../../src/models/GPT.js';

/**
 * Test suite for GPT Model
 */

// Test utilities
function assertClose(actual, expected, tolerance = 1e-6, message = '') {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(`${message} Expected ${expected}, got ${actual}, diff: ${diff}`);
  }
}

function assertMatrixShape(matrix, expectedRows, expectedCols, message = '') {
  if (matrix.rows !== expectedRows || matrix.columns !== expectedCols) {
    throw new Error(`${message} Expected shape [${expectedRows}, ${expectedCols}], got [${matrix.rows}, ${matrix.columns}]`);
  }
}

function assertFinite(value, message = '') {
  if (!isFinite(value) || isNaN(value)) {
    throw new Error(`${message} Value should be finite, got ${value}`);
  }
}

// Test GPT Initialization
function testGPTInitialization() {
  console.log('Testing GPT initialization...');
  
  const vocabSize = 100;
  const maxLen = 64;
  const dModel = 128;
  const nHeads = 4;
  const nLayers = 2;
  const dFF = 256;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Check basic properties
  if (gpt.vocabSize !== vocabSize) {
    throw new Error('Vocab size not set correctly');
  }
  if (gpt.maxLen !== maxLen) {
    throw new Error('Max length not set correctly');
  }
  if (gpt.dModel !== dModel) {
    throw new Error('Model dimension not set correctly');
  }
  
  // Check components exist
  if (!gpt.tokEmb || !gpt.posEmb || !gpt.lnF || !gpt.lmHead) {
    throw new Error('Required components not initialized');
  }
  
  // Check number of transformer blocks
  if (gpt.blocks.length !== nLayers) {
    throw new Error(`Expected ${nLayers} transformer blocks, got ${gpt.blocks.length}`);
  }
  
  console.log('✓ GPT initialization test passed');
}

// Test Forward Pass (Inference)
function testGPTForwardInference() {
  console.log('Testing GPT forward pass (inference)...');
  
  const vocabSize = 50;
  const maxLen = 32;
  const dModel = 64;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 128;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Create input tokens (batch_size=2, seq_len=8)
  const X = new Matrix([
    [1, 5, 10, 15, 20, 25, 30, 35],
    [2, 6, 11, 16, 21, 26, 31, 36]
  ]);
  
  const logits = gpt.forward(X);
  
  // Check output shape: (batch_size, seq_len, vocab_size)
  assertMatrixShape(logits, 2, 8, 'Logits shape for inference');
  
  // Check that logits contain finite values
  for (let i = 0; i < logits.rows; i++) {
    for (let j = 0; j < logits.columns; j++) {
      const tokenLogits = logits.data[i][j];
      if (!Array.isArray(tokenLogits) || tokenLogits.length !== vocabSize) {
        throw new Error(`Expected logits to be array of length ${vocabSize}`);
      }
      for (const logit of tokenLogits) {
        assertFinite(logit, `Logit at position [${i}, ${j}]`);
      }
    }
  }
  
  console.log('✓ GPT forward inference test passed');
}

// Test Forward Pass (Training)
function testGPTForwardTraining() {
  console.log('Testing GPT forward pass (training)...');
  
  const vocabSize = 50;
  const maxLen = 32;
  const dModel = 64;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 128;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Create input tokens and targets
  const X = new Matrix([
    [1, 5, 10, 15],
    [2, 6, 11, 16]
  ]);
  const targets = new Matrix([
    [5, 10, 15, 20],
    [6, 11, 16, 21]
  ]);
  
  const result = gpt.forward(X, targets);
  
  // Should return [logits, loss]
  if (!Array.isArray(result) || result.length !== 2) {
    throw new Error('Training forward should return [logits, loss]');
  }
  
  const [logits, loss] = result;
  
  // Check logits shape
  assertMatrixShape(logits, 2, 4, 'Logits shape for training');
  
  // Check loss is finite and positive
  assertFinite(loss, 'Training loss');
  if (loss <= 0) {
    throw new Error('Loss should be positive');
  }
  
  console.log('✓ GPT forward training test passed');
}

// Test Backward Pass
function testGPTBackward() {
  console.log('Testing GPT backward pass...');
  
  const vocabSize = 30;
  const maxLen = 16;
  const dModel = 32;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 64;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Forward pass with targets
  const X = new Matrix([[1, 5, 10]]);
  const targets = new Matrix([[5, 10, 15]]);
  
  const [logits, loss] = gpt.forward(X, targets);
  
  // Backward pass
  gpt.backward();
  
  // Check that gradients exist and are finite
  const grads = gpt.grads();
  
  let gradCount = 0;
  for (const [name, grad] of Object.entries(grads)) {
    if (grad instanceof Matrix) {
      gradCount++;
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          const val = grad.get(i, j);
          assertFinite(val, `Gradient ${name} at [${i}, ${j}]`);
        }
      }
    }
  }
  
  if (gradCount === 0) {
    throw new Error('No gradients found after backward pass');
  }
  
  console.log('✓ GPT backward test passed');
}

// Test Causal Mask
function testCausalMask() {
  console.log('Testing causal mask creation...');
  
  const gpt = new GPT(10, 8, 16, 2, 1, 32);
  const mask = gpt._createCausalMask(4);
  
  // Check mask shape
  assertMatrixShape(mask, 4, 4, 'Causal mask shape');
  
  // Check mask values
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const val = mask.get(i, j);
      if (i < j) {
        // Future positions should be masked (large negative)
        if (val > -1e8) {
          throw new Error(`Future position [${i}, ${j}] should be masked`);
        }
      } else {
        // Current and past positions should be 0
        if (val !== 0) {
          throw new Error(`Past/current position [${i}, ${j}] should be 0`);
        }
      }
    }
  }
  
  console.log('✓ Causal mask test passed');
}

// Test Text Generation
function testTextGeneration() {
  console.log('Testing text generation...');
  
  const vocabSize = 20;
  const maxLen = 16;
  const dModel = 32;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 64;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Start with a single token
  const startTokens = new Matrix([[1]]);
  const maxNewTokens = 5;
  
  const generated = gpt.generate(startTokens, maxNewTokens, 1.0);
  
  // Check output shape
  const expectedLength = 1 + maxNewTokens;
  assertMatrixShape(generated, 1, expectedLength, 'Generated sequence shape');
  
  // Check that all tokens are valid (within vocab range)
  for (let j = 0; j < generated.columns; j++) {
    const token = generated.get(0, j);
    if (token < 0 || token >= vocabSize || !Number.isInteger(token)) {
      throw new Error(`Invalid token ${token} at position ${j}`);
    }
  }
  
  // Check that first token is unchanged
  if (generated.get(0, 0) !== 1) {
    throw new Error('First token should remain unchanged');
  }
  
  console.log('✓ Text generation test passed');
}

// Test Generation with EOS Token
function testGenerationWithEOS() {
  console.log('Testing text generation with EOS token...');
  
  const vocabSize = 10;
  const maxLen = 16;
  const dModel = 32;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 64;
  
  const gpt = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const startTokens = new Matrix([[1]]);
  const maxNewTokens = 10;
  const eosTokenId = 0; // Use token 0 as EOS
  
  const generated = gpt.generate(startTokens, maxNewTokens, 1.0, eosTokenId);
  
  // Should stop early if EOS is generated, or continue to max length
  if (generated.columns < 1 || generated.columns > 1 + maxNewTokens) {
    throw new Error('Generated sequence length is invalid');
  }
  
  console.log('✓ Generation with EOS test passed');
}

// Test Parameter Collection
function testParameterCollection() {
  console.log('Testing parameter collection...');
  
  const gpt = new GPT(20, 16, 32, 2, 1, 64);
  const params = gpt.params();
  
  // Check that we have parameters from all components
  const expectedPrefixes = ['tok_emb', 'pos_emb', 'blocks.0', 'ln_f', 'lm_head'];
  
  for (const prefix of expectedPrefixes) {
    const hasPrefix = Object.keys(params).some(key => key.startsWith(prefix));
    if (!hasPrefix) {
      throw new Error(`Missing parameters with prefix: ${prefix}`);
    }
  }
  
  // Check that all parameters are matrices
  for (const [name, param] of Object.entries(params)) {
    if (!(param instanceof Matrix)) {
      throw new Error(`Parameter ${name} is not a Matrix`);
    }
  }
  
  console.log('✓ Parameter collection test passed');
}

// Test Gradient Collection
function testGradientCollection() {
  console.log('Testing gradient collection...');
  
  const gpt = new GPT(20, 16, 32, 2, 1, 64);
  
  // Forward and backward pass to generate gradients
  const X = new Matrix([[1, 5]]);
  const targets = new Matrix([[5, 10]]);
  
  gpt.forward(X, targets);
  gpt.backward();
  
  const grads = gpt.grads();
  
  // Check that we have gradients from core components (at least some gradients should exist)
  const coreComponents = ['blocks.0', 'ln_f', 'lm_head'];
  let foundGradients = 0;
  
  for (const prefix of coreComponents) {
    const hasPrefix = Object.keys(grads).some(key => key.startsWith(prefix));
    if (hasPrefix) {
      foundGradients++;
    }
  }
  
  if (foundGradients === 0) {
    throw new Error('No gradients found from core components');
  }
  
  // Check that all gradients are matrices
  for (const [name, grad] of Object.entries(grads)) {
    if (!(grad instanceof Matrix)) {
      throw new Error(`Gradient ${name} is not a Matrix`);
    }
  }
  
  console.log('✓ Gradient collection test passed');
}

// Run all tests
function runAllTests() {
  console.log('=== Testing GPT Model ===\n');
  
  try {
    testGPTInitialization();
    testGPTForwardInference();
    testGPTForwardTraining();
    testGPTBackward();
    testCausalMask();
    testTextGeneration();
    testGenerationWithEOS();
    testParameterCollection();
    testGradientCollection();
    
    console.log('\n🎉 All GPT tests passed!');
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