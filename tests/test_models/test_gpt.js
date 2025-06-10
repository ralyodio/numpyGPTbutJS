/**
 * Comprehensive test suite for GPT model
 * Tests model architecture, forward/backward passes, text generation, and edge cases
 */

import { Matrix } from 'ml-matrix';
import { GPT } from '../../src/models/GPT.js';

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

function testGPTInitialization() {
  console.log('Testing GPT initialization...');
  
  const vocabSize = 100;
  const maxLen = 64;
  const dModel = 128;
  const nHeads = 4;
  const nLayers = 2;
  const dFF = 512;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Test basic properties
  assert(model.vocabSize === vocabSize, 'Vocab size should be set correctly');
  assert(model.maxLen === maxLen, 'Max length should be set correctly');
  assert(model.dModel === dModel, 'Model dimension should be set correctly');
  
  // Test component initialization
  assert(model.tokEmb !== undefined, 'Token embedding should be initialized');
  assert(model.posEmb !== undefined, 'Positional embedding should be initialized');
  assert(model.blocks.length === nLayers, 'Should have correct number of transformer blocks');
  assert(model.lnF !== undefined, 'Final layer norm should be initialized');
  assert(model.lmHead !== undefined, 'Language modeling head should be initialized');
  
  console.log('✓ GPT initialization tests passed');
}

function testForwardPassInference() {
  console.log('Testing GPT forward pass (inference)...');
  
  const vocabSize = 50;
  const maxLen = 32;
  const dModel = 64;
  const nHeads = 2;
  const nLayers = 2;
  const dFF = 128;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Test single sequence
  const B = 1;
  const T = 8;
  const input = Matrix.zeros(B, T);
  
  // Fill with valid token indices
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < T; j++) {
      input.set(i, j, Math.floor(Math.random() * vocabSize));
    }
  }
  
  const logits = model.forward(input);
  
  // Check output dimensions
  assert(logits.rows === B, 'Output should have correct batch size');
  assert(logits.columns === T, 'Output should have correct sequence length');
  assert(logits.data[0][0].length === vocabSize, 'Output should have correct vocab size');
  
  // Check that outputs are finite
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < vocabSize; v++) {
        const val = logits.data[b][t][v];
        assert(isFinite(val), `Logit at (${b}, ${t}, ${v}) should be finite`);
      }
    }
  }
  
  console.log(`  Output shape: (${B}, ${T}, ${vocabSize})`);
  console.log('✓ Forward pass inference tests passed');
}

function testForwardPassTraining() {
  console.log('Testing GPT forward pass (training)...');
  
  const vocabSize = 30;
  const maxLen = 16;
  const dModel = 32;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 64;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const B = 2;
  const T = 4;
  const input = Matrix.zeros(B, T);
  const targets = Matrix.zeros(B, T);
  
  // Fill with valid token indices
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < T; j++) {
      input.set(i, j, Math.floor(Math.random() * vocabSize));
      targets.set(i, j, Math.floor(Math.random() * vocabSize));
    }
  }
  
  const [logits, loss] = model.forward(input, targets);
  
  // Check output dimensions
  assert(logits.rows === B, 'Logits should have correct batch size');
  assert(logits.columns === T, 'Logits should have correct sequence length');
  assert(logits.data[0][0].length === vocabSize, 'Logits should have correct vocab size');
  
  // Check loss properties
  assert(typeof loss === 'number', 'Loss should be a number');
  assert(loss > 0, 'Loss should be positive');
  assert(isFinite(loss), 'Loss should be finite');
  
  console.log(`  Loss: ${loss.toFixed(4)}`);
  console.log('✓ Forward pass training tests passed');
}

function testBackwardPass() {
  console.log('Testing GPT backward pass...');
  
  const vocabSize = 20;
  const maxLen = 8;
  const dModel = 16;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 32;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const B = 1;
  const T = 4;
  const input = Matrix.zeros(B, T);
  const targets = Matrix.zeros(B, T);
  
  // Fill with valid token indices
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < T; j++) {
      input.set(i, j, Math.floor(Math.random() * vocabSize));
      targets.set(i, j, Math.floor(Math.random() * vocabSize));
    }
  }
  
  // Forward pass
  const [logits, loss] = model.forward(input, targets);
  
  // Backward pass
  model.backward();
  
  // Check that gradients exist
  const grads = model.grads();
  assert(Object.keys(grads).length > 0, 'Should have gradients');
  
  // Check that gradients are finite
  let gradCount = 0;
  for (const [name, grad] of Object.entries(grads)) {
    if (grad instanceof Matrix) {
      gradCount++;
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          const val = grad.get(i, j);
          assert(isFinite(val), `Gradient ${name} at (${i}, ${j}) should be finite`);
        }
      }
    }
  }
  
  assert(gradCount > 0, 'Should have matrix gradients');
  
  console.log(`  Computed gradients for ${gradCount} parameter matrices`);
  console.log('✓ Backward pass tests passed');
}

function testTextGeneration() {
  console.log('Testing text generation...');
  
  const vocabSize = 25;
  const maxLen = 16;
  const dModel = 24;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 48;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Test basic generation
  const startTokens = new Matrix([[1, 2, 3]]);
  const maxNewTokens = 5;
  const temperature = 1.0;
  
  const generated = model.generate(startTokens, maxNewTokens, temperature);
  
  // Check output properties
  assert(generated.rows === 1, 'Generated sequence should have 1 row');
  assert(generated.columns === startTokens.columns + maxNewTokens, 
         'Generated sequence should have correct length');
  
  // Check that all tokens are valid
  for (let j = 0; j < generated.columns; j++) {
    const token = generated.get(0, j);
    assert(Number.isInteger(token), 'Generated tokens should be integers');
    assert(token >= 0 && token < vocabSize, 'Generated tokens should be in vocab range');
  }
  
  // Check that original tokens are preserved
  for (let j = 0; j < startTokens.columns; j++) {
    assert(generated.get(0, j) === startTokens.get(0, j), 
           'Original tokens should be preserved');
  }
  
  console.log(`  Generated sequence length: ${generated.columns}`);
  console.log('✓ Text generation tests passed');
}

function testGenerationWithEOS() {
  console.log('Testing text generation with EOS token...');
  
  const vocabSize = 20;
  const maxLen = 12;
  const dModel = 16;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 32;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const startTokens = new Matrix([[1, 2]]);
  const maxNewTokens = 10;
  const temperature = 1.0;
  const eosTokenId = 0; // Use token 0 as EOS
  
  const generated = model.generate(startTokens, maxNewTokens, temperature, eosTokenId);
  
  // Check that generation respects EOS token
  assert(generated.rows === 1, 'Generated sequence should have 1 row');
  assert(generated.columns >= startTokens.columns, 'Should have at least original tokens');
  assert(generated.columns <= startTokens.columns + maxNewTokens, 
         'Should not exceed max length');
  
  console.log(`  Generated with EOS: ${generated.columns} tokens`);
  console.log('✓ Generation with EOS tests passed');
}

function testParameterCollection() {
  console.log('Testing parameter collection...');
  
  const vocabSize = 15;
  const maxLen = 8;
  const dModel = 12;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 24;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const params = model.params();
  
  // Check that we have parameters
  assert(Object.keys(params).length > 0, 'Should have parameters');
  
  // Check for expected parameter groups
  const expectedGroups = ['tok_emb', 'pos_emb', 'blocks', 'ln_f', 'lm_head'];
  for (const group of expectedGroups) {
    const hasGroup = Object.keys(params).some(key => key.startsWith(group));
    assert(hasGroup, `Should have parameters for ${group}`);
  }
  
  // Check that all parameters are matrices
  let paramCount = 0;
  for (const [name, param] of Object.entries(params)) {
    assert(param instanceof Matrix, `Parameter ${name} should be a Matrix`);
    paramCount++;
  }
  
  console.log(`  Found ${paramCount} parameter matrices`);
  console.log('✓ Parameter collection tests passed');
}

function testGradientCollection() {
  console.log('Testing gradient collection...');
  
  const vocabSize = 12;
  const maxLen = 6;
  const dModel = 8;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 16;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const B = 1;
  const T = 3;
  const input = Matrix.zeros(B, T);
  const targets = Matrix.zeros(B, T);
  
  // Fill with valid token indices
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < T; j++) {
      input.set(i, j, Math.floor(Math.random() * vocabSize));
      targets.set(i, j, Math.floor(Math.random() * vocabSize));
    }
  }
  
  // Forward and backward pass
  model.forward(input, targets);
  model.backward();
  
  const grads = model.grads();
  
  // Check that we have gradients
  assert(Object.keys(grads).length > 0, 'Should have gradients');
  
  // Check that gradients correspond to parameters
  const params = model.params();
  let matchingGrads = 0;
  for (const paramName of Object.keys(params)) {
    if (grads[paramName]) {
      matchingGrads++;
      assert(grads[paramName] instanceof Matrix, 
             `Gradient ${paramName} should be a Matrix`);
    }
  }
  
  console.log(`  Found gradients for ${matchingGrads} parameters`);
  console.log('✓ Gradient collection tests passed');
}

function testBatchProcessing() {
  console.log('Testing batch processing...');
  
  const vocabSize = 20;
  const maxLen = 8;
  const dModel = 16;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 32;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Test different batch sizes
  const batchSizes = [1, 2, 4];
  const T = 4;
  
  for (const B of batchSizes) {
    const input = Matrix.zeros(B, T);
    const targets = Matrix.zeros(B, T);
    
    // Fill with valid token indices
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < T; j++) {
        input.set(i, j, Math.floor(Math.random() * vocabSize));
        targets.set(i, j, Math.floor(Math.random() * vocabSize));
      }
    }
    
    const [logits, loss] = model.forward(input, targets);
    
    // Check dimensions
    assert(logits.rows === B, `Batch size ${B}: logits should have correct batch dimension`);
    assert(logits.columns === T, `Batch size ${B}: logits should have correct sequence dimension`);
    assert(isFinite(loss), `Batch size ${B}: loss should be finite`);
  }
  
  console.log(`  Tested batch sizes: ${batchSizes.join(', ')}`);
  console.log('✓ Batch processing tests passed');
}

function testSequenceLengths() {
  console.log('Testing different sequence lengths...');
  
  const vocabSize = 15;
  const maxLen = 16;
  const dModel = 12;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 24;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  // Test different sequence lengths
  const sequenceLengths = [1, 4, 8, 12];
  const B = 1;
  
  for (const T of sequenceLengths) {
    if (T <= maxLen) {
      const input = Matrix.zeros(B, T);
      
      // Fill with valid token indices
      for (let i = 0; i < B; i++) {
        for (let j = 0; j < T; j++) {
          input.set(i, j, Math.floor(Math.random() * vocabSize));
        }
      }
      
      const logits = model.forward(input);
      
      // Check dimensions
      assert(logits.rows === B, `Sequence length ${T}: should have correct batch dimension`);
      assert(logits.columns === T, `Sequence length ${T}: should have correct sequence dimension`);
    }
  }
  
  console.log(`  Tested sequence lengths: ${sequenceLengths.join(', ')}`);
  console.log('✓ Sequence length tests passed');
}

function testModelConsistency() {
  console.log('Testing model consistency...');
  
  const vocabSize = 10;
  const maxLen = 6;
  const dModel = 8;
  const nHeads = 2;
  const nLayers = 1;
  const dFF = 16;
  
  const model = new GPT(vocabSize, maxLen, dModel, nHeads, nLayers, dFF);
  
  const B = 1;
  const T = 3;
  const input = Matrix.zeros(B, T);
  
  // Fill with specific token indices
  input.set(0, 0, 1);
  input.set(0, 1, 2);
  input.set(0, 2, 3);
  
  // Run forward pass twice with same input
  const logits1 = model.forward(input);
  const logits2 = model.forward(input);
  
  // Results should be identical (deterministic)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < vocabSize; v++) {
        const val1 = logits1.data[b][t][v];
        const val2 = logits2.data[b][t][v];
        assertClose(val1, val2, 1e-10, 
                   `Logits should be identical at (${b}, ${t}, ${v})`);
      }
    }
  }
  
  console.log('✓ Model consistency tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing GPT Model ===\n');
  
  try {
    testGPTInitialization();
    testForwardPassInference();
    testForwardPassTraining();
    testBackwardPass();
    testTextGeneration();
    testGenerationWithEOS();
    testParameterCollection();
    testGradientCollection();
    testBatchProcessing();
    testSequenceLengths();
    testModelConsistency();
    
    console.log('\n🎉 All GPT model tests passed!');
    
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