import { Matrix } from 'ml-matrix';
import { MultiHeadAttention } from '../../src/nn/modules/attention.js';

/**
 * Simple test runner for MultiHeadAttention module
 * Tests the core attention mechanism functionality
 */

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

function testMultiHeadAttention() {
  console.log('🧪 Testing MultiHeadAttention...');

  // Test 1: Constructor
  console.log('  ✓ Testing constructor...');
  const dModel = 8;
  const nHeads = 2;
  const attention = new MultiHeadAttention(dModel, nHeads);
  
  assert(attention.dModel === dModel, 'dModel should be set correctly');
  assert(attention.nHeads === nHeads, 'nHeads should be set correctly');
  assert(attention.dK === dModel / nHeads, 'dK should be d_model / n_heads');
  
  // Test invalid configuration
  try {
    new MultiHeadAttention(7, 2); // 7 is not divisible by 2
    assert(false, 'Should throw error for invalid d_model/n_heads combination');
  } catch (error) {
    assert(error.message.includes('divisible'), 'Should throw divisibility error');
  }

  // Test 2: Forward pass basic functionality
  console.log('  ✓ Testing forward pass...');
  const B = 2; // Batch size
  const T = 4; // Sequence length
  const input = Matrix.ones(B * T, dModel); // Simple input for testing
  
  const output = attention.forward(input, B, T);
  
  assert(output.rows === B * T, 'Output should have correct number of rows');
  assert(output.columns === dModel, 'Output should have correct number of columns');
  
  // Check that cache is populated
  assert(attention.cache.X !== undefined, 'Input should be cached');
  assert(attention.cache.Q !== undefined, 'Q should be cached');
  assert(attention.cache.K !== undefined, 'K should be cached');
  assert(attention.cache.V !== undefined, 'V should be cached');
  assert(attention.cache.attnWeights !== undefined, 'Attention weights should be cached');

  // Test 3: Forward pass with causal mask
  console.log('  ✓ Testing forward pass with causal mask...');
  
  // Create causal mask (upper triangular with large negative values)
  const mask = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    for (let j = 0; j < T; j++) {
      row.push(j > i ? -1e9 : 0); // Mask future tokens
    }
    mask.push(row);
  }
  
  const maskedOutput = attention.forward(input, B, T, mask);
  
  assert(maskedOutput.rows === B * T, 'Masked output should have correct rows');
  assert(maskedOutput.columns === dModel, 'Masked output should have correct columns');
  
  // Verify that attention weights respect the mask (future positions should have ~0 attention)
  const attnWeights = attention.cache.attnWeights;
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < T; i++) {
        for (let j = i + 1; j < T; j++) {
          // Future positions should have very small attention weights
          assert(attnWeights[b][h][i][j] < 1e-6, 
            `Attention weight at [${b}][${h}][${i}][${j}] should be ~0 due to causal mask`);
        }
      }
    }
  }

  // Test 4: Backward pass
  console.log('  ✓ Testing backward pass...');
  
  // Create gradient from next layer
  const dZ = Matrix.ones(B * T, dModel);
  const dX = attention.backward(dZ);
  
  assert(dX.rows === B * T, 'Input gradient should have correct rows');
  assert(dX.columns === dModel, 'Input gradient should have correct columns');
  
  // Check that gradients are computed for all linear layers
  const grads = attention.grads();
  assert(grads['W_q.W'] !== undefined, 'Query weight gradients should exist');
  assert(grads['W_q.b'] !== undefined, 'Query bias gradients should exist');
  assert(grads['W_k.W'] !== undefined, 'Key weight gradients should exist');
  assert(grads['W_k.b'] !== undefined, 'Key bias gradients should exist');
  assert(grads['W_v.W'] !== undefined, 'Value weight gradients should exist');
  assert(grads['W_v.b'] !== undefined, 'Value bias gradients should exist');
  assert(grads['W_o.W'] !== undefined, 'Output weight gradients should exist');
  assert(grads['W_o.b'] !== undefined, 'Output bias gradients should exist');

  // Test 5: Parameter management
  console.log('  ✓ Testing parameter management...');
  
  const params = attention.params();
  assert(params['W_q.W'] !== undefined, 'Query weights should be accessible');
  assert(params['W_q.b'] !== undefined, 'Query bias should be accessible');
  assert(params['W_k.W'] !== undefined, 'Key weights should be accessible');
  assert(params['W_k.b'] !== undefined, 'Key bias should be accessible');
  assert(params['W_v.W'] !== undefined, 'Value weights should be accessible');
  assert(params['W_v.b'] !== undefined, 'Value bias should be accessible');
  assert(params['W_o.W'] !== undefined, 'Output weights should be accessible');
  assert(params['W_o.b'] !== undefined, 'Output bias should be accessible');
  
  // Verify parameter shapes
  assert(params['W_q.W'].rows === dModel, 'Query weight matrix should have correct input dim');
  assert(params['W_q.W'].columns === dModel, 'Query weight matrix should have correct output dim');
  assert(params['W_k.W'].rows === dModel, 'Key weight matrix should have correct input dim');
  assert(params['W_k.W'].columns === dModel, 'Key weight matrix should have correct output dim');
  assert(params['W_v.W'].rows === dModel, 'Value weight matrix should have correct input dim');
  assert(params['W_v.W'].columns === dModel, 'Value weight matrix should have correct output dim');
  assert(params['W_o.W'].rows === dModel, 'Output weight matrix should have correct input dim');
  assert(params['W_o.W'].columns === dModel, 'Output weight matrix should have correct output dim');

  // Test 6: Attention weights properties
  console.log('  ✓ Testing attention weights properties...');
  
  // Run forward pass to get fresh attention weights
  attention.forward(input, B, T);
  const freshAttnWeights = attention.cache.attnWeights;
  
  // Check that attention weights sum to 1 for each query position
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < T; i++) {
        let sum = 0;
        for (let j = 0; j < T; j++) {
          sum += freshAttnWeights[b][h][i][j];
        }
        assertClose(sum, 1.0, 1e-5, 
          `Attention weights should sum to 1 for position [${b}][${h}][${i}]`);
      }
    }
  }

  // Test 7: Different head configurations
  console.log('  ✓ Testing different head configurations...');
  
  const attention4Head = new MultiHeadAttention(8, 4);
  const output4Head = attention4Head.forward(input, B, T);
  
  assert(output4Head.rows === B * T, '4-head output should have correct rows');
  assert(output4Head.columns === dModel, '4-head output should have correct columns');
  assert(attention4Head.dK === 2, '4-head should have d_k = 2');

  // Test 8: Edge cases
  console.log('  ✓ Testing edge cases...');
  
  // Single token sequence
  const singleInput = Matrix.ones(1, dModel);
  const singleOutput = attention.forward(singleInput, 1, 1);
  assert(singleOutput.rows === 1, 'Single token output should have 1 row');
  assert(singleOutput.columns === dModel, 'Single token output should have correct columns');
  
  // Single head (should work like regular attention)
  const singleHeadAttn = new MultiHeadAttention(dModel, 1);
  const singleHeadOutput = singleHeadAttn.forward(input, B, T);
  assert(singleHeadOutput.rows === B * T, 'Single head output should have correct rows');
  assert(singleHeadOutput.columns === dModel, 'Single head output should have correct columns');

  console.log('✅ All MultiHeadAttention tests passed!');
}

// Run the tests
try {
  testMultiHeadAttention();
  console.log('🎉 All tests completed successfully!');
} catch (error) {
  console.error('❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}