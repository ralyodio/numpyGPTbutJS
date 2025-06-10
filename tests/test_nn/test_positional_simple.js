import { Matrix } from 'ml-matrix';
import { PositionalEncoding } from '../../src/nn/modules/positional.js';

/**
 * Simple test runner for PositionalEncoding module
 * Tests basic functionality without Jest complexity
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

function testPositionalEncoding() {
  console.log('🧪 Testing PositionalEncoding...');

  // Test 1: Constructor
  console.log('  ✓ Testing constructor...');
  const maxLen = 10;
  const dModel = 4;
  const posEnc = new PositionalEncoding(maxLen, dModel);
  
  assert(posEnc.maxLen === maxLen, 'maxLen should be set correctly');
  assert(posEnc.dModel === dModel, 'dModel should be set correctly');
  assert(posEnc.W.rows === maxLen, 'Weight matrix should have correct rows');
  assert(posEnc.W.columns === dModel, 'Weight matrix should have correct columns');

  // Test 2: Forward pass
  console.log('  ✓ Testing forward pass...');
  const B = 2;
  const T = 3;
  const input = Matrix.ones(B * T, dModel);
  
  const output = posEnc.forward(input, T);
  
  assert(output.rows === B * T, 'Output should have correct number of rows');
  assert(output.columns === dModel, 'Output should have correct number of columns');
  
  // Check that output = input + positional embeddings
  const outputArray = output.to2DArray();
  const weightsArray = posEnc.W.to2DArray();
  
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const idx = b * T + t;
      for (let d = 0; d < dModel; d++) {
        const expected = 1 + weightsArray[t][d]; // 1 (input) + positional embedding
        assertClose(outputArray[idx][d], expected, 1e-6, 
          `Output at [${idx}][${d}] should be input + positional embedding`);
      }
    }
  }

  // Test 3: Backward pass
  console.log('  ✓ Testing backward pass...');
  const dZ = Matrix.ones(B * T, dModel);
  const dX = posEnc.backward(dZ);
  
  assert(dX.rows === B * T, 'Input gradient should have correct rows');
  assert(dX.columns === dModel, 'Input gradient should have correct columns');
  
  // Check that input gradient is unchanged (addition operation)
  const dXArray = dX.to2DArray();
  for (let i = 0; i < B * T; i++) {
    for (let j = 0; j < dModel; j++) {
      assertClose(dXArray[i][j], 1, 1e-6, 'Input gradient should be unchanged');
    }
  }
  
  // Check positional embedding gradients
  assert(posEnc.dW !== null, 'Positional embedding gradients should be computed');
  assert(posEnc.dW.rows === maxLen, 'Gradient matrix should have correct rows');
  assert(posEnc.dW.columns === dModel, 'Gradient matrix should have correct columns');
  
  const dWArray = posEnc.dW.to2DArray();
  // For positions 0, 1, 2 (T=3), gradient should be sum over batch dimension
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < dModel; d++) {
      assertClose(dWArray[t][d], B, 1e-6, 
        `Gradient at position ${t} should be sum over batch dimension`);
    }
  }
  
  // For unused positions (t >= T), gradient should be 0
  for (let t = T; t < maxLen; t++) {
    for (let d = 0; d < dModel; d++) {
      assertClose(dWArray[t][d], 0, 1e-6, 
        `Gradient at unused position ${t} should be zero`);
    }
  }

  // Test 4: Parameter management
  console.log('  ✓ Testing parameter management...');
  const params = posEnc.params();
  assert(params.hasOwnProperty('W'), 'Parameters should include W');
  assert(params.W === posEnc.W, 'Parameter W should reference the weight matrix');
  
  const grads = posEnc.grads();
  assert(grads.hasOwnProperty('W'), 'Gradients should include W');
  assert(grads.W === posEnc.dW, 'Gradient W should reference the gradient matrix');

  // Test 5: Edge cases
  console.log('  ✓ Testing edge cases...');
  
  // Single token
  const singleInput = Matrix.ones(1, dModel);
  const singleOutput = posEnc.forward(singleInput);
  assert(singleOutput.rows === 1, 'Single token output should have 1 row');
  assert(singleOutput.columns === dModel, 'Single token output should have correct columns');
  
  // Maximum sequence length
  const maxInput = Matrix.ones(maxLen, dModel);
  const maxOutput = posEnc.forward(maxInput);
  assert(maxOutput.rows === maxLen, 'Max length output should have maxLen rows');
  assert(maxOutput.columns === dModel, 'Max length output should have correct columns');

  console.log('✅ All PositionalEncoding tests passed!');
}

// Run the tests
try {
  testPositionalEncoding();
  console.log('🎉 All tests completed successfully!');
} catch (error) {
  console.error('❌ Test failed:', error.message);
  process.exit(1);
}