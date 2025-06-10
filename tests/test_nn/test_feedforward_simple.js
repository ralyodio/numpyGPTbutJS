import { Matrix } from 'ml-matrix';
import { FeedForward } from '../../src/nn/modules/feedforward.js';

/**
 * Simple test runner for FeedForward module
 * Tests the feed-forward network functionality
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

function testFeedForward() {
  console.log('🧪 Testing FeedForward...');

  // Test 1: Constructor
  console.log('  ✓ Testing constructor...');
  const dModel = 8;
  const dFF = 16;
  const ff = new FeedForward(dModel, dFF);
  
  assert(ff.dModel === dModel, 'dModel should be set correctly');
  assert(ff.dFF === dFF, 'dFF should be set correctly');
  assert(ff.linear1 !== undefined, 'linear1 should be initialized');
  assert(ff.relu !== undefined, 'relu should be initialized');
  assert(ff.linear2 !== undefined, 'linear2 should be initialized');

  // Test 2: Forward pass
  console.log('  ✓ Testing forward pass...');
  const B = 2;
  const T = 4;
  const input = Matrix.ones(B * T, dModel);
  
  const output = ff.forward(input);
  
  assert(output.rows === B * T, 'Output should have correct number of rows');
  assert(output.columns === dModel, 'Output should have correct number of columns');
  
  // Check that output is different from input (transformation occurred)
  let isDifferent = false;
  for (let i = 0; i < output.rows && !isDifferent; i++) {
    for (let j = 0; j < output.columns && !isDifferent; j++) {
      if (Math.abs(output.get(i, j) - input.get(i, j)) > 1e-6) {
        isDifferent = true;
      }
    }
  }
  assert(isDifferent, 'Output should be different from input after transformation');

  // Test 3: Backward pass
  console.log('  ✓ Testing backward pass...');
  const dZ = Matrix.ones(B * T, dModel);
  const dX = ff.backward(dZ);
  
  assert(dX.rows === B * T, 'Input gradient should have correct rows');
  assert(dX.columns === dModel, 'Input gradient should have correct columns');
  
  // Check that gradients are computed for both linear layers
  const grads = ff.grads();
  assert(grads['linear1.W'] !== undefined, 'Linear1 weight gradients should exist');
  assert(grads['linear1.b'] !== undefined, 'Linear1 bias gradients should exist');
  assert(grads['linear2.W'] !== undefined, 'Linear2 weight gradients should exist');
  assert(grads['linear2.b'] !== undefined, 'Linear2 bias gradients should exist');

  // Test 4: Parameter management
  console.log('  ✓ Testing parameter management...');
  const params = ff.params();
  assert(params['linear1.W'] !== undefined, 'Linear1 weights should be accessible');
  assert(params['linear1.b'] !== undefined, 'Linear1 bias should be accessible');
  assert(params['linear2.W'] !== undefined, 'Linear2 weights should be accessible');
  assert(params['linear2.b'] !== undefined, 'Linear2 bias should be accessible');
  
  // Verify parameter shapes
  assert(params['linear1.W'].rows === dModel, 'Linear1 weight matrix should have correct input dim');
  assert(params['linear1.W'].columns === dFF, 'Linear1 weight matrix should have correct output dim');
  assert(params['linear2.W'].rows === dFF, 'Linear2 weight matrix should have correct input dim');
  assert(params['linear2.W'].columns === dModel, 'Linear2 weight matrix should have correct output dim');

  // Test 5: Gradient shapes
  console.log('  ✓ Testing gradient shapes...');
  assert(grads['linear1.W'].rows === dModel, 'Linear1 weight gradients should have correct shape');
  assert(grads['linear1.W'].columns === dFF, 'Linear1 weight gradients should have correct shape');
  assert(grads['linear2.W'].rows === dFF, 'Linear2 weight gradients should have correct shape');
  assert(grads['linear2.W'].columns === dModel, 'Linear2 weight gradients should have correct shape');

  // Test 6: Different input sizes
  console.log('  ✓ Testing different input sizes...');
  
  // Single token
  const singleInput = Matrix.ones(1, dModel);
  const singleOutput = ff.forward(singleInput);
  assert(singleOutput.rows === 1, 'Single token output should have 1 row');
  assert(singleOutput.columns === dModel, 'Single token output should have correct columns');
  
  // Larger batch
  const largeInput = Matrix.ones(16, dModel);
  const largeOutput = ff.forward(largeInput);
  assert(largeOutput.rows === 16, 'Large batch output should have correct rows');
  assert(largeOutput.columns === dModel, 'Large batch output should have correct columns');

  // Test 7: ReLU activation effect
  console.log('  ✓ Testing ReLU activation effect...');
  
  // Create input that will produce negative values after first linear layer
  const negativeInput = Matrix.zeros(2, dModel);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < dModel; j++) {
      negativeInput.set(i, j, -1); // All negative values
    }
  }
  
  const negativeOutput = ff.forward(negativeInput);
  
  // The output should exist and be valid (ReLU should handle negative intermediate values)
  assert(negativeOutput.rows === 2, 'Negative input output should have correct rows');
  assert(negativeOutput.columns === dModel, 'Negative input output should have correct columns');

  // Test 8: Gradient flow
  console.log('  ✓ Testing gradient flow...');
  
  // Test with different gradient magnitudes
  const gradInput = Matrix.zeros(B * T, dModel);
  for (let i = 0; i < B * T; i++) {
    for (let j = 0; j < dModel; j++) {
      gradInput.set(i, j, (i + 1) * (j + 1)); // Different gradient values
    }
  }
  
  // Forward pass first
  ff.forward(input);
  
  // Then backward pass
  const gradOutput = ff.backward(gradInput);
  
  assert(gradOutput.rows === B * T, 'Gradient output should have correct rows');
  assert(gradOutput.columns === dModel, 'Gradient output should have correct columns');
  
  // Check that gradients are non-zero (indicating proper flow)
  let hasNonZeroGrad = false;
  for (let i = 0; i < gradOutput.rows && !hasNonZeroGrad; i++) {
    for (let j = 0; j < gradOutput.columns && !hasNonZeroGrad; j++) {
      if (Math.abs(gradOutput.get(i, j)) > 1e-6) {
        hasNonZeroGrad = true;
      }
    }
  }
  assert(hasNonZeroGrad, 'Should have non-zero gradients indicating proper flow');

  console.log('✅ All FeedForward tests passed!');
}

// Run the tests
try {
  testFeedForward();
  console.log('🎉 All tests completed successfully!');
} catch (error) {
  console.error('❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}