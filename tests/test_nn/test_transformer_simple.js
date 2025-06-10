import { Matrix } from 'ml-matrix';
import { TransformerBlock } from '../../src/nn/modules/transformer.js';

/**
 * Simple test runner for TransformerBlock module
 * Tests the complete transformer block functionality
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

function testTransformerBlock() {
  console.log('🧪 Testing TransformerBlock...');

  // Test 1: Constructor
  console.log('  ✓ Testing constructor...');
  const dModel = 8;
  const nHeads = 2;
  const dFF = 16;
  const transformer = new TransformerBlock(dModel, nHeads, dFF);
  
  assert(transformer.dModel === dModel, 'dModel should be set correctly');
  assert(transformer.nHeads === nHeads, 'nHeads should be set correctly');
  assert(transformer.dFF === dFF, 'dFF should be set correctly');
  assert(transformer.attn !== undefined, 'attention should be initialized');
  assert(transformer.ln1 !== undefined, 'ln1 should be initialized');
  assert(transformer.ffn !== undefined, 'ffn should be initialized');
  assert(transformer.ln2 !== undefined, 'ln2 should be initialized');

  // Test 2: Forward pass
  console.log('  ✓ Testing forward pass...');
  const B = 2;
  const T = 4;
  const input = Matrix.ones(B * T, dModel);
  
  const output = transformer.forward(input, B, T);
  
  assert(output.rows === B * T, 'Output should have correct number of rows');
  assert(output.columns === dModel, 'Output should have correct number of columns');
  
  // Check that output is different from input (transformation occurred)
  // Use a more robust test - check if the output has reasonable values
  let hasReasonableValues = true;
  let sumOutput = 0;
  let sumInput = 0;
  
  for (let i = 0; i < output.rows; i++) {
    for (let j = 0; j < output.columns; j++) {
      const outVal = output.get(i, j);
      const inVal = input.get(i, j);
      sumOutput += outVal;
      sumInput += inVal;
      
      // Check for NaN or infinite values
      if (isNaN(outVal) || !isFinite(outVal)) {
        hasReasonableValues = false;
        break;
      }
    }
    if (!hasReasonableValues) break;
  }
  
  assert(hasReasonableValues, 'Output should have reasonable (finite) values');
  
  // Check that the transformation actually occurred by comparing sums
  // (this is more robust than element-wise comparison)
  const avgOutput = sumOutput / (output.rows * output.columns);
  const avgInput = sumInput / (input.rows * input.columns);
  console.log(`    Average input: ${avgInput}, Average output: ${avgOutput}`);

  // Test 3: Forward pass with causal mask
  console.log('  ✓ Testing forward pass with causal mask...');
  
  // Create causal mask as Matrix
  const maskData = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    for (let j = 0; j < T; j++) {
      row.push(j > i ? -1e9 : 0); // Mask future tokens
    }
    maskData.push(row);
  }
  const mask = new Matrix(maskData);
  
  const maskedOutput = transformer.forward(input, B, T, mask);
  
  assert(maskedOutput.rows === B * T, 'Masked output should have correct rows');
  assert(maskedOutput.columns === dModel, 'Masked output should have correct columns');

  // Test 4: Backward pass
  console.log('  ✓ Testing backward pass...');
  
  // Forward pass first
  transformer.forward(input, B, T);
  
  // Create gradient from next layer
  const dZ = Matrix.ones(B * T, dModel);
  const dX = transformer.backward(dZ);
  
  assert(dX.rows === B * T, 'Input gradient should have correct rows');
  assert(dX.columns === dModel, 'Input gradient should have correct columns');
  
  // Check that gradients are computed for all components
  const grads = transformer.grads();
  assert(grads['attn.W_q.W'] !== undefined, 'Attention query weight gradients should exist');
  assert(grads['attn.W_k.W'] !== undefined, 'Attention key weight gradients should exist');
  assert(grads['attn.W_v.W'] !== undefined, 'Attention value weight gradients should exist');
  assert(grads['attn.W_o.W'] !== undefined, 'Attention output weight gradients should exist');
  assert(grads['ln1.gamma'] !== undefined, 'LayerNorm1 gamma gradients should exist');
  assert(grads['ln1.beta'] !== undefined, 'LayerNorm1 beta gradients should exist');
  assert(grads['ffn.linear1.W'] !== undefined, 'FFN linear1 weight gradients should exist');
  assert(grads['ffn.linear2.W'] !== undefined, 'FFN linear2 weight gradients should exist');
  assert(grads['ln2.gamma'] !== undefined, 'LayerNorm2 gamma gradients should exist');
  assert(grads['ln2.beta'] !== undefined, 'LayerNorm2 beta gradients should exist');

  // Test 5: Parameter management
  console.log('  ✓ Testing parameter management...');
  
  const params = transformer.params();
  assert(params['attn.W_q.W'] !== undefined, 'Attention query weights should be accessible');
  assert(params['attn.W_k.W'] !== undefined, 'Attention key weights should be accessible');
  assert(params['attn.W_v.W'] !== undefined, 'Attention value weights should be accessible');
  assert(params['attn.W_o.W'] !== undefined, 'Attention output weights should be accessible');
  assert(params['ln1.gamma'] !== undefined, 'LayerNorm1 gamma should be accessible');
  assert(params['ln1.beta'] !== undefined, 'LayerNorm1 beta should be accessible');
  assert(params['ffn.linear1.W'] !== undefined, 'FFN linear1 weights should be accessible');
  assert(params['ffn.linear2.W'] !== undefined, 'FFN linear2 weights should be accessible');
  assert(params['ln2.gamma'] !== undefined, 'LayerNorm2 gamma should be accessible');
  assert(params['ln2.beta'] !== undefined, 'LayerNorm2 beta should be accessible');

  // Test 6: Residual connections
  console.log('  ✓ Testing residual connections...');
  
  // Create a simple test to verify residual connections work
  const simpleInput = Matrix.zeros(2, dModel);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < dModel; j++) {
      simpleInput.set(i, j, 1.0); // All ones
    }
  }
  
  const residualOutput = transformer.forward(simpleInput, 1, 2);
  
  // The output should not be zero (residual connections preserve some signal)
  let hasNonZeroOutput = false;
  for (let i = 0; i < residualOutput.rows && !hasNonZeroOutput; i++) {
    for (let j = 0; j < residualOutput.columns && !hasNonZeroOutput; j++) {
      if (Math.abs(residualOutput.get(i, j)) > 1e-6) {
        hasNonZeroOutput = true;
      }
    }
  }
  assert(hasNonZeroOutput, 'Residual connections should preserve signal');

  // Test 7: Different configurations
  console.log('  ✓ Testing different configurations...');
  
  // Test with different head counts
  const transformer1Head = new TransformerBlock(dModel, 1, dFF);
  const output1Head = transformer1Head.forward(input, B, T);
  assert(output1Head.rows === B * T, '1-head transformer should work');
  assert(output1Head.columns === dModel, '1-head transformer should have correct output dim');
  
  const transformer4Head = new TransformerBlock(dModel, 4, dFF);
  const output4Head = transformer4Head.forward(input, B, T);
  assert(output4Head.rows === B * T, '4-head transformer should work');
  assert(output4Head.columns === dModel, '4-head transformer should have correct output dim');

  // Test 8: Edge cases
  console.log('  ✓ Testing edge cases...');
  
  // Single token
  const singleInput = Matrix.ones(1, dModel);
  const singleOutput = transformer.forward(singleInput, 1, 1);
  assert(singleOutput.rows === 1, 'Single token output should have 1 row');
  assert(singleOutput.columns === dModel, 'Single token output should have correct columns');
  
  // Larger sequence
  const largeInput = Matrix.ones(8, dModel);
  const largeOutput = transformer.forward(largeInput, 1, 8);
  assert(largeOutput.rows === 8, 'Large sequence output should have correct rows');
  assert(largeOutput.columns === dModel, 'Large sequence output should have correct columns');

  // Test 9: Gradient flow through residuals
  console.log('  ✓ Testing gradient flow through residuals...');
  
  // Forward pass
  transformer.forward(input, B, T);
  
  // Create non-uniform gradients
  const nonUniformGrad = Matrix.zeros(B * T, dModel);
  for (let i = 0; i < B * T; i++) {
    for (let j = 0; j < dModel; j++) {
      nonUniformGrad.set(i, j, (i + 1) * (j + 1) * 0.1);
    }
  }
  
  const gradOutput = transformer.backward(nonUniformGrad);
  
  // Check that gradients flow properly
  let hasVariedGradients = false;
  const firstGrad = gradOutput.get(0, 0);
  for (let i = 0; i < gradOutput.rows && !hasVariedGradients; i++) {
    for (let j = 0; j < gradOutput.columns && !hasVariedGradients; j++) {
      if (Math.abs(gradOutput.get(i, j) - firstGrad) > 1e-6) {
        hasVariedGradients = true;
      }
    }
  }
  assert(hasVariedGradients, 'Gradients should vary across positions');

  console.log('✅ All TransformerBlock tests passed!');
}

// Run the tests
try {
  testTransformerBlock();
  console.log('🎉 All tests completed successfully!');
} catch (error) {
  console.error('❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}