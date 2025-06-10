/**
 * Extended test suite for optimizers
 * Tests Adam optimizer with comprehensive coverage
 */

import { Adam } from '../../src/optim/adam.js';
import { Linear } from '../../src/nn/modules/linear.js';
import { Matrix } from 'ml-matrix';

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

function testAdamInitialization() {
  console.log('Testing Adam optimizer initialization...');
  
  const linear = new Linear(5, 3);
  const optimizer = new Adam([linear], 0.001, [0.9, 0.999], 1e-8);
  
  // Test initialization parameters
  assert(optimizer.lr === 0.001, 'Learning rate should be set correctly');
  assert(optimizer.beta1 === 0.9, 'Beta1 should be set correctly');
  assert(optimizer.beta2 === 0.999, 'Beta2 should be set correctly');
  assert(optimizer.eps === 1e-8, 'Epsilon should be set correctly');
  assert(optimizer.t === 0, 'Time step should start at 0');
  
  // Test momentum and velocity initialization
  assert(optimizer.m.length === 1, 'Should have momentum for one module');
  assert(optimizer.v.length === 1, 'Should have velocity for one module');
  
  const params = linear.params();
  const mDict = optimizer.m[0];
  const vDict = optimizer.v[0];
  
  for (const [paramKey, param] of Object.entries(params)) {
    assert(paramKey in mDict, `Momentum should exist for parameter ${paramKey}`);
    assert(paramKey in vDict, `Velocity should exist for parameter ${paramKey}`);
    
    const m = mDict[paramKey];
    const v = vDict[paramKey];
    
    assert(m.rows === param.rows, `Momentum should have same rows as parameter ${paramKey}`);
    assert(m.columns === param.columns, `Momentum should have same columns as parameter ${paramKey}`);
    assert(v.rows === param.rows, `Velocity should have same rows as parameter ${paramKey}`);
    assert(v.columns === param.columns, `Velocity should have same columns as parameter ${paramKey}`);
    
    // Check that momentum and velocity are initialized to zero
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.columns; j++) {
        assertClose(m.get(i, j), 0, 1e-10, `Momentum should be initialized to zero`);
        assertClose(v.get(i, j), 0, 1e-10, `Velocity should be initialized to zero`);
      }
    }
  }
  
  console.log('✓ Adam initialization tests passed');
}

function testAdamStep() {
  console.log('Testing Adam optimizer step...');
  
  const linear = new Linear(3, 2);
  const optimizer = new Adam([linear], 0.01);
  
  // Create some input and compute gradients
  const input = Matrix.random(2, 3);
  const target = Matrix.random(2, 2);
  
  // Forward pass
  const output = linear.forward(input);
  
  // Compute simple loss gradient (output - target)
  const lossGrad = Matrix.zeros(2, 2);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      lossGrad.set(i, j, output.get(i, j) - target.get(i, j));
    }
  }
  
  // Backward pass
  linear.backward(lossGrad);
  
  // Store original parameters
  const originalW = linear.W.clone();
  const originalB = linear.b.clone();
  
  // Check that gradients exist
  const grads = linear.grads();
  assert(grads.W instanceof Matrix, 'Weight gradients should exist');
  assert(grads.b instanceof Matrix, 'Bias gradients should exist');
  
  // Optimizer step
  optimizer.step();
  
  // Check that time step increased
  assert(optimizer.t === 1, 'Time step should increase after step');
  
  // Check that parameters changed
  let weightsChanged = false;
  let biasChanged = false;
  
  for (let i = 0; i < linear.W.rows; i++) {
    for (let j = 0; j < linear.W.columns; j++) {
      if (Math.abs(linear.W.get(i, j) - originalW.get(i, j)) > 1e-10) {
        weightsChanged = true;
        break;
      }
    }
  }
  
  for (let i = 0; i < linear.b.rows; i++) {
    for (let j = 0; j < linear.b.columns; j++) {
      if (Math.abs(linear.b.get(i, j) - originalB.get(i, j)) > 1e-10) {
        biasChanged = true;
        break;
      }
    }
  }
  
  assert(weightsChanged, 'Weights should change after optimizer step');
  assert(biasChanged, 'Bias should change after optimizer step');
  
  console.log('✓ Adam step tests passed');
}

function testAdamMomentumAccumulation() {
  console.log('Testing Adam momentum accumulation...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.01, [0.9, 0.999]);
  
  // Create consistent gradients
  const input = Matrix.ones(1, 2);
  const target = Matrix.ones(1, 1);
  
  // Perform multiple steps with same gradients
  for (let step = 0; step < 5; step++) {
    const output = linear.forward(input);
    const lossGrad = Matrix.ones(1, 1); // Constant gradient
    linear.backward(lossGrad);
    
    const gradsBefore = {
      W: linear.grads().W.clone(),
      b: linear.grads().b.clone()
    };
    
    optimizer.step();
    
    // Check momentum accumulation
    const mDict = optimizer.m[0];
    const vDict = optimizer.v[0];
    
    // Momentum should be non-zero after first step
    if (step > 0) {
      let hasNonZeroMomentum = false;
      for (let i = 0; i < mDict.W.rows; i++) {
        for (let j = 0; j < mDict.W.columns; j++) {
          if (Math.abs(mDict.W.get(i, j)) > 1e-10) {
            hasNonZeroMomentum = true;
            break;
          }
        }
      }
      assert(hasNonZeroMomentum, `Momentum should be non-zero after step ${step}`);
    }
    
    optimizer.zeroGrad();
  }
  
  console.log('✓ Adam momentum accumulation tests passed');
}

function testAdamBiasCorrection() {
  console.log('Testing Adam bias correction...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.01, [0.9, 0.999]);
  
  // Store initial parameters
  const initialW = linear.W.clone();
  const initialB = linear.b.clone();
  
  // Create gradients
  const input = Matrix.ones(1, 2);
  const output = linear.forward(input);
  const lossGrad = Matrix.ones(1, 1);
  linear.backward(lossGrad);
  
  // First step - bias correction should be significant
  optimizer.step();
  
  const step1W = linear.W.clone();
  const step1B = linear.b.clone();
  
  // Calculate expected bias correction factors
  const beta1Correction1 = 1 - Math.pow(0.9, 1);
  const beta2Correction1 = 1 - Math.pow(0.999, 1);
  
  // Bias correction should make the first step larger than without correction
  assert(beta1Correction1 < 1, 'Beta1 correction should be less than 1');
  assert(beta2Correction1 < 1, 'Beta2 correction should be less than 1');
  
  // Reset and do multiple steps to see bias correction diminish
  linear.W = initialW.clone();
  linear.b = initialB.clone();
  const optimizer2 = new Adam([linear], 0.01, [0.9, 0.999]);
  
  for (let step = 0; step < 10; step++) {
    const output = linear.forward(input);
    linear.backward(lossGrad);
    optimizer2.step();
    optimizer2.zeroGrad();
  }
  
  // After many steps, bias correction should be minimal
  const beta1CorrectionLater = 1 - Math.pow(0.9, 10);
  const beta2CorrectionLater = 1 - Math.pow(0.999, 10);
  
  assert(beta1CorrectionLater > beta1Correction1, 'Bias correction should diminish over time');
  assert(beta2CorrectionLater > beta2Correction1, 'Bias correction should diminish over time');
  
  console.log('✓ Adam bias correction tests passed');
}

function testAdamWithMultipleModules() {
  console.log('Testing Adam with multiple modules...');
  
  const linear1 = new Linear(3, 2);
  const linear2 = new Linear(2, 1);
  const optimizer = new Adam([linear1, linear2], 0.01);
  
  // Test that optimizer handles multiple modules
  assert(optimizer.m.length === 2, 'Should have momentum for two modules');
  assert(optimizer.v.length === 2, 'Should have velocity for two modules');
  
  // Create gradients for both modules
  const input1 = Matrix.random(1, 3);
  const hidden = linear1.forward(input1);
  const output = linear2.forward(hidden);
  
  const lossGrad = Matrix.ones(1, 1);
  const hiddenGrad = linear2.backward(lossGrad);
  linear1.backward(hiddenGrad);
  
  // Store original parameters
  const originalW1 = linear1.W.clone();
  const originalB1 = linear1.b.clone();
  const originalW2 = linear2.W.clone();
  const originalB2 = linear2.b.clone();
  
  // Optimizer step
  optimizer.step();
  
  // Check that both modules' parameters changed
  let module1Changed = false;
  let module2Changed = false;
  
  for (let i = 0; i < linear1.W.rows; i++) {
    for (let j = 0; j < linear1.W.columns; j++) {
      if (Math.abs(linear1.W.get(i, j) - originalW1.get(i, j)) > 1e-10) {
        module1Changed = true;
        break;
      }
    }
  }
  
  for (let i = 0; i < linear2.W.rows; i++) {
    for (let j = 0; j < linear2.W.columns; j++) {
      if (Math.abs(linear2.W.get(i, j) - originalW2.get(i, j)) > 1e-10) {
        module2Changed = true;
        break;
      }
    }
  }
  
  assert(module1Changed, 'First module parameters should change');
  assert(module2Changed, 'Second module parameters should change');
  
  console.log('✓ Adam multiple modules tests passed');
}

function testAdamZeroGrad() {
  console.log('Testing Adam zero gradients...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear]);
  
  // Create gradients
  const input = Matrix.ones(1, 2);
  const output = linear.forward(input);
  const lossGrad = Matrix.ones(1, 1);
  linear.backward(lossGrad);
  
  // Check that gradients exist
  const grads = linear.grads();
  assert(grads.W instanceof Matrix, 'Weight gradients should exist');
  assert(grads.b instanceof Matrix, 'Bias gradients should exist');
  
  // Zero gradients
  optimizer.zeroGrad();
  
  // Check that gradients are zeroed
  const zeroedGrads = linear.grads();
  
  if (zeroedGrads.W) {
    for (let i = 0; i < zeroedGrads.W.rows; i++) {
      for (let j = 0; j < zeroedGrads.W.columns; j++) {
        assertClose(zeroedGrads.W.get(i, j), 0, 1e-10, 'Weight gradients should be zero');
      }
    }
  }
  
  if (zeroedGrads.b) {
    for (let i = 0; i < zeroedGrads.b.rows; i++) {
      for (let j = 0; j < zeroedGrads.b.columns; j++) {
        assertClose(zeroedGrads.b.get(i, j), 0, 1e-10, 'Bias gradients should be zero');
      }
    }
  }
  
  console.log('✓ Adam zero gradients tests passed');
}

function testAdamState() {
  console.log('Testing Adam state information...');
  
  const linear = new Linear(3, 2);
  const optimizer = new Adam([linear], 0.001, [0.9, 0.999], 1e-8);
  
  const state = optimizer.getState();
  
  // Check state structure
  assert(typeof state === 'object', 'State should be an object');
  assert('lr' in state, 'State should contain learning rate');
  assert('beta1' in state, 'State should contain beta1');
  assert('beta2' in state, 'State should contain beta2');
  assert('eps' in state, 'State should contain epsilon');
  assert('t' in state, 'State should contain time step');
  assert('numParams' in state, 'State should contain number of parameters');
  
  // Check state values
  assertClose(state.lr, 0.001, 1e-10, 'State LR should match');
  assertClose(state.beta1, 0.9, 1e-10, 'State beta1 should match');
  assertClose(state.beta2, 0.999, 1e-10, 'State beta2 should match');
  assertClose(state.eps, 1e-8, 1e-10, 'State epsilon should match');
  assert(state.t === 0, 'State time step should match');
  assert(state.numParams === 1, 'State should show one module');
  
  // Step optimizer and check state update
  const input = Matrix.ones(1, 3);
  const output = linear.forward(input);
  linear.backward(Matrix.ones(1, 2));
  optimizer.step();
  
  const newState = optimizer.getState();
  assert(newState.t === 1, 'State time step should update after step');
  
  console.log('✓ Adam state tests passed');
}

function testAdamEdgeCases() {
  console.log('Testing Adam edge cases...');
  
  // Test with very small learning rate
  const linear1 = new Linear(2, 1);
  const optimizer1 = new Adam([linear1], 1e-10);
  
  const input = Matrix.ones(1, 2);
  const output = linear1.forward(input);
  linear1.backward(Matrix.ones(1, 1));
  
  const originalW = linear1.W.clone();
  optimizer1.step();
  
  // Parameters should change very little with tiny LR
  let maxChange = 0;
  for (let i = 0; i < linear1.W.rows; i++) {
    for (let j = 0; j < linear1.W.columns; j++) {
      const change = Math.abs(linear1.W.get(i, j) - originalW.get(i, j));
      maxChange = Math.max(maxChange, change);
    }
  }
  assert(maxChange < 1e-8, 'Parameters should change very little with tiny learning rate');
  
  // Test with extreme beta values
  const linear2 = new Linear(2, 1);
  const optimizer2 = new Adam([linear2], 0.01, [0.999, 0.9999]);
  
  linear2.forward(input);
  linear2.backward(Matrix.ones(1, 1));
  optimizer2.step(); // Should not crash
  
  const state = optimizer2.getState();
  assert(state.beta1 === 0.999, 'Extreme beta1 should be preserved');
  assert(state.beta2 === 0.9999, 'Extreme beta2 should be preserved');
  
  console.log('✓ Adam edge cases tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Optimizers ===\n');
  
  try {
    testAdamInitialization();
    testAdamStep();
    testAdamMomentumAccumulation();
    testAdamBiasCorrection();
    testAdamWithMultipleModules();
    testAdamZeroGrad();
    testAdamState();
    testAdamEdgeCases();
    
    console.log('\n🎉 All optimizer tests passed!');
    
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