import { Matrix } from 'ml-matrix';
import { Optimizer } from '../../src/optim/optimizer.js';
import { Adam } from '../../src/optim/adam.js';
import { Linear } from '../../src/nn/modules/linear.js';

/**
 * Test suite for Optimizers
 */

// Test utilities
function assertClose(actual, expected, tolerance = 1e-6, message = '') {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(`${message} Expected ${expected}, got ${actual}, diff: ${diff}`);
  }
}

function assertFinite(value, message = '') {
  if (!isFinite(value) || isNaN(value)) {
    throw new Error(`${message} Value should be finite, got ${value}`);
  }
}

// Test Base Optimizer
function testOptimizerAbstract() {
  console.log('Testing Optimizer abstract class...');
  
  // Should not be able to instantiate abstract class
  try {
    new Optimizer([], 0.001);
    throw new Error('Should not be able to instantiate abstract Optimizer');
  } catch (error) {
    if (error.message.includes('Cannot instantiate abstract class')) {
      console.log('✓ Abstract class protection works');
    } else {
      throw error;
    }
  }
  
  console.log('✓ Optimizer abstract test passed');
}

// Test Zero Grad Functionality
function testZeroGrad() {
  console.log('Testing zero_grad functionality...');
  
  // Create a simple module with gradients
  const linear = new Linear(3, 2);
  
  // Set some non-zero gradients manually
  const params = linear.params();
  const grads = linear.grads();
  
  // Initialize gradients with non-zero values
  for (const [key, grad] of Object.entries(grads)) {
    if (grad instanceof Matrix) {
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          grad.set(i, j, Math.random());
        }
      }
    }
  }
  
  // Create optimizer and zero gradients
  const optimizer = new Adam([linear], 0.01);
  optimizer.zeroGrad();
  
  // Check that all gradients are zero
  const zeroedGrads = linear.grads();
  for (const [key, grad] of Object.entries(zeroedGrads)) {
    if (grad instanceof Matrix) {
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          assertClose(grad.get(i, j), 0.0, 1e-10, `Gradient ${key}[${i},${j}] should be zero`);
        }
      }
    }
  }
  
  console.log('✓ Zero grad test passed');
}

// Test Adam Optimizer Initialization
function testAdamInitialization() {
  console.log('Testing Adam optimizer initialization...');
  
  const linear1 = new Linear(3, 2);
  const linear2 = new Linear(2, 1);
  const modules = [linear1, linear2];
  
  const optimizer = new Adam(modules, 0.001, [0.9, 0.999], 1e-8);
  
  // Check basic properties
  assertClose(optimizer.lr, 0.001, 1e-10, 'Learning rate');
  assertClose(optimizer.beta1, 0.9, 1e-10, 'Beta1');
  assertClose(optimizer.beta2, 0.999, 1e-10, 'Beta2');
  assertClose(optimizer.eps, 1e-8, 1e-10, 'Epsilon');
  assertClose(optimizer.t, 0, 1e-10, 'Time step should start at 0');
  
  // Check momentum and velocity buffers are initialized
  if (optimizer.m.length !== modules.length) {
    throw new Error('Momentum buffers not initialized correctly');
  }
  if (optimizer.v.length !== modules.length) {
    throw new Error('Velocity buffers not initialized correctly');
  }
  
  // Check that momentum and velocity are zero-initialized
  for (let i = 0; i < modules.length; i++) {
    const params = modules[i].params();
    for (const [key, param] of Object.entries(params)) {
      if (param instanceof Matrix) {
        const m = optimizer.m[i][key];
        const v = optimizer.v[i][key];
        
        if (!m || !v) {
          throw new Error(`Momentum/velocity not initialized for ${key}`);
        }
        
        // Check dimensions match
        if (m.rows !== param.rows || m.columns !== param.columns) {
          throw new Error(`Momentum dimensions don't match for ${key}`);
        }
        if (v.rows !== param.rows || v.columns !== param.columns) {
          throw new Error(`Velocity dimensions don't match for ${key}`);
        }
        
        // Check values are zero
        for (let row = 0; row < param.rows; row++) {
          for (let col = 0; col < param.columns; col++) {
            assertClose(m.get(row, col), 0.0, 1e-10, `Momentum ${key}[${row},${col}]`);
            assertClose(v.get(row, col), 0.0, 1e-10, `Velocity ${key}[${row},${col}]`);
          }
        }
      }
    }
  }
  
  console.log('✓ Adam initialization test passed');
}

// Test Adam Optimizer Step
function testAdamStep() {
  console.log('Testing Adam optimizer step...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1, [0.9, 0.999], 1e-8);
  
  // Get initial parameters
  const params = linear.params();
  const initialWeights = {};
  for (const [key, param] of Object.entries(params)) {
    if (param instanceof Matrix) {
      initialWeights[key] = param.clone();
    }
  }
  
  // Set some gradients manually - need to ensure gradients exist first
  // Perform a dummy forward/backward pass to initialize gradients
  const X = new Matrix([[1.0, 2.0]]);
  const output = linear.forward(X);
  const dOutput = new Matrix([[1.0]]); // Dummy gradient
  linear.backward(dOutput);
  
  // Now set specific gradient values
  const grads = linear.grads();
  for (const [key, grad] of Object.entries(grads)) {
    if (grad instanceof Matrix) {
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          grad.set(i, j, 0.1); // Small positive gradient
        }
      }
    }
  }
  
  // Perform optimization step
  optimizer.step();
  
  // Check that time step increased
  assertClose(optimizer.t, 1, 1e-10, 'Time step should increase');
  
  // Check that parameters changed
  const updatedParams = linear.params();
  for (const [key, param] of Object.entries(updatedParams)) {
    if (param instanceof Matrix) {
      const initial = initialWeights[key];
      for (let i = 0; i < param.rows; i++) {
        for (let j = 0; j < param.columns; j++) {
          const initialVal = initial.get(i, j);
          const updatedVal = param.get(i, j);
          
          // Parameters should have decreased (gradient descent)
          if (updatedVal >= initialVal) {
            throw new Error(`Parameter ${key}[${i},${j}] should have decreased`);
          }
          
          assertFinite(updatedVal, `Parameter ${key}[${i},${j}] should be finite`);
        }
      }
    }
  }
  
  console.log('✓ Adam step test passed');
}

// Test Adam Multiple Steps
function testAdamMultipleSteps() {
  console.log('Testing Adam optimizer multiple steps...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.01, [0.9, 0.999], 1e-8);
  
  // Initialize gradients first
  const X = new Matrix([[1.0, 2.0]]);
  const output = linear.forward(X);
  const dOutput = new Matrix([[1.0]]);
  linear.backward(dOutput);
  
  // Perform multiple optimization steps
  for (let step = 0; step < 5; step++) {
    // Set consistent gradients
    const grads = linear.grads();
    for (const [key, grad] of Object.entries(grads)) {
      if (grad instanceof Matrix) {
        for (let i = 0; i < grad.rows; i++) {
          for (let j = 0; j < grad.columns; j++) {
            grad.set(i, j, 0.1);
          }
        }
      }
    }
    
    optimizer.step();
    
    // Check time step
    assertClose(optimizer.t, step + 1, 1e-10, `Time step at iteration ${step + 1}`);
    
    // Check that all parameters are finite
    const params = linear.params();
    for (const [key, param] of Object.entries(params)) {
      if (param instanceof Matrix) {
        for (let i = 0; i < param.rows; i++) {
          for (let j = 0; j < param.columns; j++) {
            assertFinite(param.get(i, j), `Parameter ${key}[${i},${j}] at step ${step + 1}`);
          }
        }
      }
    }
  }
  
  console.log('✓ Adam multiple steps test passed');
}

// Test Adam State
function testAdamState() {
  console.log('Testing Adam optimizer state...');
  
  const linear = new Linear(2, 1);
  const lr = 0.001;
  const betas = [0.9, 0.999];
  const eps = 1e-8;
  
  const optimizer = new Adam([linear], lr, betas, eps);
  
  const state = optimizer.getState();
  
  assertClose(state.lr, lr, 1e-10, 'State learning rate');
  assertClose(state.beta1, betas[0], 1e-10, 'State beta1');
  assertClose(state.beta2, betas[1], 1e-10, 'State beta2');
  assertClose(state.eps, eps, 1e-10, 'State epsilon');
  assertClose(state.t, 0, 1e-10, 'State time step');
  assertClose(state.numParams, 1, 1e-10, 'State number of parameter groups');
  
  console.log('✓ Adam state test passed');
}

// Test Adam with Multiple Modules
function testAdamMultipleModules() {
  console.log('Testing Adam with multiple modules...');
  
  const linear1 = new Linear(3, 2);
  const linear2 = new Linear(2, 1);
  const modules = [linear1, linear2];
  
  const optimizer = new Adam(modules, 0.01);
  
  // Initialize gradients for all modules first
  let X = new Matrix([[1.0, 2.0, 3.0]]);
  for (const module of modules) {
    const output = module.forward(X);
    const dOutput = Matrix.ones(output.rows, output.columns);
    module.backward(dOutput);
    X = output; // Chain the modules for next iteration
  }
  
  // Set gradients for all modules
  for (const module of modules) {
    const grads = module.grads();
    for (const [key, grad] of Object.entries(grads)) {
      if (grad instanceof Matrix) {
        for (let i = 0; i < grad.rows; i++) {
          for (let j = 0; j < grad.columns; j++) {
            grad.set(i, j, 0.1);
          }
        }
      }
    }
  }
  
  // Perform optimization step
  optimizer.step();
  
  // Check that all modules were updated
  for (const module of modules) {
    const params = module.params();
    for (const [key, param] of Object.entries(params)) {
      if (param instanceof Matrix) {
        for (let i = 0; i < param.rows; i++) {
          for (let j = 0; j < param.columns; j++) {
            assertFinite(param.get(i, j), `Parameter ${key}[${i},${j}] should be finite`);
          }
        }
      }
    }
  }
  
  console.log('✓ Adam multiple modules test passed');
}

// Run all tests
function runAllTests() {
  console.log('=== Testing Optimizers ===\n');
  
  try {
    testOptimizerAbstract();
    testZeroGrad();
    testAdamInitialization();
    testAdamStep();
    testAdamMultipleSteps();
    testAdamState();
    testAdamMultipleModules();
    
    console.log('\n🎉 All optimizer tests passed!');
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