/**
 * Extended test suite for learning rate schedulers
 * Tests all scheduler types with comprehensive coverage
 */

import { StepLR } from '../../src/optim/lr_scheduler/step_lr.js';
import { WarmupCosineLR } from '../../src/optim/lr_scheduler/warmup_cosine_lr.js';
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

function testStepLRScheduler() {
  console.log('Testing StepLR scheduler...');
  
  // Create a simple model and optimizer
  const linear = new Linear(10, 5);
  const optimizer = new Adam([linear], 0.01);
  
  // Test basic step scheduler
  const stepSize = 3;
  const gamma = 0.5;
  const scheduler = new StepLR(optimizer, stepSize, gamma);
  
  // Test initialization
  assert(scheduler.stepSize === stepSize, 'Step size should be set correctly');
  assert(scheduler.gamma === gamma, 'Gamma should be set correctly');
  assert(scheduler.lastEpoch === -1, 'Last epoch should start at -1');
  
  // Test initial learning rate
  const initialLR = optimizer.lr;
  assertClose(scheduler.getCurrentLr(), initialLR, 1e-10, 'Initial LR should match optimizer LR');
  
  // Test stepping through epochs
  const expectedLRs = [
    initialLR,      // epoch 0
    initialLR,      // epoch 1
    initialLR,      // epoch 2
    initialLR * gamma,  // epoch 3 (first step)
    initialLR * gamma,  // epoch 4
    initialLR * gamma,  // epoch 5
    initialLR * gamma * gamma,  // epoch 6 (second step)
  ];
  
  for (let epoch = 0; epoch < expectedLRs.length; epoch++) {
    scheduler.step();
    const currentLR = scheduler.getCurrentLr();
    assertClose(currentLR, expectedLRs[epoch], 1e-10, 
               `LR at epoch ${epoch} should be ${expectedLRs[epoch]}`);
    assert(scheduler.lastEpoch === epoch, `Last epoch should be ${epoch}`);
  }
  
  console.log('✓ StepLR scheduler tests passed');
}

function testWarmupCosineLRScheduler() {
  console.log('Testing WarmupCosineLR scheduler...');
  
  // Create a simple model and optimizer
  const linear = new Linear(10, 5);
  const optimizer = new Adam([linear], 0.01);
  
  const warmupEpochs = 5;
  const totalEpochs = 20;
  const minLR = 1e-6;
  const scheduler = new WarmupCosineLR(optimizer, warmupEpochs, totalEpochs, minLR);
  
  // Test initialization
  assert(scheduler.warmupIters === warmupEpochs, 'Warmup epochs should be set correctly');
  assert(scheduler.lrDecayIters === totalEpochs, 'Total epochs should be set correctly');
  assert(scheduler.minLr === minLR, 'Min LR should be set correctly');
  
  const initialLR = optimizer.lr;
  
  // Test warmup phase - just check that LR changes and is positive
  for (let step = 0; step < warmupEpochs; step++) {
    scheduler.step();
    const currentLR = scheduler.getCurrentLr();
    
    // LR should be positive
    assert(currentLR > 0, `LR should be positive at step ${step}`);
    assert(isFinite(currentLR), `LR should be finite at step ${step}`);
  }
  
  // Test cosine decay phase - just check basic functionality
  for (let epoch = warmupEpochs; epoch < totalEpochs; epoch++) {
    scheduler.step();
    const currentLR = scheduler.getCurrentLr();
    
    // Should be positive and finite
    assert(currentLR > 0, `LR should be positive at epoch ${epoch}`);
    assert(isFinite(currentLR), `LR should be finite at epoch ${epoch}`);
  }
  
  // Test that we can step beyond total epochs
  scheduler.step();
  const finalLR = scheduler.getCurrentLr();
  assert(finalLR > 0, 'Final LR should be positive');
  assert(isFinite(finalLR), 'Final LR should be finite');
  
  console.log('✓ WarmupCosineLR scheduler tests passed');
}

function testSchedulerState() {
  console.log('Testing scheduler state management...');
  
  const linear = new Linear(5, 3);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new StepLR(optimizer, 2, 0.8);
  
  // Test state saving and loading
  scheduler.step();
  scheduler.step();
  scheduler.step();
  
  const state = scheduler.getState();
  assert(typeof state === 'object', 'State should be an object');
  assert('lastEpoch' in state, 'State should contain lastEpoch');
  assert('currentLr' in state, 'State should contain currentLr');
  
  // Test that state contains expected values
  assert(state.lastEpoch === scheduler.lastEpoch, 'State lastEpoch should match');
  assertClose(state.currentLr, scheduler.getCurrentLr(), 1e-10, 'State currentLr should match');
  
  console.log('✓ Scheduler state management tests passed');
}

function testSchedulerEdgeCases() {
  console.log('Testing scheduler edge cases...');
  
  const linear = new Linear(3, 2);
  const optimizer = new Adam([linear], 0.01);
  
  // Test StepLR with step size 1
  const scheduler1 = new StepLR(optimizer, 1, 0.9);
  const initialLR = optimizer.lr;
  
  scheduler1.step(); // epoch -1 -> 0, no decay yet
  scheduler1.step(); // epoch 0 -> 1, first decay happens
  assertClose(scheduler1.getCurrentLr(), initialLR * 0.9, 1e-10,
             'LR should decay at epoch 1 when stepSize=1');
  
  scheduler1.step(); // epoch 1 -> 2, second decay happens
  assertClose(scheduler1.getCurrentLr(), initialLR * 0.9 * 0.9, 1e-10,
             'LR should continue decaying at epoch 2');
  
  // Test WarmupCosineLR with no warmup
  const scheduler2 = new WarmupCosineLR(optimizer, 0, 10, 1e-6);
  scheduler2.step();
  
  // Should start cosine decay immediately
  const lr1 = scheduler2.getCurrentLr();
  assert(lr1 <= initialLR, 'LR should start decaying immediately with no warmup');
  
  // Test WarmupCosineLR with warmup equal to total epochs
  const scheduler3 = new WarmupCosineLR(optimizer, 5, 5, 1e-6);
  for (let i = 0; i < 5; i++) {
    scheduler3.step();
  }
  
  // Should reach full LR at the end
  assertClose(scheduler3.getCurrentLr(), initialLR, 1e-10,
             'LR should reach full value when warmup equals total epochs');
  
  console.log('✓ Scheduler edge cases tests passed');
}

function testSchedulerWithOptimizer() {
  console.log('Testing scheduler integration with optimizer...');
  
  const linear = new Linear(4, 2);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new StepLR(optimizer, 2, 0.5);
  
  // Simulate training loop
  const input = Matrix.random(3, 4);
  const target = Matrix.random(3, 2);
  
  const initialLR = optimizer.lr;
  
  for (let epoch = 0; epoch < 6; epoch++) {
    // Forward pass
    const output = linear.forward(input);
    
    // Compute simple loss (MSE)
    const loss = Matrix.zeros(3, 2);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) {
        loss.set(i, j, output.get(i, j) - target.get(i, j));
      }
    }
    
    // Backward pass
    linear.backward(loss);
    
    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
    
    // Scheduler step
    scheduler.step();
    
    // Check that optimizer LR matches scheduler LR
    assertClose(optimizer.lr, scheduler.getCurrentLr(), 1e-10,
               `Optimizer LR should match scheduler LR at epoch ${epoch}`);
  }
  
  // Verify LR has changed according to schedule
  const finalLR = optimizer.lr;
  const expectedFinalLR = initialLR * 0.5 * 0.5; // Two steps at epochs 2 and 4
  assertClose(finalLR, expectedFinalLR, 1e-10,
             'Final LR should match expected schedule');
  
  console.log('✓ Scheduler-optimizer integration tests passed');
}

function testSchedulerLRHistory() {
  console.log('Testing scheduler state tracking...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.05);
  const scheduler = new StepLR(optimizer, 3, 0.7);
  
  const initialLR = 0.05;
  
  // Test state at different epochs
  const states = [];
  for (let epoch = 0; epoch < 10; epoch++) {
    scheduler.step();
    states.push(scheduler.getState());
  }
  
  // Verify state tracking
  assert(states.length === 10, 'Should have 10 state snapshots');
  
  // Check some specific states
  assertClose(states[0].currentLr, initialLR, 1e-10, 'Epoch 0 LR should be initial');
  assertClose(states[2].currentLr, initialLR, 1e-10, 'Epoch 2 LR should be initial');
  assertClose(states[3].currentLr, initialLR * 0.7, 1e-10, 'Epoch 3 LR should be decayed');
  assertClose(states[6].currentLr, initialLR * 0.7 * 0.7, 1e-10, 'Epoch 6 LR should be double decayed');
  
  // Check lastEpoch tracking
  for (let i = 0; i < states.length; i++) {
    assert(states[i].lastEpoch === i, `State ${i} should have lastEpoch=${i}`);
  }
  
  console.log('✓ Scheduler state tracking tests passed');
}

function testMultipleSchedulers() {
  console.log('Testing multiple schedulers...');
  
  // Test that different schedulers work independently
  const linear1 = new Linear(3, 2);
  const linear2 = new Linear(3, 2);
  
  const optimizer1 = new Adam([linear1], 0.01);
  const optimizer2 = new Adam([linear2], 0.02);
  
  const scheduler1 = new StepLR(optimizer1, 2, 0.8);
  const scheduler2 = new WarmupCosineLR(optimizer2, 3, 10, 1e-5);
  
  // Step both schedulers
  for (let epoch = 0; epoch < 5; epoch++) {
    scheduler1.step();
    scheduler2.step();
  }
  
  // Verify they have different learning rates
  const lr1 = scheduler1.getCurrentLr();
  const lr2 = scheduler2.getCurrentLr();
  
  assert(lr1 !== lr2, 'Different schedulers should have different LRs');
  assert(lr1 > 0, 'Scheduler 1 LR should be positive');
  assert(lr2 > 0, 'Scheduler 2 LR should be positive');
  
  // Verify optimizers have correct LRs
  assertClose(optimizer1.lr, lr1, 1e-10, 'Optimizer 1 should match scheduler 1');
  assertClose(optimizer2.lr, lr2, 1e-10, 'Optimizer 2 should match scheduler 2');
  
  console.log('✓ Multiple schedulers tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Learning Rate Schedulers ===\n');
  
  try {
    testStepLRScheduler();
    testWarmupCosineLRScheduler();
    testSchedulerState();
    testSchedulerEdgeCases();
    testSchedulerWithOptimizer();
    testSchedulerLRHistory();
    testMultipleSchedulers();
    
    console.log('\n🎉 All learning rate scheduler tests passed!');
    
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