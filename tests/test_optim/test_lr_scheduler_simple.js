import { LRScheduler } from '../../src/optim/lr_scheduler/lr_scheduler.js';
import { StepLR } from '../../src/optim/lr_scheduler/step_lr.js';
import { WarmupCosineLR } from '../../src/optim/lr_scheduler/warmup_cosine_lr.js';
import { Adam } from '../../src/optim/adam.js';
import { Linear } from '../../src/nn/modules/linear.js';

/**
 * Test suite for Learning Rate Schedulers
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

// Test Base LRScheduler
function testLRSchedulerAbstract() {
  console.log('Testing LRScheduler abstract class...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.01);
  
  // Should not be able to instantiate abstract class
  try {
    new LRScheduler(optimizer);
    throw new Error('Should not be able to instantiate abstract LRScheduler');
  } catch (error) {
    if (error.message.includes('Cannot instantiate abstract class')) {
      console.log('✓ Abstract class protection works');
    } else {
      throw error;
    }
  }
  
  console.log('✓ LRScheduler abstract test passed');
}

// Test StepLR Scheduler
function testStepLRBasic() {
  console.log('Testing StepLR basic functionality...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new StepLR(optimizer, 2, 0.5); // Decay by 0.5 every 2 epochs
  
  // Check initial state
  assertClose(scheduler.baseLr, 0.1, 1e-10, 'Base learning rate');
  assertClose(scheduler.stepSize, 2, 1e-10, 'Step size');
  assertClose(scheduler.gamma, 0.5, 1e-10, 'Gamma');
  assertClose(scheduler.lastEpoch, -1, 1e-10, 'Initial epoch');
  
  // Test learning rate calculation at different epochs
  const expectedLrs = [
    { epoch: 0, lr: 0.1 },      // No decay yet
    { epoch: 1, lr: 0.1 },      // No decay yet
    { epoch: 2, lr: 0.05 },     // First decay: 0.1 * 0.5^1
    { epoch: 3, lr: 0.05 },     // Still first decay
    { epoch: 4, lr: 0.025 },    // Second decay: 0.1 * 0.5^2
    { epoch: 5, lr: 0.025 },    // Still second decay
  ];
  
  for (const { epoch, lr } of expectedLrs) {
    scheduler.step(epoch);
    const actualLr = scheduler.getCurrentLr();
    assertClose(actualLr, lr, 1e-10, `Learning rate at epoch ${epoch}`);
    assertClose(scheduler.lastEpoch, epoch, 1e-10, `Last epoch at step ${epoch}`);
  }
  
  console.log('✓ StepLR basic test passed');
}

function testStepLRState() {
  console.log('Testing StepLR state...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.01);
  const scheduler = new StepLR(optimizer, 3, 0.1);
  
  scheduler.step(5); // Epoch 5, should have 1 step (5 // 3 = 1)
  
  const state = scheduler.getState();
  assertClose(state.baseLr, 0.01, 1e-10, 'State base LR');
  assertClose(state.stepSize, 3, 1e-10, 'State step size');
  assertClose(state.gamma, 0.1, 1e-10, 'State gamma');
  assertClose(state.numSteps, 1, 1e-10, 'State number of steps');
  assertClose(state.lastEpoch, 5, 1e-10, 'State last epoch');
  
  console.log('✓ StepLR state test passed');
}

// Test WarmupCosineLR Scheduler
function testWarmupCosineLRBasic() {
  console.log('Testing WarmupCosineLR basic functionality...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new WarmupCosineLR(optimizer, 5, 20, 0.01); // 5 warmup, 20 total, min 0.01
  
  // Check initial state
  assertClose(scheduler.baseLr, 0.1, 1e-10, 'Base learning rate');
  assertClose(scheduler.warmupIters, 5, 1e-10, 'Warmup iterations');
  assertClose(scheduler.lrDecayIters, 20, 1e-10, 'Decay iterations');
  assertClose(scheduler.minLr, 0.01, 1e-10, 'Minimum learning rate');
  
  console.log('✓ WarmupCosineLR basic test passed');
}

function testWarmupCosineLRWarmupPhase() {
  console.log('Testing WarmupCosineLR warmup phase...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new WarmupCosineLR(optimizer, 4, 20, 0.01, -1);
  
  // Test warmup phase (linear increase)
  const warmupTests = [
    { epoch: 0, expectedPhase: 'warmup' },
    { epoch: 1, expectedPhase: 'warmup' },
    { epoch: 2, expectedPhase: 'warmup' },
    { epoch: 3, expectedPhase: 'warmup' },
  ];
  
  for (const { epoch, expectedPhase } of warmupTests) {
    scheduler.step(epoch);
    const phase = scheduler.getCurrentPhase();
    if (phase !== expectedPhase) {
      throw new Error(`Expected phase ${expectedPhase} at epoch ${epoch}, got ${phase}`);
    }
    
    // During warmup, LR should increase linearly
    const lr = scheduler.getCurrentLr();
    const expectedLr = 0.1 * (epoch + 1) / (4 + 1); // lr * (it + 1) / (warmup_iters + 1)
    assertClose(lr, expectedLr, 1e-6, `Warmup LR at epoch ${epoch}`);
    assertFinite(lr, `LR should be finite at epoch ${epoch}`);
  }
  
  console.log('✓ WarmupCosineLR warmup phase test passed');
}

function testWarmupCosineLRDecayPhase() {
  console.log('Testing WarmupCosineLR decay phase...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new WarmupCosineLR(optimizer, 2, 10, 0.01, -1);
  
  // Test decay phase (cosine decay)
  const decayTests = [
    { epoch: 2, expectedPhase: 'decay' },
    { epoch: 5, expectedPhase: 'decay' },
    { epoch: 8, expectedPhase: 'decay' },
    { epoch: 10, expectedPhase: 'decay' },
  ];
  
  for (const { epoch, expectedPhase } of decayTests) {
    scheduler.step(epoch);
    const phase = scheduler.getCurrentPhase();
    if (phase !== expectedPhase) {
      throw new Error(`Expected phase ${expectedPhase} at epoch ${epoch}, got ${phase}`);
    }
    
    const lr = scheduler.getCurrentLr();
    assertFinite(lr, `LR should be finite at epoch ${epoch}`);
    
    // LR should be between min_lr and base_lr
    if (lr < scheduler.minLr || lr > scheduler.baseLr) {
      throw new Error(`LR ${lr} out of range [${scheduler.minLr}, ${scheduler.baseLr}] at epoch ${epoch}`);
    }
  }
  
  console.log('✓ WarmupCosineLR decay phase test passed');
}

function testWarmupCosineLRMinPhase() {
  console.log('Testing WarmupCosineLR min phase...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new WarmupCosineLR(optimizer, 2, 5, 0.01, -1);
  
  // Test min phase (after decay_iters)
  const minTests = [
    { epoch: 6, expectedPhase: 'min' },
    { epoch: 10, expectedPhase: 'min' },
    { epoch: 100, expectedPhase: 'min' },
  ];
  
  for (const { epoch, expectedPhase } of minTests) {
    scheduler.step(epoch);
    const phase = scheduler.getCurrentPhase();
    if (phase !== expectedPhase) {
      throw new Error(`Expected phase ${expectedPhase} at epoch ${epoch}, got ${phase}`);
    }
    
    const lr = scheduler.getCurrentLr();
    assertClose(lr, scheduler.minLr, 1e-10, `Min LR at epoch ${epoch}`);
  }
  
  console.log('✓ WarmupCosineLR min phase test passed');
}

function testWarmupCosineLRState() {
  console.log('Testing WarmupCosineLR state...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new WarmupCosineLR(optimizer, 5, 20, 0.01, -1);
  
  scheduler.step(10); // Middle of decay phase
  
  const state = scheduler.getState();
  assertClose(state.baseLr, 0.1, 1e-10, 'State base LR');
  assertClose(state.warmupIters, 5, 1e-10, 'State warmup iters');
  assertClose(state.lrDecayIters, 20, 1e-10, 'State decay iters');
  assertClose(state.minLr, 0.01, 1e-10, 'State min LR');
  assertClose(state.lastEpoch, 10, 1e-10, 'State last epoch');
  assertClose(state.progress, 0.5, 1e-10, 'State progress');
  
  if (state.currentPhase !== 'decay') {
    throw new Error(`Expected decay phase, got ${state.currentPhase}`);
  }
  
  console.log('✓ WarmupCosineLR state test passed');
}

// Test Scheduler Integration with Optimizer
function testSchedulerOptimizerIntegration() {
  console.log('Testing scheduler-optimizer integration...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.1);
  const scheduler = new StepLR(optimizer, 2, 0.5);
  
  // Initial LR should match optimizer
  assertClose(optimizer.lr, 0.1, 1e-10, 'Initial optimizer LR');
  
  // Step scheduler and check optimizer LR updates
  scheduler.step(0);
  assertClose(optimizer.lr, 0.1, 1e-10, 'Optimizer LR after step 0');
  
  scheduler.step(2);
  assertClose(optimizer.lr, 0.05, 1e-10, 'Optimizer LR after step 2');
  
  scheduler.step(4);
  assertClose(optimizer.lr, 0.025, 1e-10, 'Optimizer LR after step 4');
  
  console.log('✓ Scheduler-optimizer integration test passed');
}

// Test Edge Cases
function testSchedulerEdgeCases() {
  console.log('Testing scheduler edge cases...');
  
  const linear = new Linear(2, 1);
  const optimizer = new Adam([linear], 0.001);
  
  // Test StepLR with step_size = 1 (decay every epoch)
  const stepScheduler = new StepLR(optimizer, 1, 0.9);
  stepScheduler.step(5);
  const expectedLr = 0.001 * Math.pow(0.9, 5);
  assertClose(optimizer.lr, expectedLr, 1e-10, 'StepLR with step_size=1');
  
  // Test WarmupCosineLR with no warmup
  const cosineScheduler = new WarmupCosineLR(optimizer, 0, 10, 0.0001, -1);
  cosineScheduler.step(5); // Middle of decay
  const lr = cosineScheduler.getCurrentLr();
  assertFinite(lr, 'WarmupCosineLR with no warmup');
  if (lr < 0.0001 || lr > 0.001) {
    throw new Error(`LR out of expected range: ${lr}`);
  }
  
  console.log('✓ Scheduler edge cases test passed');
}

// Run all tests
function runAllTests() {
  console.log('=== Testing Learning Rate Schedulers ===\n');
  
  try {
    testLRSchedulerAbstract();
    testStepLRBasic();
    testStepLRState();
    testWarmupCosineLRBasic();
    testWarmupCosineLRWarmupPhase();
    testWarmupCosineLRDecayPhase();
    testWarmupCosineLRMinPhase();
    testWarmupCosineLRState();
    testSchedulerOptimizerIntegration();
    testSchedulerEdgeCases();
    
    console.log('\n🎉 All learning rate scheduler tests passed!');
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