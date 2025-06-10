import { LRScheduler } from './lr_scheduler.js';

/**
 * Step Learning Rate Scheduler
 * JavaScript equivalent of numpyGPT/optim/lr_scheduler/step_lr.py
 * 
 * Decays the learning rate by gamma every step_size epochs
 * Reference: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L432
 */
export class StepLR extends LRScheduler {
  /**
   * Initialize StepLR scheduler
   * @param {Optimizer} optimizer - The optimizer to schedule
   * @param {number} stepSize - Period of learning rate decay
   * @param {number} gamma - Multiplicative factor of learning rate decay (default: 0.1)
   * @param {number} lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer, stepSize, gamma = 0.1, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    
    this.stepSize = stepSize;
    this.gamma = gamma;
  }

  /**
   * Compute learning rate for current epoch
   * @returns {Array} - Array containing the learning rate
   */
  getLr() {
    // Calculate how many step_size periods have passed
    const numSteps = Math.floor(this.lastEpoch / this.stepSize);
    
    // Apply gamma decay for each step
    const lr = this.baseLr * Math.pow(this.gamma, numSteps);
    
    return [lr];
  }

  /**
   * Get scheduler state for debugging/monitoring
   * @returns {Object} - Scheduler state information
   */
  getState() {
    return {
      ...super.getState(),
      stepSize: this.stepSize,
      gamma: this.gamma,
      numSteps: Math.floor(this.lastEpoch / this.stepSize)
    };
  }
}