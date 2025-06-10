/**
 * Base Learning Rate Scheduler class
 * JavaScript equivalent of numpyGPT/optim/lr_scheduler/lr_scheduler.py
 * 
 * Abstract base class for all learning rate schedulers
 */
export class LRScheduler {
  /**
   * Initialize learning rate scheduler
   * @param {Optimizer} optimizer - The optimizer to schedule
   * @param {number} lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer, lastEpoch = -1) {
    if (this.constructor === LRScheduler) {
      throw new TypeError('Cannot instantiate abstract class LRScheduler directly');
    }
    
    this.optimizer = optimizer;
    this.baseLr = optimizer.lr;
    this.lastEpoch = lastEpoch;
  }

  /**
   * Compute learning rate(s) (abstract method)
   * Must be implemented by subclasses
   * @returns {Array} - Array of learning rates
   */
  getLr() {
    throw new Error('getLr() method must be implemented by subclass');
  }

  /**
   * Update the learning rate
   * @param {number} epoch - Current epoch (optional)
   */
  step(epoch = null) {
    if (epoch === null) {
      epoch = this.lastEpoch + 1;
    }
    
    this.lastEpoch = epoch;
    const lr = this.getLr()[0];
    this.optimizer.lr = lr;
  }

  /**
   * Get current learning rate
   * @returns {number} - Current learning rate
   */
  getCurrentLr() {
    return this.optimizer.lr;
  }

  /**
   * Get scheduler state for debugging/monitoring
   * @returns {Object} - Scheduler state information
   */
  getState() {
    return {
      baseLr: this.baseLr,
      currentLr: this.getCurrentLr(),
      lastEpoch: this.lastEpoch
    };
  }
}