import { LRScheduler } from './lr_scheduler.js';

/**
 * Warmup Cosine Learning Rate Scheduler
 * JavaScript equivalent of numpyGPT/optim/lr_scheduler/warmup_cosine_lr.py
 * 
 * Cosine Learning Rate Scheduler with linear warmup
 * Reference: https://arxiv.org/abs/1608.03983
 */
export class WarmupCosineLR extends LRScheduler {
  /**
   * Initialize WarmupCosineLR scheduler
   * @param {Optimizer} optimizer - The optimizer to schedule
   * @param {number} warmupIters - Number of warmup iterations
   * @param {number} lrDecayIters - Total number of decay iterations
   * @param {number} minLr - Minimum learning rate
   * @param {number} lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer, warmupIters, lrDecayIters, minLr, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    
    this.warmupIters = warmupIters;
    this.lrDecayIters = lrDecayIters;
    this.minLr = minLr;
    
    // Apply the initial learning rate
    if (lastEpoch === -1) {
      this.lastEpoch = 0;
      const lr = this.getLr()[0];
      this.optimizer.lr = lr;
    }
  }

  /**
   * Compute learning rate for current epoch
   * @returns {Array} - Array containing the learning rate
   */
  getLr() {
    const it = this.lastEpoch;
    const learningRate = this.baseLr;

    // 1) Linear warmup for warmup_iters steps
    if (it < this.warmupIters) {
      return [learningRate * (it + 1) / (this.warmupIters + 1)];
    }
    
    // 2) If it > lr_decay_iters, return min learning rate
    if (it > this.lrDecayIters) {
      return [this.minLr];
    }
    
    // 3) In between, use cosine decay down to min learning rate
    const decayRatio = (it - this.warmupIters) / (this.lrDecayIters - this.warmupIters);
    
    // Ensure decay_ratio is in valid range
    if (decayRatio < 0 || decayRatio > 1) {
      throw new Error(`Invalid decay ratio: ${decayRatio}. Should be between 0 and 1.`);
    }
    
    // Cosine coefficient ranges from 1 to 0
    const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decayRatio));
    
    return [this.minLr + coeff * (learningRate - this.minLr)];
  }

  /**
   * Get current phase of the scheduler
   * @returns {string} - Current phase: 'warmup', 'decay', or 'min'
   */
  getCurrentPhase() {
    const it = this.lastEpoch;
    
    if (it < this.warmupIters) {
      return 'warmup';
    } else if (it <= this.lrDecayIters) {
      return 'decay';
    } else {
      return 'min';
    }
  }

  /**
   * Get scheduler state for debugging/monitoring
   * @returns {Object} - Scheduler state information
   */
  getState() {
    return {
      ...super.getState(),
      warmupIters: this.warmupIters,
      lrDecayIters: this.lrDecayIters,
      minLr: this.minLr,
      currentPhase: this.getCurrentPhase(),
      progress: this.lastEpoch / this.lrDecayIters
    };
  }
}