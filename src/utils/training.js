/**
 * Training Utilities for numpyGPT JavaScript Implementation
 * JavaScript equivalent of numpyGPT/utils/training.py
 * 
 * Provides professional-grade training utilities including:
 * - Structured logging system
 * - Gradient clipping utilities
 * - Learning rate extraction
 * - Training progress monitoring
 */

import { Matrix } from 'ml-matrix';

/**
 * Setup structured logger for training
 * @param {string} name - Logger name
 * @param {string} level - Log level ('info', 'debug', 'warn', 'error')
 * @returns {Object} - Logger instance
 */
export function setupLogger(name = 'train', level = 'info') {
  const levels = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3
  };
  
  const currentLevel = levels[level] || 1;
  
  const logger = {
    name,
    level: currentLevel,
    
    _log(logLevel, message, ...args) {
      if (levels[logLevel] >= this.level) {
        const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
        const levelStr = logLevel.toUpperCase().padEnd(5);
        console.log(`${timestamp} - ${levelStr} - ${message}`, ...args);
      }
    },
    
    debug(message, ...args) {
      this._log('debug', message, ...args);
    },
    
    info(message, ...args) {
      this._log('info', message, ...args);
    },
    
    warn(message, ...args) {
      this._log('warn', message, ...args);
    },
    
    error(message, ...args) {
      this._log('error', message, ...args);
    }
  };
  
  return logger;
}

/**
 * Clip gradient norm to prevent exploding gradients
 * @param {Object} model - Model with grads() method
 * @param {number} maxNorm - Maximum gradient norm
 * @returns {number} - Total gradient norm before clipping
 */
export function clipGradNorm(model, maxNorm) {
  let totalNorm = 0.0;
  const grads = model.grads();
  
  // Calculate total gradient norm
  for (const [name, grad] of Object.entries(grads)) {
    if (grad instanceof Matrix) {
      for (let i = 0; i < grad.rows; i++) {
        for (let j = 0; j < grad.columns; j++) {
          const val = grad.get(i, j);
          totalNorm += val * val;
        }
      }
    }
  }
  
  totalNorm = Math.sqrt(totalNorm);
  
  // Clip gradients if necessary
  if (totalNorm > maxNorm) {
    const clipCoeff = maxNorm / (totalNorm + 1e-6);
    
    for (const [name, grad] of Object.entries(grads)) {
      if (grad instanceof Matrix) {
        for (let i = 0; i < grad.rows; i++) {
          for (let j = 0; j < grad.columns; j++) {
            grad.set(i, j, grad.get(i, j) * clipCoeff);
          }
        }
      }
    }
  }
  
  return totalNorm;
}

/**
 * Get current learning rate from optimizer
 * @param {Object} optimizer - Optimizer instance
 * @returns {number} - Current learning rate
 */
export function getLr(optimizer) {
  return optimizer.lr;
}

/**
 * Training Monitor for progress tracking and logging
 * Provides sophisticated training progress monitoring with timing and statistics
 */
export class TrainingMonitor {
  /**
   * Initialize training monitor
   * @param {number} logInterval - How often to log (in iterations)
   */
  constructor(logInterval = 100) {
    this.logInterval = logInterval;
    this.stepTimes = [];
    this.losses = [];
    this.startTime = Date.now();
    this.lastLogTime = Date.now();
  }
  
  /**
   * Log a training step
   * @param {number} iterNum - Current iteration number
   * @param {number} loss - Current loss value
   * @param {number} lr - Current learning rate
   * @param {number} gradNorm - Gradient norm (optional)
   * @returns {string|null} - Log message if it's time to log, null otherwise
   */
  logStep(iterNum, loss, lr, gradNorm = null) {
    const currentTime = Date.now();
    
    // Always track losses for statistics
    this.losses.push(loss);
    
    if (iterNum % this.logInterval === 0) {
      // Calculate timing statistics
      const stepTime = currentTime - this.lastLogTime;
      this.stepTimes.push(stepTime);
      this.lastLogTime = currentTime;
      
      // Build log message
      let msg = `iter ${iterNum.toString().padStart(6)} | loss ${loss.toFixed(4)} | lr ${lr.toExponential(2)}`;
      
      if (gradNorm !== null) {
        msg += ` | grad_norm ${gradNorm.toFixed(2)}`;
      }
      
      if (this.stepTimes.length > 0) {
        const msPerStep = stepTime / this.logInterval;
        msg += ` | ms/step ${msPerStep.toFixed(1)}`;
      }
      
      // Add total elapsed time
      const totalElapsed = (currentTime - this.startTime) / 1000;
      msg += ` | elapsed ${totalElapsed.toFixed(1)}s`;
      
      return msg;
    }
    
    return null;
  }
  
  /**
   * Get average loss over a window
   * @param {number} window - Window size for averaging
   * @returns {number} - Average loss
   */
  getAvgLoss(window = 100) {
    if (this.losses.length === 0) {
      return 0.0;
    }
    
    const startIdx = Math.max(0, this.losses.length - window);
    const windowLosses = this.losses.slice(startIdx);
    return windowLosses.reduce((sum, loss) => sum + loss, 0) / windowLosses.length;
  }
  
  /**
   * Get training statistics
   * @returns {Object} - Training statistics
   */
  getStats() {
    const totalTime = (Date.now() - this.startTime) / 1000;
    const avgStepTime = this.stepTimes.length > 0 
      ? this.stepTimes.reduce((sum, time) => sum + time, 0) / this.stepTimes.length 
      : 0;
    
    return {
      totalSteps: this.losses.length,
      totalTime,
      avgStepTime,
      avgLoss: this.getAvgLoss(),
      recentLoss: this.losses.length > 0 ? this.losses[this.losses.length - 1] : 0,
      stepsPerSecond: this.losses.length / totalTime
    };
  }
  
  /**
   * Reset monitor for new training session
   */
  reset() {
    this.stepTimes = [];
    this.losses = [];
    this.startTime = Date.now();
    this.lastLogTime = Date.now();
  }
  
  /**
   * Check if training is converging
   * @param {number} window - Window size to check
   * @param {number} threshold - Convergence threshold
   * @returns {boolean} - True if converging
   */
  isConverging(window = 50, threshold = 1e-4) {
    if (this.losses.length < window * 2) {
      return false;
    }
    
    const recent = this.getAvgLoss(window);
    const earlier = this.losses.slice(-window * 2, -window);
    const earlierAvg = earlier.reduce((sum, loss) => sum + loss, 0) / earlier.length;
    
    const improvement = earlierAvg - recent;
    return improvement > threshold;
  }
  
  /**
   * Detect if training has stalled
   * @param {number} window - Window size to check
   * @param {number} threshold - Stall threshold
   * @returns {boolean} - True if stalled
   */
  isStalled(window = 100, threshold = 1e-5) {
    if (this.losses.length < window) {
      return false;
    }
    
    const recentLosses = this.losses.slice(-window);
    const minLoss = Math.min(...recentLosses);
    const maxLoss = Math.max(...recentLosses);
    const variation = maxLoss - minLoss;
    
    return variation < threshold;
  }
}

/**
 * Early stopping utility
 */
export class EarlyStopping {
  /**
   * Initialize early stopping
   * @param {number} patience - Number of epochs to wait for improvement
   * @param {number} minDelta - Minimum change to qualify as improvement
   * @param {boolean} restore - Whether to restore best weights
   */
  constructor(patience = 10, minDelta = 1e-4, restore = true) {
    this.patience = patience;
    this.minDelta = minDelta;
    this.restore = restore;
    this.bestScore = Infinity;
    this.counter = 0;
    this.bestWeights = null;
  }
  
  /**
   * Check if should stop early
   * @param {number} score - Current validation score (lower is better)
   * @param {Object} model - Model to save weights from
   * @returns {boolean} - True if should stop
   */
  shouldStop(score, model = null) {
    if (score < this.bestScore - this.minDelta) {
      this.bestScore = score;
      this.counter = 0;
      
      // Save best weights if model provided
      if (model && this.restore) {
        this.bestWeights = this._saveWeights(model);
      }
      
      return false;
    } else {
      this.counter++;
      return this.counter >= this.patience;
    }
  }
  
  /**
   * Restore best weights to model
   * @param {Object} model - Model to restore weights to
   */
  restoreBestWeights(model) {
    if (this.bestWeights && this.restore) {
      this._loadWeights(model, this.bestWeights);
    }
  }
  
  /**
   * Save model weights (simplified implementation)
   * @param {Object} model - Model to save weights from
   * @returns {Object} - Saved weights
   */
  _saveWeights(model) {
    const params = model.params();
    const weights = {};
    
    for (const [name, param] of Object.entries(params)) {
      if (param instanceof Matrix) {
        weights[name] = param.clone();
      }
    }
    
    return weights;
  }
  
  /**
   * Load model weights (simplified implementation)
   * @param {Object} model - Model to load weights into
   * @param {Object} weights - Weights to load
   */
  _loadWeights(model, weights) {
    const params = model.params();
    
    for (const [name, savedParam] of Object.entries(weights)) {
      if (params[name] && savedParam instanceof Matrix) {
        // Copy values
        for (let i = 0; i < savedParam.rows; i++) {
          for (let j = 0; j < savedParam.columns; j++) {
            params[name].set(i, j, savedParam.get(i, j));
          }
        }
      }
    }
  }
}

/**
 * Learning rate finder utility
 */
export class LRFinder {
  /**
   * Initialize LR finder
   * @param {Object} model - Model to test
   * @param {Object} optimizer - Optimizer to use
   * @param {number} startLr - Starting learning rate
   * @param {number} endLr - Ending learning rate
   * @param {number} numSteps - Number of steps to test
   */
  constructor(model, optimizer, startLr = 1e-7, endLr = 1, numSteps = 100) {
    this.model = model;
    this.optimizer = optimizer;
    this.startLr = startLr;
    this.endLr = endLr;
    this.numSteps = numSteps;
    this.lrs = [];
    this.losses = [];
  }
  
  /**
   * Find optimal learning rate
   * @param {Function} trainStep - Function that performs one training step
   * @returns {Object} - Results with learning rates and losses
   */
  find(trainStep) {
    // Save original learning rate
    const originalLr = this.optimizer.lr;
    
    // Generate learning rate schedule
    const lrMultiplier = Math.pow(this.endLr / this.startLr, 1 / this.numSteps);
    
    for (let step = 0; step < this.numSteps; step++) {
      const lr = this.startLr * Math.pow(lrMultiplier, step);
      this.optimizer.lr = lr;
      
      // Perform training step
      const loss = trainStep();
      
      this.lrs.push(lr);
      this.losses.push(loss);
      
      // Stop if loss explodes
      if (loss > this.losses[0] * 4) {
        break;
      }
    }
    
    // Restore original learning rate
    this.optimizer.lr = originalLr;
    
    return {
      learningRates: this.lrs,
      losses: this.losses,
      suggestedLr: this._findOptimalLr()
    };
  }
  
  /**
   * Find optimal learning rate from results
   * @returns {number} - Suggested learning rate
   */
  _findOptimalLr() {
    if (this.losses.length < 2) {
      return this.startLr;
    }
    
    // Find steepest descent
    let maxGradient = 0;
    let optimalIdx = 0;
    
    for (let i = 1; i < this.losses.length - 1; i++) {
      const gradient = (this.losses[i - 1] - this.losses[i + 1]) / 2;
      if (gradient > maxGradient) {
        maxGradient = gradient;
        optimalIdx = i;
      }
    }
    
    return this.lrs[optimalIdx];
  }
}