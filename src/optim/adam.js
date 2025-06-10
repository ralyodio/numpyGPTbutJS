import { Matrix } from 'ml-matrix';
import { Optimizer } from './optimizer.js';

/**
 * Adam Optimizer
 * JavaScript equivalent of numpyGPT/optim/adam.py
 * 
 * Adam optimizer: https://arxiv.org/abs/1412.6980
 * Adaptive Moment Estimation
 */
export class Adam extends Optimizer {
  /**
   * Initialize Adam optimizer
   * @param {Array} modules - Array of modules to optimize
   * @param {number} lr - Learning rate (default: 0.001)
   * @param {Array} betas - Coefficients for computing running averages (default: [0.9, 0.999])
   * @param {number} eps - Term added to denominator for numerical stability (default: 1e-8)
   */
  constructor(modules, lr = 0.001, betas = [0.9, 0.999], eps = 1e-8) {
    super(modules, lr);
    
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.t = 0; // Time step
    
    // Initialize momentum and velocity buffers
    this.m = []; // First moment estimates
    this.v = []; // Second moment estimates
    
    for (const module of this.params) {
      const mDict = {};
      const vDict = {};
      const params = module.params();
      
      for (const [paramKey, param] of Object.entries(params)) {
        if (param instanceof Matrix) {
          // Initialize momentum and velocity with zeros
          mDict[paramKey] = Matrix.zeros(param.rows, param.columns);
          vDict[paramKey] = Matrix.zeros(param.rows, param.columns);
        }
      }
      
      this.m.push(mDict);
      this.v.push(vDict);
    }
  }

  /**
   * Perform one optimization step
   */
  step() {
    this.t += 1;

    for (let i = 0; i < this.params.length; i++) {
      const module = this.params[i];
      const params = module.params();
      const grads = module.grads();

      for (const [paramKey, param] of Object.entries(params)) {
        const grad = grads[paramKey];
        
        // Skip if gradient is null or not a Matrix
        if (!grad || !(grad instanceof Matrix)) {
          continue;
        }

        // Get momentum and velocity for this parameter
        const m = this.m[i][paramKey];
        const v = this.v[i][paramKey];

        // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
        for (let row = 0; row < param.rows; row++) {
          for (let col = 0; col < param.columns; col++) {
            const g = grad.get(row, col);
            const mVal = this.beta1 * m.get(row, col) + (1 - this.beta1) * g;
            m.set(row, col, mVal);
          }
        }

        // Update biased second raw moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        for (let row = 0; row < param.rows; row++) {
          for (let col = 0; col < param.columns; col++) {
            const g = grad.get(row, col);
            const vVal = this.beta2 * v.get(row, col) + (1 - this.beta2) * (g * g);
            v.set(row, col, vVal);
          }
        }

        // Compute bias-corrected first moment estimate: m̂_t = m_t / (1 - β1^t)
        const beta1Correction = 1 - Math.pow(this.beta1, this.t);
        
        // Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1 - β2^t)
        const beta2Correction = 1 - Math.pow(this.beta2, this.t);

        // Update parameters: θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + eps)
        for (let row = 0; row < param.rows; row++) {
          for (let col = 0; col < param.columns; col++) {
            const mHat = m.get(row, col) / beta1Correction;
            const vHat = v.get(row, col) / beta2Correction;
            
            const update = this.lr * mHat / (Math.sqrt(vHat) + this.eps);
            const newVal = param.get(row, col) - update;
            param.set(row, col, newVal);
          }
        }
      }
    }
  }

  /**
   * Get optimizer state for debugging/monitoring
   * @returns {Object} - Optimizer state information
   */
  getState() {
    return {
      lr: this.lr,
      beta1: this.beta1,
      beta2: this.beta2,
      eps: this.eps,
      t: this.t,
      numParams: this.params.length
    };
  }
}