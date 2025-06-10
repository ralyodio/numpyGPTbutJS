import { Matrix } from 'ml-matrix';

/**
 * Base Optimizer class
 * JavaScript equivalent of numpyGPT/optim/optimizer.py
 * 
 * Abstract base class for all optimizers
 */
export class Optimizer {
  /**
   * Initialize optimizer
   * @param {Array} modules - Array of modules to optimize
   * @param {number} lr - Learning rate
   */
  constructor(modules, lr = 0.001) {
    if (this.constructor === Optimizer) {
      throw new TypeError('Cannot instantiate abstract class Optimizer directly');
    }
    
    this.params = modules;
    this.lr = lr;
  }

  /**
   * Perform optimization step (abstract method)
   * Must be implemented by subclasses
   */
  step() {
    throw new Error('step() method must be implemented by subclass');
  }

  /**
   * Zero out all gradients
   */
  zeroGrad() {
    for (const module of this.params) {
      const grads = module.grads();
      
      for (const [gradKey, grad] of Object.entries(grads)) {
        if (grad !== null && grad instanceof Matrix) {
          // Fill matrix with zeros
          for (let i = 0; i < grad.rows; i++) {
            for (let j = 0; j < grad.columns; j++) {
              grad.set(i, j, 0.0);
            }
          }
        }
      }
    }
  }
}