import { Matrix } from 'ml-matrix';
import { Module } from './module.js';

/**
 * Positional Encoding - adds learnable position embeddings to input
 * JavaScript equivalent of numpyGPT/nn/modules/positional.py
 */
export class PositionalEncoding extends Module {
  constructor(maxLen, dModel) {
    super();
    this.maxLen = maxLen;
    this.dModel = dModel;

    // Initialize positional embedding weights with small random values
    this.W = Matrix.random(maxLen, dModel, { random: () => this._randn() * 0.02 });
    this.dW = null;
    this.cacheInput = null;
  }

  /**
   * Generate random number from standard normal distribution
   * Box-Muller transform
   */
  _randn() {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * Forward pass: add positional embeddings to input
   * @param {Matrix} X - Input tensor (B*T, d_model) - flattened from (B, T, d_model)
   * @param {number} seqLen - Sequence length T (optional, will infer if not provided)
   * @returns {Matrix} - Output with positional encoding added
   */
  forward(X, seqLen = null) {
    this.cacheInput = X;

    const totalTokens = X.rows;
    const dModel = X.columns;

    // If sequence length not provided, assume each token is a separate sequence
    // This is a simplification - in practice, this would be provided by the model
    let T, B;
    if (seqLen !== null) {
      T = seqLen;
      B = Math.floor(totalTokens / T);
    } else {
      // For testing purposes, try to infer reasonable B and T
      // Assume square-ish batches or single sequence
      if (totalTokens <= this.maxLen) {
        T = totalTokens;
        B = 1;
      } else {
        // Try to find factors
        T = Math.min(this.maxLen, totalTokens);
        B = Math.floor(totalTokens / T);
      }
    }

    const result = Matrix.zeros(totalTokens, dModel);

    // Add positional embeddings
    for (let i = 0; i < totalTokens; i++) {
      // Calculate which position in sequence this token represents
      const posInSeq = i % T;
      
      for (let d = 0; d < dModel; d++) {
        const inputVal = X.get(i, d);
        const posVal = this.W.get(posInSeq, d);
        result.set(i, d, inputVal + posVal);
      }
    }

    // Store shape info for backward pass
    this.batchInfo = { B, T, totalTokens };
    return result;
  }

  /**
   * Backward pass: compute gradients for positional embeddings
   * @param {Matrix} dZ - Gradient from next layer
   * @returns {Matrix} - Gradient w.r.t input (unchanged)
   */
  backward(dZ) {
    const { B, T } = this.batchInfo;

    // Initialize gradient matrix
    this.dW = Matrix.zeros(this.maxLen, this.dModel);

    // out = X + W[:T, :], so ∂out/∂W[t] = 1 for every batch element at position t
    // ∂L/∂W[t] = sum over all batch gradients at position t
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < this.dModel; d++) {
        let gradSum = 0;
        for (let b = 0; b < B; b++) {
          const inputIdx = b * T + t;
          if (inputIdx < dZ.rows) {
            gradSum += dZ.get(inputIdx, d);
          }
        }
        this.dW.set(t, d, gradSum);
      }
    }

    // Return gradient w.r.t input (unchanged since addition)
    return dZ;
  }

  /**
   * Get parameters
   * @returns {Object} - Dictionary of parameters
   */
  params() {
    return {
      W: this.W,
    };
  }

  /**
   * Get gradients
   * @returns {Object} - Dictionary of gradients
   */
  grads() {
    return {
      W: this.dW,
    };
  }
}