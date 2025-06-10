import { Matrix } from 'ml-matrix';
import { Module } from './module.js';
import { Linear } from './linear.js';
import { Softmax } from './activation.js';

/**
 * Simplified Multi-Head Attention mechanism
 * JavaScript equivalent of numpyGPT/nn/modules/attention.py
 * 
 * This version uses a simpler approach with 2D matrix operations
 */
export class MultiHeadAttention extends Module {
  constructor(dModel, nHeads) {
    super();
    
    if (dModel % nHeads !== 0) {
      throw new Error(`d_model (${dModel}) must be divisible by n_heads (${nHeads})`);
    }
    
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.dK = Math.floor(dModel / nHeads);

    // Linear projections for Q, K, V and output
    this.WQ = new Linear(dModel, dModel);
    this.WK = new Linear(dModel, dModel);
    this.WV = new Linear(dModel, dModel);
    this.WO = new Linear(dModel, dModel);
    
    this.softmax = new Softmax();
    this.cache = {};
  }

  /**
   * Forward pass: Multi-head attention
   * @param {Matrix} X - Input tensor (B*T, d_model)
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @param {Array} mask - Attention mask (optional)
   * @returns {Matrix} - Output tensor (B*T, d_model)
   */
  forward(X, B, T, mask = null) {
    this.batchInfo = { B, T };
    
    // Linear projections
    const Q = this.WQ.forward(X); // (B*T, d_model)
    const K = this.WK.forward(X); // (B*T, d_model)
    const V = this.WV.forward(X); // (B*T, d_model)

    // Process each head separately
    const headOutputs = [];
    const allAttnWeights = [];
    
    for (let h = 0; h < this.nHeads; h++) {
      // Extract head-specific Q, K, V
      const QHead = this._extractHead(Q, h, B, T);
      const KHead = this._extractHead(K, h, B, T);
      const VHead = this._extractHead(V, h, B, T);
      
      // Compute attention for this head
      const { output: headOutput, attnWeights } = this._computeHeadAttention(QHead, KHead, VHead, B, T, mask);
      headOutputs.push(headOutput);
      allAttnWeights.push(attnWeights);
    }
    
    // Concatenate all heads
    const concatenated = this._concatenateHeads(headOutputs, B, T);
    
    // Final linear projection
    const output = this.WO.forward(concatenated);

    // Cache for backward pass (including attention weights for testing)
    this.cache = {
      X,
      Q,
      K,
      V,
      headOutputs,
      concatenated,
      attnWeights: allAttnWeights,
    };

    return output;
  }

  /**
   * Extract head-specific features from Q, K, or V
   * @param {Matrix} QKV - Q, K, or V matrix (B*T, d_model)
   * @param {number} headIdx - Head index
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @returns {Matrix} - Head features (B*T, d_k)
   */
  _extractHead(QKV, headIdx, B, T) {
    const startIdx = headIdx * this.dK;
    const endIdx = startIdx + this.dK;
    
    const headFeatures = Matrix.zeros(B * T, this.dK);
    
    for (let i = 0; i < B * T; i++) {
      for (let j = 0; j < this.dK; j++) {
        headFeatures.set(i, j, QKV.get(i, startIdx + j));
      }
    }
    
    return headFeatures;
  }

  /**
   * Compute attention for a single head
   * @param {Matrix} Q - Query matrix (B*T, d_k)
   * @param {Matrix} K - Key matrix (B*T, d_k)
   * @param {Matrix} V - Value matrix (B*T, d_k)
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @param {Array} mask - Attention mask (optional)
   * @returns {Object} - {output: Matrix, attnWeights: Array}
   */
  _computeHeadAttention(Q, K, V, B, T, mask) {
    // Reshape to process each sequence separately
    const outputs = [];
    const allAttnWeights = [];
    
    for (let b = 0; b < B; b++) {
      // Extract sequence for this batch
      const QSeq = Matrix.zeros(T, this.dK);
      const KSeq = Matrix.zeros(T, this.dK);
      const VSeq = Matrix.zeros(T, this.dK);
      
      for (let t = 0; t < T; t++) {
        const idx = b * T + t;
        for (let d = 0; d < this.dK; d++) {
          QSeq.set(t, d, Q.get(idx, d));
          KSeq.set(t, d, K.get(idx, d));
          VSeq.set(t, d, V.get(idx, d));
        }
      }
      
      // Compute attention scores: Q @ K^T / sqrt(d_k)
      const scores = QSeq.mmul(KSeq.transpose());
      const scaleFactor = 1.0 / Math.sqrt(this.dK);
      
      // Scale scores
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          scores.set(i, j, scores.get(i, j) * scaleFactor);
        }
      }
      
      // Apply mask if provided
      if (mask !== null) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            scores.set(i, j, scores.get(i, j) + mask[i][j]);
          }
        }
      }
      
      // Apply softmax
      const attnWeights = this.softmax.forward(scores);
      
      // Store attention weights for this batch
      allAttnWeights.push(attnWeights.to2DArray());
      
      // Apply attention to values
      const seqOutput = attnWeights.mmul(VSeq);
      outputs.push(seqOutput);
    }
    
    // Flatten back to (B*T, d_k)
    const result = Matrix.zeros(B * T, this.dK);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const idx = b * T + t;
        for (let d = 0; d < this.dK; d++) {
          result.set(idx, d, outputs[b].get(t, d));
        }
      }
    }
    
    return {
      output: result,
      attnWeights: allAttnWeights,
    };
  }

  /**
   * Concatenate outputs from all heads
   * @param {Array} headOutputs - Array of head outputs
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @returns {Matrix} - Concatenated output (B*T, d_model)
   */
  _concatenateHeads(headOutputs, B, T) {
    const result = Matrix.zeros(B * T, this.dModel);
    
    for (let i = 0; i < B * T; i++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let d = 0; d < this.dK; d++) {
          const featureIdx = h * this.dK + d;
          result.set(i, featureIdx, headOutputs[h].get(i, d));
        }
      }
    }
    
    return result;
  }

  /**
   * Backward pass: Compute gradients
   * @param {Matrix} dZ - Gradient from next layer
   * @returns {Matrix} - Gradient w.r.t input
   */
  backward(dZ) {
    const { X, Q, K, V } = this.cache;
    const { B, T } = this.batchInfo;

    // Backward through output projection
    const dConcatenated = this.WO.backward(dZ);

    // Split gradients back to heads
    const dHeadOutputs = [];
    for (let h = 0; h < this.nHeads; h++) {
      const dHead = Matrix.zeros(B * T, this.dK);
      for (let i = 0; i < B * T; i++) {
        for (let d = 0; d < this.dK; d++) {
          const featureIdx = h * this.dK + d;
          dHead.set(i, d, dConcatenated.get(i, featureIdx));
        }
      }
      dHeadOutputs.push(dHead);
    }

    // Backward through each head (simplified - just pass gradients through)
    // In a full implementation, we'd compute gradients through attention mechanism
    
    // For now, distribute gradients back to Q, K, V
    const dQ = Matrix.zeros(B * T, this.dModel);
    const dK = Matrix.zeros(B * T, this.dModel);
    const dV = Matrix.zeros(B * T, this.dModel);
    
    for (let h = 0; h < this.nHeads; h++) {
      const startIdx = h * this.dK;
      for (let i = 0; i < B * T; i++) {
        for (let d = 0; d < this.dK; d++) {
          // Simplified gradient distribution
          const grad = dHeadOutputs[h].get(i, d) / 3; // Distribute equally to Q, K, V
          dQ.set(i, startIdx + d, grad);
          dK.set(i, startIdx + d, grad);
          dV.set(i, startIdx + d, grad);
        }
      }
    }

    // Backward through linear projections
    const dXQ = this.WQ.backward(dQ);
    const dXK = this.WK.backward(dK);
    const dXV = this.WV.backward(dV);

    // Sum gradients from all paths
    const dX = Matrix.zeros(dXQ.rows, dXQ.columns);
    for (let i = 0; i < dXQ.rows; i++) {
      for (let j = 0; j < dXQ.columns; j++) {
        const grad = dXQ.get(i, j) + dXK.get(i, j) + dXV.get(i, j);
        dX.set(i, j, grad);
      }
    }

    return dX;
  }

  /**
   * Get parameters
   * @returns {Object} - Dictionary of parameters
   */
  params() {
    const params = {};
    
    const wqParams = this.WQ.params();
    const wkParams = this.WK.params();
    const wvParams = this.WV.params();
    const woParams = this.WO.params();
    
    Object.keys(wqParams).forEach(key => {
      params[`W_q.${key}`] = wqParams[key];
    });
    Object.keys(wkParams).forEach(key => {
      params[`W_k.${key}`] = wkParams[key];
    });
    Object.keys(wvParams).forEach(key => {
      params[`W_v.${key}`] = wvParams[key];
    });
    Object.keys(woParams).forEach(key => {
      params[`W_o.${key}`] = woParams[key];
    });
    
    return params;
  }

  /**
   * Get gradients
   * @returns {Object} - Dictionary of gradients
   */
  grads() {
    const grads = {};
    
    const wqGrads = this.WQ.grads();
    const wkGrads = this.WK.grads();
    const wvGrads = this.WV.grads();
    const woGrads = this.WO.grads();
    
    Object.keys(wqGrads).forEach(key => {
      grads[`W_q.${key}`] = wqGrads[key];
    });
    Object.keys(wkGrads).forEach(key => {
      grads[`W_k.${key}`] = wkGrads[key];
    });
    Object.keys(wvGrads).forEach(key => {
      grads[`W_v.${key}`] = wvGrads[key];
    });
    Object.keys(woGrads).forEach(key => {
      grads[`W_o.${key}`] = woGrads[key];
    });
    
    return grads;
  }
}