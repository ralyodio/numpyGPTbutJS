import { Matrix } from 'ml-matrix';
import { Module } from './module.js';
import { Linear } from './linear.js';
import { Softmax } from './activation.js';

/**
 * Multi-Head Attention mechanism
 * JavaScript equivalent of numpyGPT/nn/modules/attention.py
 * 
 * Reference: "Attention is All You Need" https://arxiv.org/abs/1706.03762
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
   * Reshape and transpose for multi-head attention
   * @param {Matrix} X - Input matrix (B*T, d_model)
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @returns {Array} - Reshaped array (B, n_heads, T, d_k)
   */
  _reshapeForMultiHead(X, B, T) {
    const XArray = X.to2DArray();
    const result = [];
    
    // Reshape from (B*T, d_model) to (B, T, n_heads, d_k) then transpose to (B, n_heads, T, d_k)
    for (let b = 0; b < B; b++) {
      const batchHeads = [];
      for (let h = 0; h < this.nHeads; h++) {
        const headSeq = [];
        for (let t = 0; t < T; t++) {
          const tokenIdx = b * T + t;
          const headData = [];
          for (let d = 0; d < this.dK; d++) {
            const featureIdx = h * this.dK + d;
            headData.push(XArray[tokenIdx][featureIdx]);
          }
          headSeq.push(headData);
        }
        batchHeads.push(headSeq);
      }
      result.push(batchHeads);
    }
    
    return result;
  }

  /**
   * Reshape back from multi-head format
   * @param {Array} X - Multi-head array (B, n_heads, T, d_k)
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @returns {Matrix} - Reshaped matrix (B*T, d_model)
   */
  _reshapeFromMultiHead(X, B, T) {
    const result = Matrix.zeros(B * T, this.dModel);
    
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const tokenIdx = b * T + t;
        for (let h = 0; h < this.nHeads; h++) {
          for (let d = 0; d < this.dK; d++) {
            const featureIdx = h * this.dK + d;
            result.set(tokenIdx, featureIdx, X[b][h][t][d]);
          }
        }
      }
    }
    
    return result;
  }

  /**
   * Matrix multiplication for 4D arrays (B, n_heads, T, d_k)
   * @param {Array} A - First matrix (B, n_heads, T, d_k)
   * @param {Array} B - Second matrix (B, n_heads, T, d_k) or (B, n_heads, d_k, T)
   * @param {boolean} transposeB - Whether to transpose B on last two dimensions
   * @returns {Array} - Result matrix
   */
  _matmul4D(A, B, transposeB = false) {
    if (!A || !A[0] || !A[0][0] || !A[0][0][0]) {
      throw new Error('Invalid matrix A structure');
    }
    if (!B || !B[0] || !B[0][0] || !B[0][0][0]) {
      throw new Error('Invalid matrix B structure');
    }
    
    const batchSize = A.length;
    const nHeads = A[0].length;
    const seqLen = A[0][0].length;
    const dK = A[0][0][0].length;
    
    // For matrix multiplication:
    // A is (B, n_heads, T, d_k)
    // B is (B, n_heads, T, d_k)
    // When transposeB=true (Q @ K^T): result is (B, n_heads, T, T)
    // When transposeB=false (attn @ V): result is (B, n_heads, T, d_k)
    const seqLen2 = transposeB ? B[0][0].length : B[0][0][0].length;
    
    const result = [];
    
    for (let b = 0; b < batchSize; b++) {
      const batchResult = [];
      for (let h = 0; h < nHeads; h++) {
        const headResult = [];
        for (let i = 0; i < seqLen; i++) {
          const rowResult = [];
          for (let j = 0; j < seqLen2; j++) {
            let sum = 0;
            for (let k = 0; k < dK; k++) {
              const aVal = A[b][h][i][k];
              // For Q @ K^T: we want K[b][h][j][k] but with j and k swapped (transpose)
              // For attn @ V: we want V[b][h][j][k] (no transpose)
              const bVal = transposeB ? B[b][h][k][j] : B[b][h][j][k];
              sum += aVal * bVal;
            }
            rowResult.push(sum);
          }
          headResult.push(rowResult);
        }
        batchResult.push(headResult);
      }
      result.push(batchResult);
    }
    
    return result;
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

    // Reshape for multi-head attention: (B*T, d_model) -> (B, n_heads, T, d_k)
    const QReshaped = this._reshapeForMultiHead(Q, B, T);
    const KReshaped = this._reshapeForMultiHead(K, B, T);
    const VReshaped = this._reshapeForMultiHead(V, B, T);

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    const scores = this._matmul4D(QReshaped, KReshaped, true); // (B, n_heads, T, T)
    const scaleFactor = 1.0 / Math.sqrt(this.dK);
    
    // Scale scores
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            scores[b][h][i][j] *= scaleFactor;
          }
        }
      }
    }

    // Apply mask if provided
    if (mask !== null) {
      for (let b = 0; b < B; b++) {
        for (let h = 0; h < this.nHeads; h++) {
          for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
              scores[b][h][i][j] += mask[i][j];
            }
          }
        }
      }
    }

    // Apply softmax to get attention weights
    const originalShape = [B, this.nHeads, T, T];
    const scoresFlat = Matrix.zeros(B * this.nHeads * T, T);
    let flatIdx = 0;
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            scoresFlat.set(flatIdx, j, scores[b][h][i][j]);
          }
          flatIdx++;
        }
      }
    }

    const attnWeightsFlat = this.softmax.forward(scoresFlat);
    
    // Reshape attention weights back to 4D
    const attnWeights = [];
    flatIdx = 0;
    for (let b = 0; b < B; b++) {
      const batchWeights = [];
      for (let h = 0; h < this.nHeads; h++) {
        const headWeights = [];
        for (let i = 0; i < T; i++) {
          const rowWeights = [];
          for (let j = 0; j < T; j++) {
            rowWeights.push(attnWeightsFlat.get(flatIdx, j));
          }
          headWeights.push(rowWeights);
          flatIdx++;
        }
        batchWeights.push(headWeights);
      }
      attnWeights.push(batchWeights);
    }

    // Apply attention to values: attn_weights @ V
    const attnOutput = this._matmul4D(attnWeights, VReshaped, false); // (B, n_heads, T, d_k)

    // Reshape back to (B*T, d_model)
    const attnOutputFlat = this._reshapeFromMultiHead(attnOutput, B, T);

    // Final linear projection
    const output = this.WO.forward(attnOutputFlat);

    // Cache for backward pass
    this.cache = {
      X,
      Q: QReshaped,
      K: KReshaped,
      V: VReshaped,
      attnWeights,
      originalShape,
      scoresFlat,
      attnWeightsFlat,
    };

    return output;
  }

  /**
   * Backward pass: Compute gradients
   * @param {Matrix} dZ - Gradient from next layer
   * @returns {Matrix} - Gradient w.r.t input
   */
  backward(dZ) {
    const { X, Q, K, V, attnWeights, originalShape, scoresFlat, attnWeightsFlat } = this.cache;
    const { B, T } = this.batchInfo;

    // Backward through output projection
    const dAttnOutputFlat = this.WO.backward(dZ);

    // Reshape to multi-head format
    const dAttnOutput = this._reshapeForMultiHead(dAttnOutputFlat, B, T);

    // Backward through attention application: attn_weights @ V
    // ∂L/∂attn_weights = ∂L/∂attn_output @ V^T
    // ∂L/∂V = attn_weights^T @ ∂L/∂attn_output
    const dAttnWeights = this._matmul4D(dAttnOutput, V, true);
    const dV = this._matmul4D(attnWeights, dAttnOutput, false); // Note: need to transpose attnWeights

    // Transpose attnWeights for proper multiplication
    const attnWeightsT = [];
    for (let b = 0; b < B; b++) {
      const batchT = [];
      for (let h = 0; h < this.nHeads; h++) {
        const headT = [];
        for (let i = 0; i < T; i++) {
          const rowT = [];
          for (let j = 0; j < T; j++) {
            rowT.push(attnWeights[b][h][j][i]); // Transpose
          }
          headT.push(rowT);
        }
        batchT.push(headT);
      }
      attnWeightsT.push(batchT);
    }
    const dVCorrected = this._matmul4D(attnWeightsT, dAttnOutput, false);

    // Flatten attention weight gradients for softmax backward
    const dAttnWeightsFlat = Matrix.zeros(B * this.nHeads * T, T);
    let flatIdx = 0;
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            dAttnWeightsFlat.set(flatIdx, j, dAttnWeights[b][h][i][j]);
          }
          flatIdx++;
        }
      }
    }

    // Backward through softmax
    const dScoresFlat = this.softmax.backward(dAttnWeightsFlat);

    // Reshape scores gradients back to 4D
    const dScores = [];
    flatIdx = 0;
    for (let b = 0; b < B; b++) {
      const batchScores = [];
      for (let h = 0; h < this.nHeads; h++) {
        const headScores = [];
        for (let i = 0; i < T; i++) {
          const rowScores = [];
          for (let j = 0; j < T; j++) {
            rowScores.push(dScoresFlat.get(flatIdx, j));
          }
          headScores.push(rowScores);
          flatIdx++;
        }
        batchScores.push(headScores);
      }
      dScores.push(batchScores);
    }

    // Scale gradients
    const scaleFactor = 1.0 / Math.sqrt(this.dK);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < this.nHeads; h++) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            dScores[b][h][i][j] *= scaleFactor;
          }
        }
      }
    }

    // Backward through Q @ K^T
    // ∂L/∂Q = ∂L/∂scores @ K
    // ∂L/∂K = Q^T @ ∂L/∂scores
    const dQ = this._matmul4D(dScores, K, false);
    const dK = this._matmul4D(Q, dScores, false); // Need to transpose Q

    // Transpose Q for proper multiplication
    const QT = [];
    for (let b = 0; b < B; b++) {
      const batchT = [];
      for (let h = 0; h < this.nHeads; h++) {
        const headT = [];
        for (let i = 0; i < this.dK; i++) {
          const rowT = [];
          for (let j = 0; j < T; j++) {
            rowT.push(Q[b][h][j][i]); // Transpose
          }
          headT.push(rowT);
        }
        batchT.push(headT);
      }
      QT.push(batchT);
    }
    const dKCorrected = this._matmul4D(QT, dScores, false);

    // Reshape gradients back to flat format
    const dQFlat = this._reshapeFromMultiHead(dQ, B, T);
    const dKFlat = this._reshapeFromMultiHead(dKCorrected, B, T);
    const dVFlat = this._reshapeFromMultiHead(dVCorrected, B, T);

    // Backward through linear projections
    const dXQ = this.WQ.backward(dQFlat);
    const dXK = this.WK.backward(dKFlat);
    const dXV = this.WV.backward(dVFlat);

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
    
    // Add parameters from all linear layers with prefixes
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
    
    // Add gradients from all linear layers with prefixes
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