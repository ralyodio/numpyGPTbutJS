import { Matrix } from 'ml-matrix';
import { Module } from './module.js';
import { MultiHeadAttention } from './attention.js';
import { FeedForward } from './feedforward.js';
import { LayerNorm } from './layerNorm.js';

/**
 * Transformer Block
 * JavaScript equivalent of numpyGPT/nn/modules/transformer.py
 * 
 * Pre-norm architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
 */
export class TransformerBlock extends Module {
  constructor(dModel, nHeads, dFF) {
    super();
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.dFF = dFF;

    this.attn = new MultiHeadAttention(dModel, nHeads);
    this.ln1 = new LayerNorm(dModel);
    this.ffn = new FeedForward(dModel, dFF);
    this.ln2 = new LayerNorm(dModel);
  }

  /**
   * Forward pass: Pre-norm transformer block with residual connections
   * @param {Matrix} X - Input tensor (B*T, d_model)
   * @param {number} B - Batch size
   * @param {number} T - Sequence length
   * @param {Array} mask - Attention mask (optional)
   * @returns {Matrix} - Output tensor (B*T, d_model)
   */
  forward(X, B, T, mask = null) {
    this.batchInfo = { B, T };
    this.cacheX = X;

    // First sub-layer: LayerNorm -> MultiHeadAttention -> Residual
    const ln1Out = this.ln1.forward(X);
    const attnOut = this.attn.forward(ln1Out, B, T, mask);
    
    // Residual connection: X = X + attn_out
    const X1 = this._addResidual(X, attnOut);

    // Second sub-layer: LayerNorm -> FeedForward -> Residual
    const ln2Out = this.ln2.forward(X1);
    const ffnOut = this.ffn.forward(ln2Out);
    
    // Residual connection: X = X + ffn_out
    const X2 = this._addResidual(X1, ffnOut);

    // Cache intermediate values for backward pass
    this.cache = {
      X: X,
      ln1Out: ln1Out,
      attnOut: attnOut,
      X1: X1,
      ln2Out: ln2Out,
      ffnOut: ffnOut,
    };

    return X2;
  }

  /**
   * Add residual connection
   * @param {Matrix} X - Original input
   * @param {Matrix} residual - Residual to add
   * @returns {Matrix} - X + residual
   */
  _addResidual(X, residual) {
    const result = Matrix.zeros(X.rows, X.columns);
    for (let i = 0; i < X.rows; i++) {
      for (let j = 0; j < X.columns; j++) {
        result.set(i, j, X.get(i, j) + residual.get(i, j));
      }
    }
    return result;
  }

  /**
   * Backward pass: Compute gradients through residual connections
   * @param {Matrix} dZ - Gradient from next layer
   * @returns {Matrix} - Gradient w.r.t input
   */
  backward(dZ) {
    const { X, ln1Out, attnOut, X1, ln2Out, ffnOut } = this.cache;

    // Residual: y = x + f(x), so ∂L/∂x = ∂L/∂y * I + ∂L/∂f(x) · ∂f/∂x
    // Gradient flows through both paths

    // Backward through second residual connection
    const dFFNOut = this._copyMatrix(dZ);  // Gradient to FFN output
    const dX1 = this._copyMatrix(dZ);      // Gradient through residual path

    // Backward through FFN
    const dFFNIn = this.ffn.backward(dFFNOut);
    const dLN2Out = dFFNIn;
    const dX2 = this.ln2.backward(dLN2Out);

    // Sum gradients from both paths
    const dX1Total = this._addMatrices(dX1, dX2);

    // Backward through first residual connection
    const dAttnOut = this._copyMatrix(dX1Total);  // Gradient to attention output
    const dX3 = this._copyMatrix(dX1Total);       // Gradient through residual path

    // Backward through attention
    const dAttnIn = this.attn.backward(dAttnOut);
    const dLN1Out = dAttnIn;
    const dX4 = this.ln1.backward(dLN1Out);

    // Sum gradients from both paths
    const dXFinal = this._addMatrices(dX3, dX4);

    return dXFinal;
  }

  /**
   * Copy matrix (equivalent to .copy() in NumPy)
   * @param {Matrix} mat - Matrix to copy
   * @returns {Matrix} - Copy of the matrix
   */
  _copyMatrix(mat) {
    const result = Matrix.zeros(mat.rows, mat.columns);
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.columns; j++) {
        result.set(i, j, mat.get(i, j));
      }
    }
    return result;
  }

  /**
   * Add two matrices element-wise
   * @param {Matrix} A - First matrix
   * @param {Matrix} B - Second matrix
   * @returns {Matrix} - A + B
   */
  _addMatrices(A, B) {
    const result = Matrix.zeros(A.rows, A.columns);
    for (let i = 0; i < A.rows; i++) {
      for (let j = 0; j < A.columns; j++) {
        result.set(i, j, A.get(i, j) + B.get(i, j));
      }
    }
    return result;
  }

  /**
   * Get parameters
   * @returns {Object} - Dictionary of parameters
   */
  params() {
    const params = {};
    
    const attnParams = this.attn.params();
    const ln1Params = this.ln1.params();
    const ffnParams = this.ffn.params();
    const ln2Params = this.ln2.params();
    
    Object.keys(attnParams).forEach(key => {
      params[`attn.${key}`] = attnParams[key];
    });
    Object.keys(ln1Params).forEach(key => {
      params[`ln1.${key}`] = ln1Params[key];
    });
    Object.keys(ffnParams).forEach(key => {
      params[`ffn.${key}`] = ffnParams[key];
    });
    Object.keys(ln2Params).forEach(key => {
      params[`ln2.${key}`] = ln2Params[key];
    });
    
    return params;
  }

  /**
   * Get gradients
   * @returns {Object} - Dictionary of gradients
   */
  grads() {
    const grads = {};
    
    const attnGrads = this.attn.grads();
    const ln1Grads = this.ln1.grads();
    const ffnGrads = this.ffn.grads();
    const ln2Grads = this.ln2.grads();
    
    Object.keys(attnGrads).forEach(key => {
      grads[`attn.${key}`] = attnGrads[key];
    });
    Object.keys(ln1Grads).forEach(key => {
      grads[`ln1.${key}`] = ln1Grads[key];
    });
    Object.keys(ffnGrads).forEach(key => {
      grads[`ffn.${key}`] = ffnGrads[key];
    });
    Object.keys(ln2Grads).forEach(key => {
      grads[`ln2.${key}`] = ln2Grads[key];
    });
    
    return grads;
  }
}