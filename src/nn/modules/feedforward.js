import { Module } from './module.js';
import { Linear } from './linear.js';
import { ReLU } from './activation.js';

/**
 * Feed Forward Network
 * JavaScript equivalent of numpyGPT/nn/modules/feedforward.py
 * 
 * Two-layer MLP with ReLU activation: Linear -> ReLU -> Linear
 */
export class FeedForward extends Module {
  constructor(dModel, dFF) {
    super();
    this.dModel = dModel;
    this.dFF = dFF;

    this.linear1 = new Linear(dModel, dFF);
    this.relu = new ReLU();
    this.linear2 = new Linear(dFF, dModel);
  }

  /**
   * Forward pass: Linear -> ReLU -> Linear
   * @param {Matrix} X - Input tensor (B*T, d_model)
   * @returns {Matrix} - Output tensor (B*T, d_model)
   */
  forward(X) {
    let output = this.linear1.forward(X); // (B*T, d_ff)
    output = this.relu.forward(output);   // (B*T, d_ff)
    output = this.linear2.forward(output); // (B*T, d_model)
    return output;
  }

  /**
   * Backward pass: backpropagate through Linear <- ReLU <- Linear
   * @param {Matrix} dZ - Gradient from next layer
   * @returns {Matrix} - Gradient w.r.t input
   */
  backward(dZ) {
    let grad = this.linear2.backward(dZ);
    grad = this.relu.backward(grad);
    grad = this.linear1.backward(grad);
    return grad;
  }

  /**
   * Get parameters
   * @returns {Object} - Dictionary of parameters
   */
  params() {
    const params = {};
    
    const linear1Params = this.linear1.params();
    const linear2Params = this.linear2.params();
    
    Object.keys(linear1Params).forEach(key => {
      params[`linear1.${key}`] = linear1Params[key];
    });
    Object.keys(linear2Params).forEach(key => {
      params[`linear2.${key}`] = linear2Params[key];
    });
    
    return params;
  }

  /**
   * Get gradients
   * @returns {Object} - Dictionary of gradients
   */
  grads() {
    const grads = {};
    
    const linear1Grads = this.linear1.grads();
    const linear2Grads = this.linear2.grads();
    
    Object.keys(linear1Grads).forEach(key => {
      grads[`linear1.${key}`] = linear1Grads[key];
    });
    Object.keys(linear2Grads).forEach(key => {
      grads[`linear2.${key}`] = linear2Grads[key];
    });
    
    return grads;
  }
}