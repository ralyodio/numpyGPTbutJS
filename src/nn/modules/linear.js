import { Matrix } from 'ml-matrix';
import { Module } from './module.js';

/**
 * Linear (fully connected) layer
 * JavaScript equivalent of numpyGPT/nn/modules/linear.py
 */
export class Linear extends Module {
    constructor(inFeatures, outFeatures) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        // Xavier/Glorot, Best for tanh/sigmoid, https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        // He/Kaiming, Best for ReLU/LeakyReLU, https://arxiv.org/abs/1502.01852
        // Lecun, Best for linear activations or SELU, https://arxiv.org/abs/2406.00348

        // Using He/Kaiming since we have ReLU and LeakyReLU activations
        const scale = Math.sqrt(2.0 / inFeatures);

        // Initialize weights with He initialization
        this.W = Matrix.random(inFeatures, outFeatures, { random: () => this._randn() * scale });
        this.b = Matrix.zeros(1, outFeatures);

        this.dW = null;
        this.db = null;
        this.cacheInput = null;
    }

    /**
     * Generate random number from standard normal distribution
     * Box-Muller transform
     */
    _randn() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    /**
     * Forward pass: Y = XW + b
     * @param {Matrix|Array} X - Input tensor (B, in_features) or (B, T, in_features)
     * @returns {Matrix} - Output tensor (B, out_features) or (B, T, out_features)
     */
    forward(X) {
        // Convert to Matrix if needed
        if (Array.isArray(X)) {
            X = new Matrix(X);
        }

        this.cacheInput = X;

        if (X.rows === 1 && X.columns === this.inFeatures) {
            // 2D case: (B, in_features)
            const out = X.mmul(this.W).add(this.b);
            return out;
        } else {
            // Handle 3D case: (B, T, in_features) - flatten to (B*T, in_features)
            const originalShape = [X.rows, X.columns];
            const XReshaped = X.reshape(X.rows * X.columns / this.inFeatures, this.inFeatures);
            const out = XReshaped.mmul(this.W).add(this.b);

            // Reshape back to (B, T, out_features)
            const newRows = originalShape[0];
            const newCols = this.outFeatures;
            return out.reshape(newRows, newCols);
        }
    }

    /**
     * Backward pass: compute gradients
     * @param {Matrix} dZ - Gradient from next layer
     * @returns {Matrix} - Gradient w.r.t input
     */
    backward(dZ) {
        const X = this.cacheInput;

        if (Array.isArray(dZ)) {
            dZ = new Matrix(dZ);
        }

        if (X.rows === 1 && X.columns === this.inFeatures) {
            // 2D case
            // Y = XW + b, so ∂Y/∂W = X^T, ∂Y/∂b = I, ∂Y/∂X = W^T
            this.dW = X.transpose().mmul(dZ);  // ∂L/∂W = X^T @ ∂L/∂Y
            this.db = dZ.sum('column');        // ∂L/∂b = Σ ∂L/∂Y

            return dZ.mmul(this.W.transpose()); // ∂L/∂X = ∂L/∂Y @ W^T
        } else {
            // 3D case - flatten and process
            const originalShape = [X.rows, X.columns];
            const XReshaped = X.reshape(X.rows * X.columns / this.inFeatures, this.inFeatures);
            const dZReshaped = dZ.reshape(dZ.rows * dZ.columns / this.outFeatures, this.outFeatures);

            // Y = XW + b, so ∂Y/∂W = X^T, ∂Y/∂b = I, ∂Y/∂X = W^T
            this.dW = XReshaped.transpose().mmul(dZReshaped);  // ∂L/∂W = X^T @ ∂L/∂Y
            this.db = dZReshaped.sum('column');               // ∂L/∂b = Σ ∂L/∂Y

            const dXReshaped = dZReshaped.mmul(this.W.transpose()); // ∂L/∂X = ∂L/∂Y @ W^T
            return dXReshaped.reshape(originalShape[0], originalShape[1]);
        }
    }

    /**
     * Get parameters
     * @returns {Object} - Dictionary of parameters
     */
    params() {
        return {
            'W': this.W,
            'b': this.b,
        };
    }

    /**
     * Get gradients
     * @returns {Object} - Dictionary of gradients
     */
    grads() {
        return {
            'W': this.dW,
            'b': this.db,
        };
    }
}
