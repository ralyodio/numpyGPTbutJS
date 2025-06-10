import { Matrix } from 'ml-matrix';
import { Module } from './module.js';

/**
 * Softmax activation function
 * JavaScript equivalent of numpyGPT/nn/modules/activation.py Softmax class
 */
export class Softmax extends Module {
    constructor() {
        super();
        this.cacheOutput = null;
    }

    /**
     * Forward pass: apply softmax activation
     * @param {Matrix} X - Input tensor
     * @returns {Matrix} - Softmax output
     */
    forward(X) {
        const rows = X.rows;
        const cols = X.columns;
        const result = Matrix.zeros(rows, cols);

        // Apply softmax to each row
        for (let i = 0; i < rows; i++) {
            const row = X.getRow(i);

            // Softmax trick for numerical stability: subtract max
            const maxVal = Math.max(...row);
            const shiftedRow = row.map(x => x - maxVal);

            // Compute exponentials
            const expRow = shiftedRow.map(x => Math.exp(x));

            // Compute sum of exponentials
            const sumExp = expRow.reduce((sum, val) => sum + val, 0);

            // Normalize
            const softmaxRow = expRow.map(x => x / sumExp);
            result.setRow(i, softmaxRow);
        }

        this.cacheOutput = result;
        return result;
    }

    /**
     * Backward pass: compute gradients
     * @param {Matrix} dZOrYTrue - Either gradient from next layer or true labels
     * @param {Array} YTrue - True labels (optional, for cross-entropy + softmax)
     * @returns {Matrix} - Gradient w.r.t input
     */
    backward(dZOrYTrue, YTrue = null) {
        if (YTrue !== null) {
            // CrossEntropy + Softmax derivative simplifies to: (ŷ - y) / N
            const YHat = this.cacheOutput;
            const batchSize = YHat.rows;
            const numClasses = YHat.columns;

            // Create one-hot encoded true labels
            const Y = Matrix.zeros(batchSize, numClasses);
            for (let i = 0; i < batchSize; i++) {
                Y.set(i, YTrue[i], 1);
            }

            // Compute gradient: (ŷ - y) / N
            const result = Matrix.zeros(batchSize, numClasses);
            for (let i = 0; i < batchSize; i++) {
                for (let j = 0; j < numClasses; j++) {
                    const grad = (YHat.get(i, j) - Y.get(i, j)) / batchSize;
                    result.set(i, j, grad);
                }
            }
            return result;
        } else {
            // Standard softmax backward pass using Jacobian
            const dZ = dZOrYTrue;
            const softmaxOutput = this.cacheOutput;
            const rows = softmaxOutput.rows;
            const cols = softmaxOutput.columns;

            const result = Matrix.zeros(rows, cols);

            for (let i = 0; i < rows; i++) {
                // For each sample, compute: σ_i(∂L/∂σ_i - Σ_j ∂L/∂σ_j·σ_j)
                let sumTerm = 0;
                for (let j = 0; j < cols; j++) {
                    sumTerm += dZ.get(i, j) * softmaxOutput.get(i, j);
                }

                for (let j = 0; j < cols; j++) {
                    const grad = softmaxOutput.get(i, j) * (dZ.get(i, j) - sumTerm);
                    result.set(i, j, grad);
                }
            }

            return result;
        }
    }

    params() {
        return {};
    }

    grads() {
        return {};
    }
}

/**
 * ReLU activation function
 * JavaScript equivalent of numpyGPT/nn/modules/activation.py ReLU class
 */
export class ReLU extends Module {
    constructor() {
        super();
        this.cacheInput = null;
    }

    /**
     * Forward pass: apply ReLU activation
     * @param {Matrix} X - Input tensor
     * @returns {Matrix} - ReLU output
     */
    forward(X) {
        this.cacheInput = X;

        const rows = X.rows;
        const cols = X.columns;
        const result = Matrix.zeros(rows, cols);

        // Apply ReLU: max(0, x)
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.set(i, j, Math.max(0, X.get(i, j)));
            }
        }

        return result;
    }

    /**
     * Backward pass: compute gradients
     * @param {Matrix} dZ - Gradient from next layer
     * @returns {Matrix} - Gradient w.r.t input
     */
    backward(dZ) {
        const X = this.cacheInput;
        const rows = X.rows;
        const cols = X.columns;
        const result = Matrix.zeros(rows, cols);

        // ReLU derivative: 1 if x > 0, else 0
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const grad = X.get(i, j) > 0 ? 1 : 0;
                result.set(i, j, dZ.get(i, j) * grad);
            }
        }

        return result;
    }

    params() {
        return {};
    }

    grads() {
        return {};
    }
}

/**
 * Leaky ReLU activation function
 * JavaScript equivalent of numpyGPT/nn/modules/activation.py LeakyReLU class
 */
export class LeakyReLU extends Module {
    constructor(alpha = 0.01) {
        super();
        this.alpha = alpha;
        this.cacheInput = null;
    }

    /**
     * Forward pass: apply Leaky ReLU activation
     * @param {Matrix} X - Input tensor
     * @returns {Matrix} - Leaky ReLU output
     */
    forward(X) {
        this.cacheInput = X;

        const rows = X.rows;
        const cols = X.columns;
        const result = Matrix.zeros(rows, cols);

        // Apply Leaky ReLU: x if x > 0, else α*x
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = X.get(i, j);
                result.set(i, j, val > 0 ? val : this.alpha * val);
            }
        }

        return result;
    }

    /**
     * Backward pass: compute gradients
     * @param {Matrix} dZ - Gradient from next layer
     * @returns {Matrix} - Gradient w.r.t input
     */
    backward(dZ) {
        const X = this.cacheInput;
        const rows = X.rows;
        const cols = X.columns;
        const result = Matrix.zeros(rows, cols);

        // Leaky ReLU derivative: 1 if x > 0, else α
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const grad = X.get(i, j) > 0 ? 1.0 : this.alpha;
                result.set(i, j, dZ.get(i, j) * grad);
            }
        }

        return result;
    }

    params() {
        return {};
    }

    grads() {
        return {};
    }
}
