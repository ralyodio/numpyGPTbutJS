import { Matrix } from 'ml-matrix';
import { Module } from './module.js';

/**
 * Embedding layer - converts token indices to dense vectors
 * JavaScript equivalent of numpyGPT/nn/modules/embedding.py
 */
export class Embedding extends Module {
    constructor(vocabSize, embedDim) {
        super();
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;

        // Initialize embedding weights with small random values
        this.W = Matrix.random(vocabSize, embedDim, { random: () => this._randn() * 0.02 });
        this.dW = null;
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
     * Forward pass: lookup embeddings for input indices
     * @param {Array|Matrix} X - Input token indices (B, T)
     * @returns {Matrix} - Embedded vectors (B, T, embed_dim)
     */
    forward(X) {
        // Convert to regular array if Matrix
        let indices;
        if (X instanceof Matrix) {
            indices = X.to2DArray();
            this.cacheInput = indices;
        } else {
            indices = X;
            this.cacheInput = X;
        }

        const B = indices.length;
        const T = indices[0].length;

        // Create output matrix (B, T, embed_dim)
        const output = [];

        for (let b = 0; b < B; b++) {
            const batchOutput = [];
            for (let t = 0; t < T; t++) {
                const tokenIdx = indices[b][t];
                // Get embedding vector for this token
                const embedding = this.W.getRow(tokenIdx);
                batchOutput.push(embedding);
            }
            output.push(batchOutput);
        }

        // Convert to proper 3D structure
        const result = new Matrix(B * T, this.embedDim);
        let idx = 0;
        for (let b = 0; b < B; b++) {
            for (let t = 0; t < T; t++) {
                const tokenIdx = indices[b][t];
                const embedding = this.W.getRow(tokenIdx);
                result.setRow(idx, embedding);
                idx++;
            }
        }

        // Store shape info for backward pass
        this.outputShape = [B, T, this.embedDim];
        return result;
    }

    /**
     * Backward pass: accumulate gradients for embedding weights
     * @param {Matrix} dZ - Gradient from next layer (B*T, embed_dim)
     */
    backward(dZ) {
        // Initialize gradient matrix
        this.dW = Matrix.zeros(this.vocabSize, this.embedDim);

        const indices = this.cacheInput;
        const B = indices.length;
        const T = indices[0].length;

        // Convert dZ to 2D array for easier manipulation
        const dZArray = dZ.to2DArray();

        // Accumulate gradients for each token
        let idx = 0;
        for (let b = 0; b < B; b++) {
            for (let t = 0; t < T; t++) {
                const tokenIdx = indices[b][t];
                const grad = dZArray[idx];

                // Add gradient to the corresponding embedding row
                // We add the gradients to the corresponding rows of the weight matrix W.
                // Since Z[i, j] = W[X[i, j]], the gradient ∂L/∂W[k] is the sum of ∂L/∂Z[i, j]
                // over all (i, j) where X[i, j] == k.
                const currentRow = this.dW.getRow(tokenIdx);
                for (let d = 0; d < this.embedDim; d++) {
                    currentRow[d] += grad[d];
                }
                this.dW.setRow(tokenIdx, currentRow);

                idx++;
            }
        }

        // Embedding layer doesn't propagate gradients to input (indices are discrete)
        return null;
    }

    /**
     * Get parameters
     * @returns {Object} - Dictionary of parameters
     */
    params() {
        return {
            'W': this.W,
        };
    }

    /**
     * Get gradients
     * @returns {Object} - Dictionary of gradients
     */
    grads() {
        return {
            'W': this.dW,
        };
    }
}
