import { Matrix } from 'ml-matrix';
import { Module } from './module.js';

/**
 * Layer Normalization
 * JavaScript equivalent of numpyGPT/nn/modules/layerNorm.py
 */
export class LayerNorm extends Module {
    constructor(dModel, eps = 1e-5) {
        super();
        this.dModel = dModel;
        this.eps = eps;

        // Learnable parameters
        this.gamma = Matrix.ones(1, dModel);  // Scale parameter
        this.beta = Matrix.zeros(1, dModel);  // Shift parameter

        this.dgamma = null;
        this.dbeta = null;
        this.cache = {};
    }

    /**
     * Forward pass: normalize along the last dimension
     * @param {Matrix} X - Input tensor (B, T, d_model) flattened to (B*T, d_model)
     * @returns {Matrix} - Normalized output
     */
    forward(X) {
        // X is expected to be (B*T, d_model)
        const batchSize = X.rows;
        const features = X.columns;

        // Compute mean and variance along the feature dimension (axis=-1)
        const mean = Matrix.zeros(batchSize, 1);
        const variance = Matrix.zeros(batchSize, 1);

        // Calculate mean for each sample
        for (let i = 0; i < batchSize; i++) {
            let sum = 0;
            for (let j = 0; j < features; j++) {
                sum += X.get(i, j);
            }
            mean.set(i, 0, sum / features);
        }

        // Calculate variance for each sample
        for (let i = 0; i < batchSize; i++) {
            let sumSquaredDiff = 0;
            const meanVal = mean.get(i, 0);
            for (let j = 0; j < features; j++) {
                const diff = X.get(i, j) - meanVal;
                sumSquaredDiff += diff * diff;
            }
            variance.set(i, 0, sumSquaredDiff / features);
        }

        // Normalize: X_norm = (X - mean) / sqrt(var + eps)
        const XNorm = Matrix.zeros(batchSize, features);
        for (let i = 0; i < batchSize; i++) {
            const meanVal = mean.get(i, 0);
            const varVal = variance.get(i, 0);
            const stdDev = Math.sqrt(varVal + this.eps);

            for (let j = 0; j < features; j++) {
                const normalized = (X.get(i, j) - meanVal) / stdDev;
                XNorm.set(i, j, normalized);
            }
        }

        // Apply learnable parameters: out = gamma * X_norm + beta
        const out = Matrix.zeros(batchSize, features);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < features; j++) {
                const gammaVal = this.gamma.get(0, j);
                const betaVal = this.beta.get(0, j);
                const normalizedVal = XNorm.get(i, j);
                out.set(i, j, gammaVal * normalizedVal + betaVal);
            }
        }

        // Cache values for backward pass
        this.cache = {
            X,
            mean,
            variance,
            XNorm,
        };

        return out;
    }

    /**
     * Backward pass: compute gradients
     * @param {Matrix} dZ - Gradient from next layer
     * @returns {Matrix} - Gradient w.r.t input
     */
    backward(dZ) {
        const X = this.cache.X;
        const mean = this.cache.mean;
        const variance = this.cache.variance;
        const XNorm = this.cache.XNorm;

        const batchSize = X.rows;
        const N = X.columns; // Number of features

        // Initialize gradients
        this.dgamma = Matrix.zeros(1, this.dModel);
        this.dbeta = Matrix.zeros(1, this.dModel);

        // Compute gradients for gamma and beta
        // dgamma = sum over batch dimension of (dZ * X_norm)
        // dbeta = sum over batch dimension of dZ
        for (let j = 0; j < N; j++) {
            let dgammaSum = 0;
            let dbetaSum = 0;

            for (let i = 0; i < batchSize; i++) {
                dgammaSum += dZ.get(i, j) * XNorm.get(i, j);
                dbetaSum += dZ.get(i, j);
            }

            this.dgamma.set(0, j, dgammaSum);
            this.dbeta.set(0, j, dbetaSum);
        }

        // Compute gradient w.r.t normalized input
        const dXNorm = Matrix.zeros(batchSize, N);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < N; j++) {
                dXNorm.set(i, j, dZ.get(i, j) * this.gamma.get(0, j));
            }
        }

        // Compute gradient w.r.t variance
        const dVar = Matrix.zeros(batchSize, 1);
        for (let i = 0; i < batchSize; i++) {
            let sum = 0;
            const meanVal = mean.get(i, 0);
            const varVal = variance.get(i, 0);

            for (let j = 0; j < N; j++) {
                sum += dXNorm.get(i, j) * (X.get(i, j) - meanVal);
            }

            dVar.set(i, 0, sum * (-0.5) * Math.pow(varVal + this.eps, -1.5));
        }

        // Compute gradient w.r.t mean
        const dMean = Matrix.zeros(batchSize, 1);
        for (let i = 0; i < batchSize; i++) {
            const varVal = variance.get(i, 0);
            const meanVal = mean.get(i, 0);

            // Direct effect of mean on normalized values
            let directSum = 0;
            for (let j = 0; j < N; j++) {
                directSum += dXNorm.get(i, j);
            }
            const directEffect = directSum * (-1) / Math.sqrt(varVal + this.eps);

            // Indirect effect through variance
            let indirectSum = 0;
            for (let j = 0; j < N; j++) {
                indirectSum += -2 * (X.get(i, j) - meanVal);
            }
            const indirectEffect = dVar.get(i, 0) * indirectSum / N;

            dMean.set(i, 0, directEffect + indirectEffect);
        }

        // Compute gradient w.r.t input
        const dX = Matrix.zeros(batchSize, N);
        for (let i = 0; i < batchSize; i++) {
            const varVal = variance.get(i, 0);
            const meanVal = mean.get(i, 0);
            const dMeanVal = dMean.get(i, 0);
            const dVarVal = dVar.get(i, 0);

            for (let j = 0; j < N; j++) {
                // Three components of the gradient
                const comp1 = dXNorm.get(i, j) / Math.sqrt(varVal + this.eps);
                const comp2 = dMeanVal / N;
                const comp3 = dVarVal * 2 * (X.get(i, j) - meanVal) / N;

                dX.set(i, j, comp1 + comp2 + comp3);
            }
        }

        return dX;
    }

    /**
     * Get parameters
     * @returns {Object} - Dictionary of parameters
     */
    params() {
        return {
            'gamma': this.gamma,
            'beta': this.beta,
        };
    }

    /**
     * Get gradients
     * @returns {Object} - Dictionary of gradients
     */
    grads() {
        return {
            'gamma': this.dgamma,
            'beta': this.dbeta,
        };
    }
}
