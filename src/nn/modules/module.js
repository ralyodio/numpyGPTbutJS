/**
 * Base Module class for all neural network components
 * JavaScript equivalent of numpyGPT/nn/modules/module.py
 */
export class Module {
    constructor() {
        this.training = true;
    }

    /**
     * Forward pass - must be implemented by subclasses
     * @param {*} x - Input data
     * @returns {*} - Output data
     */
    forward(x) {
        throw new Error('forward method must be implemented by subclass');
    }

    /**
     * Backward pass - must be implemented by subclasses
     * @param {*} grad - Gradient from next layer
     * @returns {*} - Gradient w.r.t input
     */
    backward(grad) {
        throw new Error('backward method must be implemented by subclass');
    }

    /**
     * Set module to training mode
     */
    train() {
        this.training = true;
    }

    /**
     * Set module to evaluation mode
     */
    eval() {
        this.training = false;
    }

    /**
     * Get all parameters of this module
     * @returns {Object} - Dictionary of parameter names to values
     */
    params() {
        return {};
    }

    /**
     * Get all gradients of this module
     * @returns {Object} - Dictionary of parameter names to gradients
     */
    grads() {
        return {};
    }

    /**
     * Make module callable like a function
     * @param {...*} args - Arguments to forward pass
     * @returns {*} - Output of forward pass
     */
    __call__(...args) {
        return this.forward(...args);
    }
}