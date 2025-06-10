import { Matrix } from 'ml-matrix';

/**
 * Neural Network Functional Operations
 * JavaScript equivalent of numpyGPT/nn/functional.py
 */

/**
 * Cross-entropy loss function
 * @param {Matrix} logits - Raw model outputs (N, num_classes)
 * @param {Array} targets - Target class indices (N,)
 * @param {number} eps - Small epsilon for numerical stability
 * @returns {number} - Average cross-entropy loss
 */
export function crossEntropyLoss(logits, targets, eps = 1e-7) {
  const N = targets.length;
  const numClasses = logits.columns;

  // Convert logits to 2D array for easier manipulation
  const logitsArray = logits.to2DArray();

  // Compute log-softmax via log-sum-exp trick for numerical stability
  const logSoftmax = [];

  for (let i = 0; i < N; i++) {
    const row = logitsArray[i];
    
    // Find max for numerical stability
    const maxLogit = Math.max(...row);
    
    // Shift logits
    const shifted = row.map(x => x - maxLogit);
    
    // Compute log-sum-exp
    const sumExp = shifted.reduce((sum, x) => sum + Math.exp(x), 0);
    const logSumExp = Math.log(sumExp + eps);
    
    // Compute log-softmax
    const logSoftmaxRow = shifted.map(x => x - logSumExp);
    logSoftmax.push(logSoftmaxRow);
  }

  // Compute negative log-likelihood: -log P(correct_class)
  let totalNLL = 0;
  for (let i = 0; i < N; i++) {
    const targetClass = targets[i];
    const nll = -logSoftmax[i][targetClass];
    totalNLL += nll;
  }

  // Return average loss
  return totalNLL / N;
}

/**
 * Softmax function (standalone)
 * @param {Matrix} X - Input matrix (N, num_classes)
 * @returns {Matrix} - Softmax probabilities (N, num_classes)
 */
export function softmax(X) {
  const rows = X.rows;
  const cols = X.columns;
  const result = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    const row = X.getRow(i);
    
    // Find max for numerical stability
    const maxVal = Math.max(...row);
    
    // Compute exponentials with shifted values
    const expRow = row.map(x => Math.exp(x - maxVal));
    
    // Compute sum of exponentials
    const sumExp = expRow.reduce((sum, val) => sum + val, 0);
    
    // Normalize to get probabilities
    const softmaxRow = expRow.map(x => x / sumExp);
    
    // Set the row in result matrix
    result.setRow(i, softmaxRow);
  }

  return result;
}

/**
 * Log-softmax function (standalone)
 * @param {Matrix} X - Input matrix (N, num_classes)
 * @returns {Matrix} - Log-softmax values (N, num_classes)
 */
export function logSoftmax(X) {
  const rows = X.rows;
  const cols = X.columns;
  const result = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    const row = X.getRow(i);
    
    // Find max for numerical stability
    const maxVal = Math.max(...row);
    
    // Shift values
    const shifted = row.map(x => x - maxVal);
    
    // Compute log-sum-exp
    const sumExp = shifted.reduce((sum, x) => sum + Math.exp(x), 0);
    const logSumExp = Math.log(sumExp);
    
    // Compute log-softmax
    const logSoftmaxRow = shifted.map(x => x - logSumExp);
    
    // Set the row in result matrix
    result.setRow(i, logSoftmaxRow);
  }

  return result;
}

/**
 * ReLU function (standalone)
 * @param {Matrix} X - Input matrix
 * @returns {Matrix} - ReLU output
 */
export function relu(X) {
  const rows = X.rows;
  const cols = X.columns;
  const result = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result.set(i, j, Math.max(0, X.get(i, j)));
    }
  }

  return result;
}

/**
 * Sigmoid function (standalone)
 * @param {Matrix} X - Input matrix
 * @returns {Matrix} - Sigmoid output
 */
export function sigmoid(X) {
  const rows = X.rows;
  const cols = X.columns;
  const result = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const val = X.get(i, j);
      result.set(i, j, 1 / (1 + Math.exp(-val)));
    }
  }

  return result;
}

/**
 * Tanh function (standalone)
 * @param {Matrix} X - Input matrix
 * @returns {Matrix} - Tanh output
 */
export function tanh(X) {
  const rows = X.rows;
  const cols = X.columns;
  const result = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const val = X.get(i, j);
      result.set(i, j, Math.tanh(val));
    }
  }

  return result;
}