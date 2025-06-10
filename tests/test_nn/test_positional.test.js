import { Matrix } from 'ml-matrix';
import { PositionalEncoding } from '../../src/nn/modules/positional.js';

describe('PositionalEncoding', () => {
  let posEnc;
  const maxLen = 10;
  const dModel = 4;

  beforeEach(() => {
    posEnc = new PositionalEncoding(maxLen, dModel);
  });

  describe('constructor', () => {
    test('should initialize with correct dimensions', () => {
      expect(posEnc.maxLen).toBe(maxLen);
      expect(posEnc.dModel).toBe(dModel);
      expect(posEnc.W.rows).toBe(maxLen);
      expect(posEnc.W.columns).toBe(dModel);
    });

    test('should initialize weights with small values', () => {
      const weights = posEnc.W.to2DArray();
      for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
          expect(Math.abs(weights[i][j])).toBeLessThan(0.1); // Small random values
        }
      }
    });
  });

  describe('forward pass', () => {
    test('should add positional embeddings to input', () => {
      // Create test input: 2 batches, 3 sequence length, 4 features
      const B = 2;
      const T = 3;
      const input = Matrix.ones(B * T, dModel); // All ones for easy testing

      const output = posEnc.forward(input);

      expect(output.rows).toBe(B * T);
      expect(output.columns).toBe(dModel);

      // Output should be input + positional embeddings
      // Since input is all ones, output[i,j] = 1 + W[pos,j]
      const outputArray = output.to2DArray();
      const weightsArray = posEnc.W.to2DArray();

      for (let b = 0; b < B; b++) {
        for (let t = 0; t < T; t++) {
          const idx = b * T + t;
          for (let d = 0; d < dModel; d++) {
            const expected = 1 + weightsArray[t][d]; // 1 (input) + positional embedding
            expect(outputArray[idx][d]).toBeCloseTo(expected, 6);
          }
        }
      }
    });

    test('should handle different sequence lengths', () => {
      const input1 = Matrix.ones(2, dModel); // Sequence length 2
      const input2 = Matrix.ones(5, dModel); // Sequence length 5

      const output1 = posEnc.forward(input1);
      const output2 = posEnc.forward(input2);

      expect(output1.rows).toBe(2);
      expect(output2.rows).toBe(5);
      expect(output1.columns).toBe(dModel);
      expect(output2.columns).toBe(dModel);
    });

    test('should cache input for backward pass', () => {
      const input = Matrix.ones(4, dModel);
      posEnc.forward(input);

      expect(posEnc.cacheInput).toBeDefined();
      expect(posEnc.batchInfo).toBeDefined();
      expect(posEnc.batchInfo.totalTokens).toBe(4);
    });
  });

  describe('backward pass', () => {
    test('should compute gradients correctly', () => {
      const B = 2;
      const T = 3;
      const input = Matrix.ones(B * T, dModel);

      // Forward pass
      posEnc.forward(input);

      // Create gradient from next layer (all ones for simplicity)
      const dZ = Matrix.ones(B * T, dModel);

      // Backward pass
      const dX = posEnc.backward(dZ);

      // Check that input gradient is unchanged (addition operation)
      expect(dX.rows).toBe(B * T);
      expect(dX.columns).toBe(dModel);
      const dXArray = dX.to2DArray();
      for (let i = 0; i < B * T; i++) {
        for (let j = 0; j < dModel; j++) {
          expect(dXArray[i][j]).toBe(1); // Should be unchanged
        }
      }

      // Check positional embedding gradients
      expect(posEnc.dW).toBeDefined();
      expect(posEnc.dW.rows).toBe(maxLen);
      expect(posEnc.dW.columns).toBe(dModel);

      const dWArray = posEnc.dW.to2DArray();
      // For positions 0, 1, 2 (T=3), gradient should be sum over batch dimension
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < dModel; d++) {
          expect(dWArray[t][d]).toBe(B); // Sum of B ones = B
        }
      }

      // For unused positions (t >= T), gradient should be 0
      for (let t = T; t < maxLen; t++) {
        for (let d = 0; d < dModel; d++) {
          expect(dWArray[t][d]).toBe(0);
        }
      }
    });

    test('should handle varying gradient magnitudes', () => {
      const B = 1;
      const T = 2;
      const input = Matrix.zeros(B * T, dModel);

      posEnc.forward(input);

      // Create gradient with different values
      const dZ = new Matrix([
        [1, 2, 3, 4], // Position 0
        [5, 6, 7, 8], // Position 1
      ]);

      posEnc.backward(dZ);

      const dWArray = posEnc.dW.to2DArray();

      // Check gradients for each position
      expect(dWArray[0]).toEqual([1, 2, 3, 4]); // Position 0
      expect(dWArray[1]).toEqual([5, 6, 7, 8]); // Position 1

      // Unused positions should be zero
      for (let t = 2; t < maxLen; t++) {
        expect(dWArray[t]).toEqual([0, 0, 0, 0]);
      }
    });
  });

  describe('parameter management', () => {
    test('should return correct parameters', () => {
      const params = posEnc.params();
      expect(params).toHaveProperty('W');
      expect(params.W).toBe(posEnc.W);
    });

    test('should return correct gradients', () => {
      // Need to run forward and backward first
      const input = Matrix.ones(2, dModel);
      posEnc.forward(input);
      posEnc.backward(Matrix.ones(2, dModel));

      const grads = posEnc.grads();
      expect(grads).toHaveProperty('W');
      expect(grads.W).toBe(posEnc.dW);
    });

    test('should return null gradients before backward pass', () => {
      const grads = posEnc.grads();
      expect(grads.W).toBeNull();
    });
  });

  describe('edge cases', () => {
    test('should handle maximum sequence length', () => {
      const input = Matrix.ones(maxLen, dModel);
      const output = posEnc.forward(input);

      expect(output.rows).toBe(maxLen);
      expect(output.columns).toBe(dModel);
    });

    test('should handle single token input', () => {
      const input = Matrix.ones(1, dModel);
      const output = posEnc.forward(input);

      expect(output.rows).toBe(1);
      expect(output.columns).toBe(dModel);

      // Should add position 0 embedding
      const outputArray = output.to2DArray();
      const weightsArray = posEnc.W.to2DArray();

      for (let d = 0; d < dModel; d++) {
        const expected = 1 + weightsArray[0][d];
        expect(outputArray[0][d]).toBeCloseTo(expected, 6);
      }
    });
  });
});