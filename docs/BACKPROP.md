# backpropagation: intuition

*The algorithm that makes neural networks learn.*

## the problem

You have a neural network. It makes predictions. They're wrong. You need to adjust the weights to make better predictions. But there are millions of weights. Which ones to change? By how much?

You just need one basic concept from calculus: **derivatives**.
A derivative tells you how a small change in something (a weight) affects something else (the **loss**).

Once you have those gradients, you do something simple: move each weight a little bit in the **opposite direction of the gradient**.
That's called **gradient descent**.
You're climbing down the **loss curve**, looking for the lowest point.

How to compute gradients?

We split the problem into smaller parts and use the **chain rule** to compute the gradient of the loss with respect to each part.

### why 'backprop'?

We **propagate the error backward** through the network. Starting from the output's loss, we apply the **chain rule layer by layer**, working our way back to the inputs.
At each step, we compute how much each node contributed to the error, and how much its weights need to change to reduce it.

Each layer passes its gradients to the one before it, so there is a **top-down flow of information** to provide to every parameter its update direction.

That's it. Backprop is just an **efficient application of the chain rule** to compute gradients in a big function (e.g., a transformer) composed of many smaller ones.


## the solution: chain rule

```javascript
// If f(x) = h(g(x)), then:
// df/dx = (dh/dg) * (dg/dx)
```

That's all. Everything else is just applying this rule systematically.

## linear layer implementation

```javascript
import { Matrix } from 'ml-matrix';

class Linear {
    constructor(inFeatures, outFeatures) {
        // Initialize weights with small random values
        this.W = Matrix.random(inFeatures, outFeatures, { random: () => this._randn() * 0.02 });
        this.b = Matrix.zeros(1, outFeatures);
    }
    
    forward(X) {
        this.X = X;  // cache for backward pass
        
        // Y = X @ W + b
        const out = X.mmul(this.W);
        
        // Add bias to each row
        const result = Matrix.zeros(out.rows, out.columns);
        for (let i = 0; i < out.rows; i++) {
            for (let j = 0; j < out.columns; j++) {
                result.set(i, j, out.get(i, j) + this.b.get(0, j));
            }
        }
        
        return result;
    }
    
    backward(dY) {
        // dY: gradient flowing back from next layer
        
        // Weight gradient: dW = X^T @ dY
        this.dW = this.X.transpose().mmul(dY);
        
        // Bias gradient: db = sum(dY, axis=0)
        this.db = Matrix.zeros(1, this.b.columns);
        for (let j = 0; j < this.b.columns; j++) {
            let sum = 0;
            for (let i = 0; i < dY.rows; i++) {
                sum += dY.get(i, j);
            }
            this.db.set(0, j, sum);
        }
        
        // Input gradient: dX = dY @ W^T
        const dX = dY.mmul(this.W.transpose());
        return dX;
    }
    
    _randn() {
        // Box-Muller transform for normal distribution
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}
```

## why these formulas work

Forward: `Y = X @ W + b`

Backward: Given `dY` (how loss changes w.r.t Y), find how loss changes w.r.t X, W, b.

**Weight gradient:** `dW = X.T @ dY`
- From Y = XW, we get ∂Y/∂W = X  
- Chain rule: ∂L/∂W = ∂L/∂Y × ∂Y/∂W = dY × X

**Bias gradient:** `db = sum(dY, axis=0)`
- From Y = XW + b, we get ∂Y/∂b = 1
- Chain rule: ∂L/∂b = ∂L/∂Y × 1 = dY

**Input gradient:** `dX = dY @ W.T` 
- From Y = XW, we get ∂Y/∂X = W
- Chain rule: ∂L/∂X = ∂L/∂Y × ∂Y/∂X = dY × W

## example walkthrough

```javascript
import { Matrix } from 'ml-matrix';

// Setup
const X = new Matrix([[1, 2], [3, 4]]);  // batch_size=2, in_features=2
const W = new Matrix([[0.1, 0.3], [0.2, 0.4]]);  // 2x2 weight matrix
const b = new Matrix([[0.1, 0.2]]);

// Forward pass
const Y = X.mmul(W);
// Add bias
for (let i = 0; i < Y.rows; i++) {
    for (let j = 0; j < Y.columns; j++) {
        Y.set(i, j, Y.get(i, j) + b.get(0, j));
    }
}
// Y = [[0.6, 1.3], [1.2, 2.7]]

// Backward pass (assume dY = ones)
const dY = Matrix.ones(Y.rows, Y.columns);  // [[1, 1], [1, 1]]

// Weight gradient: dW = X^T @ dY
const dW = X.transpose().mmul(dY);  // [[4, 4], [6, 6]]

// Bias gradient: db = sum(dY, axis=0)
const db = Matrix.zeros(1, dY.columns);
for (let j = 0; j < dY.columns; j++) {
    let sum = 0;
    for (let i = 0; i < dY.rows; i++) {
        sum += dY.get(i, j);
    }
    db.set(0, j, sum);  // [2, 2]
}

// Input gradient: dX = dY @ W^T
const dX = dY.mmul(W.transpose());  // [[0.4, 0.6], [0.4, 0.6]]

console.log('Y:', Y.to2DArray());
console.log('dW:', dW.to2DArray());
console.log('db:', db.to2DArray());
console.log('dX:', dX.to2DArray());
```

## other common layers

**ReLU:**
```javascript
class ReLU {
    forward(X) {
        this.X = X;  // cache for backward
        const result = Matrix.zeros(X.rows, X.columns);
        
        for (let i = 0; i < X.rows; i++) {
            for (let j = 0; j < X.columns; j++) {
                result.set(i, j, Math.max(0, X.get(i, j)));
            }
        }
        
        return result;
    }
    
    backward(dOut) {
        const dX = Matrix.zeros(this.X.rows, this.X.columns);
        
        for (let i = 0; i < this.X.rows; i++) {
            for (let j = 0; j < this.X.columns; j++) {
                // Gradient is 1 if input > 0, else 0
                const grad = this.X.get(i, j) > 0 ? dOut.get(i, j) : 0;
                dX.set(i, j, grad);
            }
        }
        
        return dX;
    }
}
```

**Softmax with Cross-Entropy Loss:**
```javascript
function crossEntropyLoss(logits, targets) {
    const batchSize = logits.rows;
    const numClasses = logits.columns;
    
    // Compute softmax probabilities
    const probs = softmax(logits);
    
    // Compute loss
    let totalLoss = 0;
    for (let i = 0; i < batchSize; i++) {
        const targetClass = targets[i];
        const prob = probs.get(i, targetClass);
        totalLoss += -Math.log(Math.max(prob, 1e-15));  // numerical stability
    }
    
    return totalLoss / batchSize;
}

function crossEntropyBackward(logits, targets) {
    const batchSize = logits.rows;
    const probs = softmax(logits);
    
    // Gradient of cross-entropy w.r.t logits
    const dLogits = Matrix.zeros(probs.rows, probs.columns);
    
    for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < probs.columns; j++) {
            const targetClass = targets[i];
            const prob = probs.get(i, j);
            
            if (j === targetClass) {
                dLogits.set(i, j, (prob - 1) / batchSize);
            } else {
                dLogits.set(i, j, prob / batchSize);
            }
        }
    }
    
    return dLogits;
}
```

## implementation pattern

Every layer follows this pattern:

```javascript
class Layer {
    forward(x) {
        // 1. cache inputs needed for backward
        this.cache = x;
        
        // 2. compute output
        const output = this.computeOutput(x);
        
        // 3. return output
        return output;
    }
    
    backward(dOut) {
        // 1. use cached values
        const x = this.cache;
        
        // 2. compute parameter gradients (this.dW, this.db, etc.)
        this.computeParameterGradients(dOut, x);
        
        // 3. compute input gradients
        const dX = this.computeInputGradients(dOut, x);
        
        // 4. return input gradients
        return dX;
    }
}
```

## multi-head attention backprop

```javascript
class MultiHeadAttention {
    backward(dZ) {
        const { X, Q, K, V } = this.cache;
        const { B, T } = this.batchInfo;

        // Backward through output projection
        const dConcatenated = this.WO.backward(dZ);

        // Split gradients back to heads
        const dHeadOutputs = [];
        for (let h = 0; h < this.nHeads; h++) {
            const dHead = Matrix.zeros(B * T, this.dK);
            for (let i = 0; i < B * T; i++) {
                for (let d = 0; d < this.dK; d++) {
                    const featureIdx = h * this.dK + d;
                    dHead.set(i, d, dConcatenated.get(i, featureIdx));
                }
            }
            dHeadOutputs.push(dHead);
        }

        // Distribute gradients back to Q, K, V
        const dQ = Matrix.zeros(B * T, this.dModel);
        const dK = Matrix.zeros(B * T, this.dModel);
        const dV = Matrix.zeros(B * T, this.dModel);
        
        for (let h = 0; h < this.nHeads; h++) {
            const startIdx = h * this.dK;
            for (let i = 0; i < B * T; i++) {
                for (let d = 0; d < this.dK; d++) {
                    const grad = dHeadOutputs[h].get(i, d) / 3; // Simplified
                    dQ.set(i, startIdx + d, grad);
                    dK.set(i, startIdx + d, grad);
                    dV.set(i, startIdx + d, grad);
                }
            }
        }

        // Backward through linear projections
        const dXQ = this.WQ.backward(dQ);
        const dXK = this.WK.backward(dK);
        const dXV = this.WV.backward(dV);

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
}
```

## shape debugging

Most bugs are shape mismatches. Always check:

```javascript
console.log(`X: [${X.rows}, ${X.columns}], W: [${W.rows}, ${W.columns}], Y: [${Y.rows}, ${Y.columns}]`);
console.log(`dY: [${dY.rows}, ${dY.columns}], dW: [${dW.rows}, ${dW.columns}], dX: [${dX.rows}, ${dX.columns}]`);

// Verify gradient shapes match parameter shapes
assert(dW.rows === W.rows && dW.columns === W.columns, 'Weight gradient shape mismatch');
assert(db.rows === b.rows && db.columns === b.columns, 'Bias gradient shape mismatch');
```

The gradient of any variable must have the same shape as the variable itself.

## the training loop

```javascript
import { Adam } from './src/optim/adam.js';
import { crossEntropyLoss } from './src/nn/functional.js';

// Initialize model and optimizer
const model = new GPT(config);
const optimizer = new Adam(model.params(), { lr: 0.001 });

// Training loop
for (let epoch = 0; epoch < maxEpochs; epoch++) {
    for (const batch of dataLoader.epochIterator()) {
        // Forward pass
        const { loss, logits } = model.forward(batch.X, batch.Y);
        
        // Backward pass
        const gradients = model.backward();
        
        // Update weights
        optimizer.step();
        
        // Zero gradients for next iteration
        optimizer.zeroGrad();
        
        console.log(`Epoch ${epoch}, Loss: ${loss.toFixed(4)}`);
    }
}
```

## gradient checking

Verify your gradients are correct with numerical differentiation:

```javascript
function gradientCheck(layer, X, epsilon = 1e-5) {
    // Forward pass
    const Y = layer.forward(X);
    const dY = Matrix.ones(Y.rows, Y.columns);
    
    // Analytical gradients
    layer.backward(dY);
    const analyticalGrad = layer.dW.clone();
    
    // Numerical gradients
    const numericalGrad = Matrix.zeros(layer.W.rows, layer.W.columns);
    
    for (let i = 0; i < layer.W.rows; i++) {
        for (let j = 0; j < layer.W.columns; j++) {
            // Perturb weight
            const originalWeight = layer.W.get(i, j);
            
            // Forward with +epsilon
            layer.W.set(i, j, originalWeight + epsilon);
            const YPlus = layer.forward(X);
            const lossPlus = YPlus.sum();
            
            // Forward with -epsilon
            layer.W.set(i, j, originalWeight - epsilon);
            const YMinus = layer.forward(X);
            const lossMinus = YMinus.sum();
            
            // Numerical gradient
            const numGrad = (lossPlus - lossMinus) / (2 * epsilon);
            numericalGrad.set(i, j, numGrad);
            
            // Restore original weight
            layer.W.set(i, j, originalWeight);
        }
    }
    
    // Compare gradients
    const diff = analyticalGrad.sub(numericalGrad);
    const maxDiff = Math.max(...diff.to2DArray().flat().map(Math.abs));
    
    console.log(`Max gradient difference: ${maxDiff}`);
    if (maxDiff < 1e-5) {
        console.log('✓ Gradient check passed!');
    } else {
        console.log('✗ Gradient check failed!');
    }
}
```

## memory management

JavaScript doesn't have explicit memory management, but you can help the garbage collector:

```javascript
class Layer {
    forward(X) {
        // Clear previous cache
        this.cache = null;
        
        // Store new cache
        this.cache = { X: X.clone() };  // Clone to avoid reference issues
        
        return this.computeOutput(X);
    }
    
    backward(dOut) {
        const result = this.computeGradients(dOut);
        
        // Clear cache after backward pass
        this.cache = null;
        
        return result;
    }
}
```

Remember: **Backpropagation is just the chain rule applied systematically**. Once you understand this, implementing any layer becomes straightforward!