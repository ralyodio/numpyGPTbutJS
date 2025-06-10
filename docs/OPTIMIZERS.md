# optimizers: how neural networks learn

*Turning gradients into better weights.*

## the problem

You have gradients. They point in the direction of steepest loss increase. You want to minimize loss. So you move in the opposite direction. But how far? How fast? 

Naive approach: `weights -= learning_rate * gradients`

This works, but slowly. Modern optimizers are smarter.

## SGD: the foundation

```javascript
import { Matrix } from 'ml-matrix';

class SGD {
    constructor(modules, lr = 0.01) {
        this.params = modules;
        this.lr = lr;
    }
    
    step() {
        for (const module of this.params) {
            const params = module.params();
            const grads = module.grads();
            
            for (const paramKey in params) {
                const param = params[paramKey];
                const grad = grads[paramKey];
                
                // θ = θ - lr * ∇θ
                for (let i = 0; i < param.rows; i++) {
                    for (let j = 0; j < param.columns; j++) {
                        const newValue = param.get(i, j) - this.lr * grad.get(i, j);
                        param.set(i, j, newValue);
                    }
                }
            }
        }
    }
}
```

**The update rule:** `θ = θ - lr * ∇θ`

**Problems:**
- Same learning rate for all parameters
- No memory of previous gradients  
- Gets stuck in narrow valleys
- Oscillates around optimal points

## Adam: adaptive moments

Adam tracks two moving averages:
- **Momentum** (first moment): direction of gradients
- **RMSprop** (second moment): magnitude of gradients

```javascript
import { Matrix } from 'ml-matrix';

class Adam {
    constructor(modules, options = {}) {
        this.params = modules;
        this.lr = options.lr || 0.001;
        this.beta1 = options.beta1 || 0.9;
        this.beta2 = options.beta2 || 0.999;
        this.eps = options.eps || 1e-8;
        this.t = 0;  // time step
        
        // Initialize momentum and velocity for each parameter
        this.m = new Map();  // first moment (momentum)
        this.v = new Map();  // second moment (velocity)
        
        this._initializeMoments();
    }
    
    _initializeMoments() {
        for (let i = 0; i < this.params.length; i++) {
            const module = this.params[i];
            const params = module.params();
            
            this.m.set(i, {});
            this.v.set(i, {});
            
            for (const paramKey in params) {
                const param = params[paramKey];
                this.m.get(i)[paramKey] = Matrix.zeros(param.rows, param.columns);
                this.v.get(i)[paramKey] = Matrix.zeros(param.rows, param.columns);
            }
        }
    }
    
    step() {
        this.t += 1;
        
        for (let i = 0; i < this.params.length; i++) {
            const module = this.params[i];
            const params = module.params();
            const grads = module.grads();
            
            for (const paramKey in grads) {
                const param = params[paramKey];
                const grad = grads[paramKey];
                const m = this.m.get(i)[paramKey];
                const v = this.v.get(i)[paramKey];
                
                // Update biased first moment estimate: m = β₁ * m + (1-β₁) * g
                for (let row = 0; row < grad.rows; row++) {
                    for (let col = 0; col < grad.columns; col++) {
                        const g = grad.get(row, col);
                        const mVal = this.beta1 * m.get(row, col) + (1 - this.beta1) * g;
                        m.set(row, col, mVal);
                        
                        // Update biased second moment estimate: v = β₂ * v + (1-β₂) * g²
                        const vVal = this.beta2 * v.get(row, col) + (1 - this.beta2) * g * g;
                        v.set(row, col, vVal);
                        
                        // Bias correction
                        const mHat = mVal / (1 - Math.pow(this.beta1, this.t));
                        const vHat = vVal / (1 - Math.pow(this.beta2, this.t));
                        
                        // Update parameter: θ = θ - lr * m̂ / (√v̂ + ε)
                        const update = this.lr * mHat / (Math.sqrt(vHat) + this.eps);
                        const newValue = param.get(row, col) - update;
                        param.set(row, col, newValue);
                    }
                }
            }
        }
    }
    
    zeroGrad() {
        for (const module of this.params) {
            const grads = module.grads();
            for (const gradKey in grads) {
                const grad = grads[gradKey];
                if (grad) {
                    for (let i = 0; i < grad.rows; i++) {
                        for (let j = 0; j < grad.columns; j++) {
                            grad.set(i, j, 0);
                        }
                    }
                }
            }
        }
    }
    
    getCurrentLr() {
        return this.lr;
    }
}
```

## why Adam works

**Momentum (`m`):** Smooths gradients. Builds velocity in consistent directions.
```javascript
// m = β₁ * m + (1-β₁) * g    // β₁=0.9 typical
const mVal = this.beta1 * m.get(row, col) + (1 - this.beta1) * g;
```

**RMSprop (`v`):** Adapts learning rate per parameter. Large gradients → smaller steps.
```javascript  
// v = β₂ * v + (1-β₂) * g²   // β₂=0.999 typical
const vVal = this.beta2 * v.get(row, col) + (1 - this.beta2) * g * g;
```

**Bias correction:** Early iterations have small `m` and `v`. Correction compensates.
```javascript
// m̂ = m / (1 - β₁ᵗ)
// v̂ = v / (1 - β₂ᵗ)
const mHat = mVal / (1 - Math.pow(this.beta1, this.t));
const vHat = vVal / (1 - Math.pow(this.beta2, this.t));
```

**Final update:** Momentum direction, RMSprop magnitude.
```javascript
// θ = θ - lr * m̂ / (√v̂ + ε)
const update = this.lr * mHat / (Math.sqrt(vHat) + this.eps);
const newValue = param.get(row, col) - update;
```

## the intuition

**SGD:** Walk with fixed step size.

**Adam:** 
- Walk in the average direction you've been going (momentum)
- Take smaller steps where the path is steep (adaptive learning rate)
- Correct for startup effects (bias correction)

## hyperparameters

**Learning rate (`lr`):** How big steps to take
- Too high: overshoots, unstable
- Too low: slow convergence
- Typical: 1e-3 to 3e-4

**Beta1 (`β₁`):** Momentum decay  
- Higher = more momentum
- Typical: 0.9

**Beta2 (`β₂`):** RMSprop decay
- Higher = longer memory of gradient magnitudes  
- Typical: 0.999

**Epsilon (`ε`):** Numerical stability
- Prevents division by zero
- Typical: 1e-8

## learning rate scheduling

Fixed learning rates are suboptimal. Common schedules:

**Warmup + Cosine Decay:**
```javascript
class WarmupCosineLR {
    constructor(optimizer, warmupSteps, maxSteps, maxLr, minLr = 0) {
        this.optimizer = optimizer;
        this.warmupSteps = warmupSteps;
        this.maxSteps = maxSteps;
        this.maxLr = maxLr;
        this.minLr = minLr;
        this.step = 0;
    }
    
    getNextLr() {
        this.step += 1;
        
        if (this.step < this.warmupSteps) {
            // Linear warmup
            return this.maxLr * this.step / this.warmupSteps;
        } else if (this.step > this.maxSteps) {
            // Minimum learning rate
            return this.minLr;
        } else {
            // Cosine decay
            const progress = (this.step - this.warmupSteps) / (this.maxSteps - this.warmupSteps);
            const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * progress));
            return this.minLr + (this.maxLr - this.minLr) * cosineDecay;
        }
    }
    
    stepLr() {
        this.optimizer.lr = this.getNextLr();
    }
}
```

**Step Decay:**
```javascript
class StepLR {
    constructor(optimizer, stepSize, gamma = 0.1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.step = 0;
        this.baseLr = optimizer.lr;
    }
    
    stepLr() {
        this.step += 1;
        const decayFactor = Math.pow(this.gamma, Math.floor(this.step / this.stepSize));
        this.optimizer.lr = this.baseLr * decayFactor;
    }
}
```

## usage examples

**Basic Adam optimizer:**
```javascript
import { Adam } from './src/optim/adam.js';
import { GPT } from './src/models/gpt.js';

const model = new GPT(config);
const optimizer = new Adam([model], { lr: 0.001 });

// Training loop
for (const batch of dataLoader.epochIterator()) {
    // Forward pass
    const { loss, logits } = model.forward(batch.X, batch.Y);
    
    // Backward pass
    model.backward();
    
    // Update weights
    optimizer.step();
    
    // Clear gradients
    optimizer.zeroGrad();
}
```

**With learning rate scheduling:**
```javascript
import { WarmupCosineLR } from './src/optim/lr_scheduler/warmup_cosine_lr.js';

const optimizer = new Adam([model], { lr: 0.001 });
const scheduler = new WarmupCosineLR(optimizer, 1000, 10000, 0.001, 0.0001);

for (let epoch = 0; epoch < maxEpochs; epoch++) {
    for (const batch of dataLoader.epochIterator()) {
        // Training step
        const { loss } = model.forward(batch.X, batch.Y);
        model.backward();
        optimizer.step();
        optimizer.zeroGrad();
        
        // Update learning rate
        scheduler.stepLr();
        
        console.log(`Loss: ${loss.toFixed(4)}, LR: ${optimizer.getCurrentLr().toFixed(6)}`);
    }
}
```

## comparison

| Optimizer | Memory | Convergence | Hyperparams | Use Case |
|-----------|---------|-------------|-------------|----------|
| SGD       | None    | Slow, noisy | 1 (lr)      | Simple problems |
| Adam      | 2x params | Fast, stable | 4 (lr, β₁, β₂, ε) | Most deep learning |

## implementation pattern

All optimizers follow this interface:

```javascript
class Optimizer {
    constructor(modules, options = {}) {
        this.params = modules;
        this.lr = options.lr || 0.01;
    }
    
    step() {
        // Update parameters using gradients
        throw new Error('step() method must be implemented');
    }
    
    zeroGrad() {
        // Reset gradients to zero to avoid contamination of the next batch
        for (const module of this.params) {
            const grads = module.grads();
            for (const gradKey in grads) {
                const grad = grads[gradKey];
                if (grad) {
                    for (let i = 0; i < grad.rows; i++) {
                        for (let j = 0; j < grad.columns; j++) {
                            grad.set(i, j, 0);
                        }
                    }
                }
            }
        }
    }
    
    getCurrentLr() {
        return this.lr;
    }
}
```

## gradient clipping

Prevent exploding gradients with gradient clipping:

```javascript
function clipGradNorm(modules, maxNorm = 1.0) {
    let totalNorm = 0;
    
    // Calculate total gradient norm
    for (const module of modules) {
        const grads = module.grads();
        for (const gradKey in grads) {
            const grad = grads[gradKey];
            if (grad) {
                for (let i = 0; i < grad.rows; i++) {
                    for (let j = 0; j < grad.columns; j++) {
                        const g = grad.get(i, j);
                        totalNorm += g * g;
                    }
                }
            }
        }
    }
    
    totalNorm = Math.sqrt(totalNorm);
    
    // Clip gradients if norm exceeds threshold
    if (totalNorm > maxNorm) {
        const clipCoeff = maxNorm / totalNorm;
        
        for (const module of modules) {
            const grads = module.grads();
            for (const gradKey in grads) {
                const grad = grads[gradKey];
                if (grad) {
                    for (let i = 0; i < grad.rows; i++) {
                        for (let j = 0; j < grad.columns; j++) {
                            const clippedValue = grad.get(i, j) * clipCoeff;
                            grad.set(i, j, clippedValue);
                        }
                    }
                }
            }
        }
    }
    
    return totalNorm;
}
```

## monitoring optimization

Track optimization progress:

```javascript
class OptimizationMonitor {
    constructor() {
        this.history = {
            loss: [],
            gradNorm: [],
            lr: [],
            step: []
        };
    }
    
    log(step, loss, gradNorm, lr) {
        this.history.step.push(step);
        this.history.loss.push(loss);
        this.history.gradNorm.push(gradNorm);
        this.history.lr.push(lr);
    }
    
    getStats() {
        const recent = this.history.loss.slice(-100);
        const avgLoss = recent.reduce((a, b) => a + b, 0) / recent.length;
        const currentGradNorm = this.history.gradNorm[this.history.gradNorm.length - 1];
        
        return {
            avgLoss,
            currentGradNorm,
            totalSteps: this.history.step.length
        };
    }
}

// Usage
const monitor = new OptimizationMonitor();

for (let step = 0; step < maxSteps; step++) {
    const { loss } = model.forward(batch.X, batch.Y);
    model.backward();
    
    const gradNorm = clipGradNorm([model], 1.0);
    optimizer.step();
    optimizer.zeroGrad();
    
    monitor.log(step, loss, gradNorm, optimizer.getCurrentLr());
    
    if (step % 100 === 0) {
        const stats = monitor.getStats();
        console.log(`Step ${step}: Loss=${stats.avgLoss.toFixed(4)}, GradNorm=${stats.currentGradNorm.toFixed(4)}`);
    }
}
```

## debugging optimization

Common optimization issues and solutions:

**Loss not decreasing:**
```javascript
// Check gradient norms
const gradNorm = clipGradNorm([model]);
if (gradNorm < 1e-6) {
    console.log('⚠️  Vanishing gradients detected');
} else if (gradNorm > 10) {
    console.log('⚠️  Exploding gradients detected');
}

// Check learning rate
if (optimizer.getCurrentLr() > 0.01) {
    console.log('⚠️  Learning rate might be too high');
} else if (optimizer.getCurrentLr() < 1e-6) {
    console.log('⚠️  Learning rate might be too low');
}
```

**Loss oscillating:**
```javascript
// Reduce learning rate or increase momentum
const newOptimizer = new Adam([model], { 
    lr: optimizer.lr * 0.5,  // Reduce LR
    beta1: 0.95  // Increase momentum
});
```

The optimizer determines how quickly your model learns. SGD is simple but slow. Adam is adaptive and fast. For modern deep learning, Adam with learning rate scheduling is the default choice.

## key takeaways

1. **Adam is usually the best choice** for deep learning
2. **Learning rate scheduling** significantly improves convergence
3. **Gradient clipping** prevents training instability
4. **Monitor gradients** to debug optimization issues
5. **Bias correction** is crucial in early training steps

Remember: The optimizer is the engine that drives learning. Choose wisely and tune carefully!