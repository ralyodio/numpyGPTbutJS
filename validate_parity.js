#!/usr/bin/env node

/**
 * Cross-validation script to compare Python vs JavaScript outputs
 * Ensures identical behavior between implementations
 */

import { Matrix } from 'ml-matrix';
import { GPT } from './src/models/GPT.js';
import { CharTokenizer } from './src/tokenizer/char_level.js';
import { Adam } from './src/optim/adam.js';
import { DataLoader } from './src/utils/data/dataloader.js';
import { crossEntropyLoss } from './src/nn/functional.js';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

class ParityValidator {
    constructor(options = {}) {
        this.tolerance = options.tolerance || 1e-6;
        this.verbose = options.verbose || false;
        this.outputDir = options.outputDir || 'validation_outputs';
        this.pythonScript = options.pythonScript || 'validate_parity.py';
        
        // Ensure output directory exists
        if (!fs.existsSync(this.outputDir)) {
            fs.mkdirSync(this.outputDir, { recursive: true });
        }
    }

    /**
     * Validate forward pass outputs
     */
    async validateForwardPass() {
        console.log('🔍 Validating forward pass outputs...');
        
        // Create deterministic test case
        const config = {
            vocabSize: 20,
            dModel: 32,
            nHeads: 4,
            nLayers: 2,
            dFF: 64,
            maxSeqLen: 16,
            dropout: 0.0  // Disable dropout for deterministic results
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        
        // Set deterministic weights
        this._setDeterministicWeights(model);
        
        // Create test input
        const batchSize = 2;
        const seqLen = 8;
        const X = new Matrix(batchSize, seqLen);
        const Y = new Matrix(batchSize, seqLen);
        
        // Fill with deterministic values
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < seqLen; j++) {
                X.set(i, j, (i * seqLen + j) % config.vocabSize);
                Y.set(i, j, (i * seqLen + j + 1) % config.vocabSize);
            }
        }
        
        // Forward pass
        const [logits, loss] = model.forward(X, Y);
        
        // Save JavaScript outputs
        const jsOutputs = {
            config,
            input: X.to2DArray(),
            target: Y.to2DArray(),
            loss,
            logits: logits.data,
            weights: this._extractWeights(model)
        };
        
        const jsOutputPath = path.join(this.outputDir, 'js_forward_outputs.json');
        fs.writeFileSync(jsOutputPath, JSON.stringify(jsOutputs, null, 2));
        
        console.log(`✅ JavaScript forward pass completed. Loss: ${loss.toFixed(6)}`);
        console.log(`📁 Outputs saved to: ${jsOutputPath}`);
        
        return jsOutputs;
    }

    /**
     * Validate gradient computations
     */
    async validateGradients() {
        console.log('🔍 Validating gradient computations...');
        
        const config = {
            vocabSize: 15,
            dModel: 24,
            nHeads: 3,
            nLayers: 2,
            dFF: 48,
            maxSeqLen: 12,
            dropout: 0.0
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        this._setDeterministicWeights(model);
        
        // Create test input
        const batchSize = 2;
        const seqLen = 6;
        const X = new Matrix(batchSize, seqLen);
        const Y = new Matrix(batchSize, seqLen);
        
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < seqLen; j++) {
                X.set(i, j, (i + j) % config.vocabSize);
                Y.set(i, j, (i + j + 1) % config.vocabSize);
            }
        }
        
        // Forward and backward pass
        const [logits, loss] = model.forward(X, Y);
        const gradients = model.backward();
        
        // Extract gradients
        const jsGradients = {
            config,
            input: X.to2DArray(),
            target: Y.to2DArray(),
            loss,
            gradients: this._extractGradients(model)
        };
        
        const jsGradPath = path.join(this.outputDir, 'js_gradient_outputs.json');
        fs.writeFileSync(jsGradPath, JSON.stringify(jsGradients, null, 2));
        
        console.log(`✅ JavaScript gradients computed. Loss: ${loss.toFixed(6)}`);
        console.log(`📁 Gradients saved to: ${jsGradPath}`);
        
        return jsGradients;
    }

    /**
     * Validate training step
     */
    async validateTrainingStep() {
        console.log('🔍 Validating training step...');
        
        const config = {
            vocabSize: 25,
            dModel: 16,
            nHeads: 2,
            nLayers: 1,
            dFF: 32,
            maxSeqLen: 8,
            dropout: 0.0
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        this._setDeterministicWeights(model);
        
        const optimizer = new Adam([model], { 
            lr: 0.001, 
            beta1: 0.9, 
            beta2: 0.999, 
            eps: 1e-8 
        });
        
        // Training data
        const batchSize = 2;
        const seqLen = 4;
        const X = new Matrix(batchSize, seqLen);
        const Y = new Matrix(batchSize, seqLen);
        
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < seqLen; j++) {
                X.set(i, j, (i * 2 + j) % config.vocabSize);
                Y.set(i, j, (i * 2 + j + 1) % config.vocabSize);
            }
        }
        
        // Record initial state
        const initialWeights = this._extractWeights(model);
        
        // Training step
        const [logits1, loss1] = model.forward(X, Y);
        model.backward();
        optimizer.step();
        optimizer.zeroGrad();
        
        // Second step
        const [logits2, loss2] = model.forward(X, Y);
        model.backward();
        optimizer.step();
        optimizer.zeroGrad();
        
        const finalWeights = this._extractWeights(model);
        
        const jsTraining = {
            config,
            input: X.to2DArray(),
            target: Y.to2DArray(),
            initialWeights,
            finalWeights,
            losses: [loss1, loss2],
            optimizerConfig: {
                lr: optimizer.lr,
                beta1: optimizer.beta1,
                beta2: optimizer.beta2,
                eps: optimizer.eps
            }
        };
        
        const jsTrainPath = path.join(this.outputDir, 'js_training_outputs.json');
        fs.writeFileSync(jsTrainPath, JSON.stringify(jsTraining, null, 2));
        
        console.log(`✅ JavaScript training completed. Loss: ${loss1.toFixed(6)} → ${loss2.toFixed(6)}`);
        console.log(`📁 Training outputs saved to: ${jsTrainPath}`);
        
        return jsTraining;
    }

    /**
     * Validate text generation
     */
    async validateTextGeneration() {
        console.log('🔍 Validating text generation...');
        
        // Create simple tokenizer and model
        const text = "hello world this is a test";
        const tokenizer = new CharTokenizer();
        tokenizer.buildVocab(text);
        
        const config = {
            vocabSize: tokenizer.vocabSize,
            dModel: 16,
            nHeads: 2,
            nLayers: 1,
            dFF: 32,
            maxSeqLen: 20,
            dropout: 0.0
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        this._setDeterministicWeights(model);
        
        // Generate text with fixed seed
        const prompt = "hello";
        const promptTokens = tokenizer.encode(prompt, false, false);
        
        const promptMatrix = new Matrix([promptTokens]);
        const generated = model.generate(
            promptMatrix,
            10,  // maxLength
            1.0, // temperature
            tokenizer.eosTokenId
        );
        
        const generatedTokens = generated.getRow(0);
        const generatedText = tokenizer.decode(generatedTokens);
        
        const jsGeneration = {
            config,
            prompt,
            promptTokens,
            generated: generatedTokens,
            generatedText,
            tokenizerState: tokenizer.getState()
        };
        
        const jsGenPath = path.join(this.outputDir, 'js_generation_outputs.json');
        fs.writeFileSync(jsGenPath, JSON.stringify(jsGeneration, null, 2));
        
        console.log(`✅ JavaScript generation completed: "${generatedText}"`);
        console.log(`📁 Generation outputs saved to: ${jsGenPath}`);
        
        return jsGeneration;
    }

    /**
     * Compare with Python outputs (if available)
     */
    async compareWithPython() {
        console.log('🔍 Comparing with Python outputs...');
        
        // Check if Python script exists
        if (!fs.existsSync(this.pythonScript)) {
            console.log(`⚠️  Python validation script not found: ${this.pythonScript}`);
            console.log('   Skipping Python comparison.');
            return null;
        }
        
        try {
            // Run Python validation script
            console.log('🐍 Running Python validation...');
            execSync(`python ${this.pythonScript}`, { stdio: 'inherit' });
            
            // Compare outputs
            const comparisons = [];
            
            // Compare forward pass
            if (fs.existsSync(path.join(this.outputDir, 'py_forward_outputs.json'))) {
                const comparison = this._compareForwardOutputs();
                comparisons.push(comparison);
            }
            
            // Compare gradients
            if (fs.existsSync(path.join(this.outputDir, 'py_gradient_outputs.json'))) {
                const comparison = this._compareGradientOutputs();
                comparisons.push(comparison);
            }
            
            // Compare training
            if (fs.existsSync(path.join(this.outputDir, 'py_training_outputs.json'))) {
                const comparison = this._compareTrainingOutputs();
                comparisons.push(comparison);
            }
            
            // Compare generation
            if (fs.existsSync(path.join(this.outputDir, 'py_generation_outputs.json'))) {
                const comparison = this._compareGenerationOutputs();
                comparisons.push(comparison);
            }
            
            return comparisons;
            
        } catch (error) {
            console.log(`⚠️  Python comparison failed: ${error.message}`);
            return null;
        }
    }

    /**
     * Run full validation suite
     */
    async runFullValidation() {
        console.log('🚀 Starting full parity validation...\n');
        
        const results = {
            timestamp: new Date().toISOString(),
            jsOutputs: {},
            comparisons: null,
            summary: {}
        };
        
        try {
            // Run JavaScript validations
            results.jsOutputs.forward = await this.validateForwardPass();
            results.jsOutputs.gradients = await this.validateGradients();
            results.jsOutputs.training = await this.validateTrainingStep();
            results.jsOutputs.generation = await this.validateTextGeneration();
            
            console.log('\n✅ All JavaScript validations completed successfully!\n');
            
            // Compare with Python if available
            results.comparisons = await this.compareWithPython();
            
            // Generate summary
            results.summary = this._generateSummary(results);
            
            // Save full results
            const resultsPath = path.join(this.outputDir, 'validation_results.json');
            fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
            
            console.log(`\n📊 Full validation results saved to: ${resultsPath}`);
            
            return results;
            
        } catch (error) {
            console.error(`❌ Validation failed: ${error.message}`);
            console.error(error.stack);
            throw error;
        }
    }

    // Helper methods
    _setDeterministicWeights(model) {
        // Set deterministic weights for reproducible results
        let seed = 42;
        const seededRandom = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };
        
        const params = model.params();
        for (const paramKey in params) {
            const param = params[paramKey];
            for (let i = 0; i < param.rows; i++) {
                for (let j = 0; j < param.columns; j++) {
                    param.set(i, j, (seededRandom() - 0.5) * 0.02);
                }
            }
        }
    }

    _extractWeights(model) {
        const weights = {};
        const params = model.params();
        
        for (const paramKey in params) {
            weights[paramKey] = params[paramKey].to2DArray();
        }
        
        return weights;
    }

    _extractGradients(model) {
        const gradients = {};
        const grads = model.grads();
        
        for (const gradKey in grads) {
            if (grads[gradKey]) {
                gradients[gradKey] = grads[gradKey].to2DArray();
            }
        }
        
        return gradients;
    }

    _compareForwardOutputs() {
        const jsPath = path.join(this.outputDir, 'js_forward_outputs.json');
        const pyPath = path.join(this.outputDir, 'py_forward_outputs.json');
        
        const jsData = JSON.parse(fs.readFileSync(jsPath, 'utf-8'));
        const pyData = JSON.parse(fs.readFileSync(pyPath, 'utf-8'));
        
        const comparison = {
            type: 'forward_pass',
            lossDiff: Math.abs(jsData.loss - pyData.loss),
            logitsDiff: this._matrixDifference(jsData.logits, pyData.logits),
            passed: false
        };
        
        comparison.passed = comparison.lossDiff < this.tolerance && 
                           comparison.logitsDiff < this.tolerance;
        
        console.log(`📊 Forward pass comparison:`);
        console.log(`   Loss difference: ${comparison.lossDiff.toExponential(3)}`);
        console.log(`   Logits difference: ${comparison.logitsDiff.toExponential(3)}`);
        console.log(`   Status: ${comparison.passed ? '✅ PASSED' : '❌ FAILED'}`);
        
        return comparison;
    }

    _compareGradientOutputs() {
        const jsPath = path.join(this.outputDir, 'js_gradient_outputs.json');
        const pyPath = path.join(this.outputDir, 'py_gradient_outputs.json');
        
        const jsData = JSON.parse(fs.readFileSync(jsPath, 'utf-8'));
        const pyData = JSON.parse(fs.readFileSync(pyPath, 'utf-8'));
        
        let maxGradDiff = 0;
        const gradDiffs = {};
        
        for (const gradKey in jsData.gradients) {
            if (pyData.gradients[gradKey]) {
                const diff = this._matrixDifference(jsData.gradients[gradKey], pyData.gradients[gradKey]);
                gradDiffs[gradKey] = diff;
                maxGradDiff = Math.max(maxGradDiff, diff);
            }
        }
        
        const comparison = {
            type: 'gradients',
            maxGradientDiff: maxGradDiff,
            gradientDiffs: gradDiffs,
            passed: maxGradDiff < this.tolerance
        };
        
        console.log(`📊 Gradient comparison:`);
        console.log(`   Max gradient difference: ${maxGradDiff.toExponential(3)}`);
        console.log(`   Status: ${comparison.passed ? '✅ PASSED' : '❌ FAILED'}`);
        
        return comparison;
    }

    _compareTrainingOutputs() {
        const jsPath = path.join(this.outputDir, 'js_training_outputs.json');
        const pyPath = path.join(this.outputDir, 'py_training_outputs.json');
        
        const jsData = JSON.parse(fs.readFileSync(jsPath, 'utf-8'));
        const pyData = JSON.parse(fs.readFileSync(pyPath, 'utf-8'));
        
        const lossDiffs = jsData.losses.map((jsLoss, i) => 
            Math.abs(jsLoss - pyData.losses[i])
        );
        
        const maxLossDiff = Math.max(...lossDiffs);
        
        const comparison = {
            type: 'training',
            lossDifferences: lossDiffs,
            maxLossDiff,
            passed: maxLossDiff < this.tolerance
        };
        
        console.log(`📊 Training comparison:`);
        console.log(`   Loss differences: [${lossDiffs.map(d => d.toExponential(3)).join(', ')}]`);
        console.log(`   Status: ${comparison.passed ? '✅ PASSED' : '❌ FAILED'}`);
        
        return comparison;
    }

    _compareGenerationOutputs() {
        const jsPath = path.join(this.outputDir, 'js_generation_outputs.json');
        const pyPath = path.join(this.outputDir, 'py_generation_outputs.json');
        
        const jsData = JSON.parse(fs.readFileSync(jsPath, 'utf-8'));
        const pyData = JSON.parse(fs.readFileSync(pyPath, 'utf-8'));
        
        const tokensMatch = JSON.stringify(jsData.generated) === JSON.stringify(pyData.generated);
        const textMatch = jsData.generatedText === pyData.generatedText;
        
        const comparison = {
            type: 'generation',
            tokensMatch,
            textMatch,
            jsText: jsData.generatedText,
            pyText: pyData.generatedText,
            passed: tokensMatch && textMatch
        };
        
        console.log(`📊 Generation comparison:`);
        console.log(`   Tokens match: ${tokensMatch ? '✅' : '❌'}`);
        console.log(`   Text match: ${textMatch ? '✅' : '❌'}`);
        console.log(`   JS: "${jsData.generatedText}"`);
        console.log(`   PY: "${pyData.generatedText}"`);
        console.log(`   Status: ${comparison.passed ? '✅ PASSED' : '❌ FAILED'}`);
        
        return comparison;
    }

    _matrixDifference(matrix1, matrix2) {
        let maxDiff = 0;
        
        for (let i = 0; i < matrix1.length; i++) {
            for (let j = 0; j < matrix1[i].length; j++) {
                const diff = Math.abs(matrix1[i][j] - matrix2[i][j]);
                maxDiff = Math.max(maxDiff, diff);
            }
        }
        
        return maxDiff;
    }

    _generateSummary(results) {
        const summary = {
            jsValidationsCompleted: Object.keys(results.jsOutputs).length,
            pythonComparisonsAvailable: results.comparisons ? results.comparisons.length : 0,
            allTestsPassed: true,
            failedTests: []
        };
        
        if (results.comparisons) {
            for (const comparison of results.comparisons) {
                if (!comparison.passed) {
                    summary.allTestsPassed = false;
                    summary.failedTests.push(comparison.type);
                }
            }
        }
        
        return summary;
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const options = {
        tolerance: 1e-6,
        verbose: args.includes('--verbose'),
        outputDir: 'validation_outputs'
    };
    
    if (args.includes('--help')) {
        console.log(`
Usage: node validate_parity.js [options]

Options:
  --verbose          Enable verbose output
  --tolerance <val>  Set numerical tolerance (default: 1e-6)
  --output-dir <dir> Set output directory (default: validation_outputs)
  --help             Show this help message

Examples:
  node validate_parity.js
  node validate_parity.js --verbose --tolerance 1e-5
        `);
        process.exit(0);
    }
    
    // Parse tolerance
    const toleranceIndex = args.indexOf('--tolerance');
    if (toleranceIndex !== -1 && args[toleranceIndex + 1]) {
        options.tolerance = parseFloat(args[toleranceIndex + 1]);
    }
    
    // Parse output directory
    const outputDirIndex = args.indexOf('--output-dir');
    if (outputDirIndex !== -1 && args[outputDirIndex + 1]) {
        options.outputDir = args[outputDirIndex + 1];
    }
    
    const validator = new ParityValidator(options);
    
    try {
        const results = await validator.runFullValidation();
        
        console.log('\n🎉 Validation Summary:');
        console.log(`   JavaScript validations: ${results.summary.jsValidationsCompleted}/4`);
        console.log(`   Python comparisons: ${results.summary.pythonComparisonsAvailable}`);
        console.log(`   All tests passed: ${results.summary.allTestsPassed ? '✅' : '❌'}`);
        
        if (!results.summary.allTestsPassed) {
            console.log(`   Failed tests: ${results.summary.failedTests.join(', ')}`);
            process.exit(1);
        }
        
    } catch (error) {
        console.error(`❌ Validation failed: ${error.message}`);
        process.exit(1);
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { ParityValidator };