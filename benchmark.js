#!/usr/bin/env node

/**
 * Performance benchmarking script for numpyGPT JavaScript implementation
 * Measures training speed, inference speed, memory usage, and throughput
 */

import { Matrix } from 'ml-matrix';
import { GPT } from './src/models/GPT.js';
import { CharTokenizer } from './src/tokenizer/char_level.js';
import { Adam } from './src/optim/adam.js';
import { DataLoader } from './src/utils/data/dataloader.js';
import fs from 'fs';
import path from 'path';

class PerformanceBenchmark {
    constructor(options = {}) {
        this.verbose = options.verbose || false;
        this.outputDir = options.outputDir || 'benchmark_results';
        this.warmupSteps = options.warmupSteps || 5;
        this.benchmarkSteps = options.benchmarkSteps || 20;
        
        // Ensure output directory exists
        if (!fs.existsSync(this.outputDir)) {
            fs.mkdirSync(this.outputDir, { recursive: true });
        }
    }

    /**
     * Benchmark forward pass performance
     */
    async benchmarkForwardPass() {
        console.log('🔥 Benchmarking forward pass performance...');
        
        const results = {};
        const modelSizes = [
            { name: 'tiny', vocabSize: 50, dModel: 64, nHeads: 2, nLayers: 2, dFF: 128, maxSeqLen: 32 },
            { name: 'small', vocabSize: 100, dModel: 128, nHeads: 4, nLayers: 4, dFF: 256, maxSeqLen: 64 },
            { name: 'medium', vocabSize: 200, dModel: 256, nHeads: 8, nLayers: 6, dFF: 512, maxSeqLen: 128 }
        ];
        
        for (const config of modelSizes) {
            console.log(`  Testing ${config.name} model...`);
            
            const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
            
            // Test different batch sizes
            const batchSizes = [1, 2, 4, 8];
            const seqLen = Math.min(32, config.maxSeqLen);
            
            results[config.name] = {
                config,
                batchResults: {}
            };
            
            for (const batchSize of batchSizes) {
                const input = Matrix.zeros(batchSize, seqLen);
                
                // Fill with random token indices
                for (let i = 0; i < batchSize; i++) {
                    for (let j = 0; j < seqLen; j++) {
                        input.set(i, j, Math.floor(Math.random() * config.vocabSize));
                    }
                }
                
                // Warmup
                for (let i = 0; i < this.warmupSteps; i++) {
                    model.forward(input);
                }
                
                // Benchmark
                const startTime = process.hrtime.bigint();
                const startMemory = process.memoryUsage();
                
                for (let i = 0; i < this.benchmarkSteps; i++) {
                    model.forward(input);
                }
                
                const endTime = process.hrtime.bigint();
                const endMemory = process.memoryUsage();
                
                const totalTimeMs = Number(endTime - startTime) / 1e6;
                const avgTimeMs = totalTimeMs / this.benchmarkSteps;
                const tokensPerSecond = (batchSize * seqLen * this.benchmarkSteps) / (totalTimeMs / 1000);
                
                results[config.name].batchResults[batchSize] = {
                    batchSize,
                    seqLen,
                    totalTimeMs,
                    avgTimeMs,
                    tokensPerSecond,
                    memoryDelta: {
                        rss: endMemory.rss - startMemory.rss,
                        heapUsed: endMemory.heapUsed - startMemory.heapUsed,
                        heapTotal: endMemory.heapTotal - startMemory.heapTotal
                    }
                };
                
                console.log(`    Batch ${batchSize}: ${avgTimeMs.toFixed(2)}ms/step, ${tokensPerSecond.toFixed(0)} tokens/sec`);
            }
        }
        
        return results;
    }

    /**
     * Benchmark training performance
     */
    async benchmarkTraining() {
        console.log('🏋️ Benchmarking training performance...');
        
        const config = {
            vocabSize: 100,
            dModel: 128,
            nHeads: 4,
            nLayers: 2,
            dFF: 256,
            maxSeqLen: 64
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        const optimizer = new Adam([model], { lr: 0.001 });
        
        const batchSize = 4;
        const seqLen = 32;
        
        // Create training data
        const X = Matrix.zeros(batchSize, seqLen);
        const Y = Matrix.zeros(batchSize, seqLen);
        
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < seqLen; j++) {
                X.set(i, j, Math.floor(Math.random() * config.vocabSize));
                Y.set(i, j, Math.floor(Math.random() * config.vocabSize));
            }
        }
        
        // Warmup
        for (let i = 0; i < this.warmupSteps; i++) {
            const [logits, loss] = model.forward(X, Y);
            model.backward();
            optimizer.step();
            optimizer.zeroGrad();
        }
        
        // Benchmark training steps
        const losses = [];
        const stepTimes = [];
        const startTime = process.hrtime.bigint();
        const startMemory = process.memoryUsage();
        
        for (let i = 0; i < this.benchmarkSteps; i++) {
            const stepStart = process.hrtime.bigint();
            
            const [logits, loss] = model.forward(X, Y);
            model.backward();
            optimizer.step();
            optimizer.zeroGrad();
            
            const stepEnd = process.hrtime.bigint();
            const stepTimeMs = Number(stepEnd - stepStart) / 1e6;
            
            losses.push(loss);
            stepTimes.push(stepTimeMs);
        }
        
        const endTime = process.hrtime.bigint();
        const endMemory = process.memoryUsage();
        
        const totalTimeMs = Number(endTime - startTime) / 1e6;
        const avgStepTime = stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length;
        const tokensPerSecond = (batchSize * seqLen * this.benchmarkSteps) / (totalTimeMs / 1000);
        
        return {
            config,
            batchSize,
            seqLen,
            steps: this.benchmarkSteps,
            totalTimeMs,
            avgStepTimeMs: avgStepTime,
            tokensPerSecond,
            losses: {
                initial: losses[0],
                final: losses[losses.length - 1],
                avg: losses.reduce((a, b) => a + b, 0) / losses.length
            },
            memoryUsage: {
                initial: startMemory,
                final: endMemory,
                delta: {
                    rss: endMemory.rss - startMemory.rss,
                    heapUsed: endMemory.heapUsed - startMemory.heapUsed,
                    heapTotal: endMemory.heapTotal - startMemory.heapTotal
                }
            }
        };
    }

    /**
     * Benchmark text generation performance
     */
    async benchmarkGeneration() {
        console.log('📝 Benchmarking text generation performance...');
        
        // Create simple tokenizer and model
        const text = "hello world this is a test of the text generation system";
        const tokenizer = new CharTokenizer();
        tokenizer.buildVocab(text);
        
        const config = {
            vocabSize: tokenizer.vocabSize,
            dModel: 64,
            nHeads: 2,
            nLayers: 2,
            dFF: 128,
            maxSeqLen: 64
        };
        
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        
        const results = {};
        const generationLengths = [10, 25, 50, 100];
        
        for (const maxLength of generationLengths) {
            if (maxLength > config.maxSeqLen - 5) continue; // Leave room for prompt
            
            const prompt = "hello";
            const promptTokens = tokenizer.encode(prompt, false, false);
            const promptMatrix = new Matrix([promptTokens]);
            
            // Warmup
            for (let i = 0; i < 3; i++) {
                model.generate(promptMatrix, 5, 1.0, tokenizer.eosTokenId);
            }
            
            // Benchmark
            const times = [];
            const generatedTexts = [];
            
            for (let i = 0; i < 10; i++) {
                const startTime = process.hrtime.bigint();
                
                const generated = model.generate(promptMatrix, maxLength, 1.0, tokenizer.eosTokenId);
                const generatedTokens = generated.getRow(0);
                const generatedText = tokenizer.decode(generatedTokens);
                
                const endTime = process.hrtime.bigint();
                const timeMs = Number(endTime - startTime) / 1e6;
                
                times.push(timeMs);
                generatedTexts.push(generatedText);
            }
            
            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const tokensPerSecond = maxLength / (avgTime / 1000);
            
            results[maxLength] = {
                maxLength,
                avgTimeMs: avgTime,
                tokensPerSecond,
                sampleTexts: generatedTexts.slice(0, 3) // Keep first 3 samples
            };
            
            console.log(`    Length ${maxLength}: ${avgTime.toFixed(2)}ms, ${tokensPerSecond.toFixed(1)} tokens/sec`);
        }
        
        return results;
    }

    /**
     * Benchmark memory usage patterns
     */
    async benchmarkMemory() {
        console.log('💾 Benchmarking memory usage patterns...');
        
        const config = {
            vocabSize: 100,
            dModel: 128,
            nHeads: 4,
            nLayers: 3,
            dFF: 256,
            maxSeqLen: 64
        };
        
        const memorySnapshots = [];
        
        // Baseline memory
        const baseline = process.memoryUsage();
        memorySnapshots.push({ stage: 'baseline', memory: baseline });
        
        // Model creation
        const model = new GPT(config.vocabSize, config.maxSeqLen, config.dModel, config.nHeads, config.nLayers, config.dFF);
        const afterModel = process.memoryUsage();
        memorySnapshots.push({ stage: 'after_model_creation', memory: afterModel });
        
        // Optimizer creation
        const optimizer = new Adam([model], { lr: 0.001 });
        const afterOptimizer = process.memoryUsage();
        memorySnapshots.push({ stage: 'after_optimizer_creation', memory: afterOptimizer });
        
        // Forward pass
        const batchSize = 4;
        const seqLen = 32;
        const X = Matrix.zeros(batchSize, seqLen);
        const Y = Matrix.zeros(batchSize, seqLen);
        
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < seqLen; j++) {
                X.set(i, j, Math.floor(Math.random() * config.vocabSize));
                Y.set(i, j, Math.floor(Math.random() * config.vocabSize));
            }
        }
        
        const [logits, loss] = model.forward(X, Y);
        const afterForward = process.memoryUsage();
        memorySnapshots.push({ stage: 'after_forward_pass', memory: afterForward });
        
        // Backward pass
        model.backward();
        const afterBackward = process.memoryUsage();
        memorySnapshots.push({ stage: 'after_backward_pass', memory: afterBackward });
        
        // Optimizer step
        optimizer.step();
        optimizer.zeroGrad();
        const afterOptimStep = process.memoryUsage();
        memorySnapshots.push({ stage: 'after_optimizer_step', memory: afterOptimStep });
        
        // Calculate deltas
        const results = {
            config,
            snapshots: memorySnapshots,
            deltas: {}
        };
        
        for (let i = 1; i < memorySnapshots.length; i++) {
            const current = memorySnapshots[i];
            const previous = memorySnapshots[i - 1];
            
            results.deltas[current.stage] = {
                rss: current.memory.rss - previous.memory.rss,
                heapUsed: current.memory.heapUsed - previous.memory.heapUsed,
                heapTotal: current.memory.heapTotal - previous.memory.heapTotal,
                external: current.memory.external - previous.memory.external
            };
        }
        
        return results;
    }

    /**
     * Run comprehensive benchmark suite
     */
    async runBenchmarks() {
        console.log('🚀 Starting comprehensive performance benchmarks...\n');
        
        const results = {
            timestamp: new Date().toISOString(),
            nodeVersion: process.version,
            platform: process.platform,
            arch: process.arch,
            benchmarks: {}
        };
        
        try {
            // Forward pass benchmarks
            results.benchmarks.forwardPass = await this.benchmarkForwardPass();
            console.log('');
            
            // Training benchmarks
            results.benchmarks.training = await this.benchmarkTraining();
            console.log('');
            
            // Generation benchmarks
            results.benchmarks.generation = await this.benchmarkGeneration();
            console.log('');
            
            // Memory benchmarks
            results.benchmarks.memory = await this.benchmarkMemory();
            console.log('');
            
            // Generate summary
            results.summary = this._generateSummary(results.benchmarks);
            
            // Save results
            const resultsPath = path.join(this.outputDir, 'benchmark_results.json');
            fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
            
            console.log(`📊 Benchmark results saved to: ${resultsPath}`);
            
            // Print summary
            this._printSummary(results.summary);
            
            return results;
            
        } catch (error) {
            console.error(`❌ Benchmark failed: ${error.message}`);
            console.error(error.stack);
            throw error;
        }
    }

    _generateSummary(benchmarks) {
        const summary = {
            forwardPass: {},
            training: {},
            generation: {},
            memory: {}
        };
        
        // Forward pass summary
        if (benchmarks.forwardPass) {
            for (const [modelSize, results] of Object.entries(benchmarks.forwardPass)) {
                const batchResults = Object.values(results.batchResults);
                summary.forwardPass[modelSize] = {
                    avgTimeMs: batchResults.reduce((sum, r) => sum + r.avgTimeMs, 0) / batchResults.length,
                    maxThroughput: Math.max(...batchResults.map(r => r.tokensPerSecond)),
                    paramCount: this._estimateParamCount(results.config)
                };
            }
        }
        
        // Training summary
        if (benchmarks.training) {
            summary.training = {
                avgStepTimeMs: benchmarks.training.avgStepTimeMs,
                tokensPerSecond: benchmarks.training.tokensPerSecond,
                lossReduction: benchmarks.training.losses.initial - benchmarks.training.losses.final,
                memoryUsageMB: benchmarks.training.memoryUsage.delta.heapUsed / (1024 * 1024)
            };
        }
        
        // Generation summary
        if (benchmarks.generation) {
            const genResults = Object.values(benchmarks.generation);
            summary.generation = {
                avgTimeMs: genResults.reduce((sum, r) => sum + r.avgTimeMs, 0) / genResults.length,
                maxThroughput: Math.max(...genResults.map(r => r.tokensPerSecond))
            };
        }
        
        // Memory summary
        if (benchmarks.memory) {
            const totalMemoryMB = benchmarks.memory.snapshots[benchmarks.memory.snapshots.length - 1].memory.heapUsed / (1024 * 1024);
            summary.memory = {
                totalMemoryMB,
                modelMemoryMB: benchmarks.memory.deltas.after_model_creation?.heapUsed / (1024 * 1024) || 0,
                peakMemoryMB: Math.max(...benchmarks.memory.snapshots.map(s => s.memory.heapUsed)) / (1024 * 1024)
            };
        }
        
        return summary;
    }

    _estimateParamCount(config) {
        // Rough parameter count estimation
        const embeddingParams = config.vocabSize * config.dModel;
        const transformerParams = config.nLayers * (
            4 * config.dModel * config.dModel + // Attention weights
            2 * config.dModel * config.dFF +    // FFN weights
            4 * config.dModel                   // Layer norm and biases
        );
        const outputParams = config.dModel * config.vocabSize;
        
        return embeddingParams + transformerParams + outputParams;
    }

    _printSummary(summary) {
        console.log('📈 Performance Summary:');
        console.log('');
        
        if (summary.forwardPass) {
            console.log('🔥 Forward Pass:');
            for (const [modelSize, stats] of Object.entries(summary.forwardPass)) {
                console.log(`   ${modelSize}: ${stats.avgTimeMs.toFixed(2)}ms avg, ${stats.maxThroughput.toFixed(0)} tokens/sec max, ~${(stats.paramCount / 1e6).toFixed(1)}M params`);
            }
            console.log('');
        }
        
        if (summary.training) {
            console.log('🏋️ Training:');
            console.log(`   Step time: ${summary.training.avgStepTimeMs.toFixed(2)}ms`);
            console.log(`   Throughput: ${summary.training.tokensPerSecond.toFixed(0)} tokens/sec`);
            console.log(`   Memory usage: ${summary.training.memoryUsageMB.toFixed(1)}MB`);
            console.log('');
        }
        
        if (summary.generation) {
            console.log('📝 Generation:');
            console.log(`   Avg time: ${summary.generation.avgTimeMs.toFixed(2)}ms`);
            console.log(`   Max throughput: ${summary.generation.maxThroughput.toFixed(1)} tokens/sec`);
            console.log('');
        }
        
        if (summary.memory) {
            console.log('💾 Memory:');
            console.log(`   Total usage: ${summary.memory.totalMemoryMB.toFixed(1)}MB`);
            console.log(`   Model size: ${summary.memory.modelMemoryMB.toFixed(1)}MB`);
            console.log(`   Peak usage: ${summary.memory.peakMemoryMB.toFixed(1)}MB`);
        }
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const options = {
        verbose: args.includes('--verbose'),
        warmupSteps: 5,
        benchmarkSteps: 20,
        outputDir: 'benchmark_results'
    };
    
    if (args.includes('--help')) {
        console.log(`
Usage: node benchmark.js [options]

Options:
  --verbose              Enable verbose output
  --warmup-steps <num>   Number of warmup steps (default: 5)
  --benchmark-steps <num> Number of benchmark steps (default: 20)
  --output-dir <dir>     Output directory (default: benchmark_results)
  --help                 Show this help message

Examples:
  node benchmark.js
  node benchmark.js --verbose --benchmark-steps 50
        `);
        process.exit(0);
    }
    
    // Parse warmup steps
    const warmupIndex = args.indexOf('--warmup-steps');
    if (warmupIndex !== -1 && args[warmupIndex + 1]) {
        options.warmupSteps = parseInt(args[warmupIndex + 1]);
    }
    
    // Parse benchmark steps
    const benchmarkIndex = args.indexOf('--benchmark-steps');
    if (benchmarkIndex !== -1 && args[benchmarkIndex + 1]) {
        options.benchmarkSteps = parseInt(args[benchmarkIndex + 1]);
    }
    
    // Parse output directory
    const outputDirIndex = args.indexOf('--output-dir');
    if (outputDirIndex !== -1 && args[outputDirIndex + 1]) {
        options.outputDir = args[outputDirIndex + 1];
    }
    
    const benchmark = new PerformanceBenchmark(options);
    
    try {
        await benchmark.runBenchmarks();
        console.log('\n🎉 All benchmarks completed successfully!');
        
    } catch (error) {
        console.error(`❌ Benchmark failed: ${error.message}`);
        process.exit(1);
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { PerformanceBenchmark };