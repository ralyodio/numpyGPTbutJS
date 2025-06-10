/**
 * TypeScript definitions for numpyGPT JavaScript Implementation
 * Provides type safety and IDE support for the JavaScript codebase
 */

import { Matrix } from 'ml-matrix';

// Base Module Types
export declare class Module {
  training: boolean;
  constructor();
  forward(x: any): any;
  backward(grad: any): any;
  train(): void;
  eval(): void;
  params(): Record<string, Matrix>;
  grads(): Record<string, Matrix>;
  __call__(...args: any[]): any;
}

// Neural Network Module Types
export declare class Linear extends Module {
  inFeatures: number;
  outFeatures: number;
  W: Matrix;
  b: Matrix;
  dW: Matrix | null;
  db: Matrix | null;
  cacheInput: Matrix | null;
  
  constructor(inFeatures: number, outFeatures: number);
  forward(X: Matrix | number[][]): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): { W: Matrix; b: Matrix };
  grads(): { W: Matrix | null; b: Matrix | null };
}

export declare class Embedding extends Module {
  vocabSize: number;
  embeddingDim: number;
  weight: Matrix;
  dWeight: Matrix | null;
  cacheInput: Matrix | null;
  
  constructor(vocabSize: number, embeddingDim: number);
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): { weight: Matrix };
  grads(): { weight: Matrix | null };
}

export declare class LayerNorm extends Module {
  normalizedShape: number;
  eps: number;
  weight: Matrix;
  bias: Matrix;
  dWeight: Matrix | null;
  dBias: Matrix | null;
  cache: any;
  
  constructor(normalizedShape: number, eps?: number);
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): { weight: Matrix; bias: Matrix };
  grads(): { weight: Matrix | null; bias: Matrix | null };
}

export declare class ReLU extends Module {
  cache: Matrix | null;
  
  constructor();
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
}

export declare class Softmax extends Module {
  cache: Matrix | null;
  
  constructor();
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
}

export declare class PositionalEncoding extends Module {
  maxLen: number;
  dModel: number;
  pe: Matrix;
  
  constructor(maxLen: number, dModel: number);
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): { pe: Matrix };
  grads(): { pe: Matrix };
}

export declare class MultiHeadAttention extends Module {
  dModel: number;
  nHeads: number;
  dK: number;
  WQ: Linear;
  WK: Linear;
  WV: Linear;
  WO: Linear;
  softmax: Softmax;
  cache: any;
  batchInfo: { B: number; T: number };
  
  constructor(dModel: number, nHeads: number);
  forward(X: Matrix, B: number, T: number, mask?: Matrix | null): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): Record<string, Matrix>;
  grads(): Record<string, Matrix>;
}

export declare class FeedForward extends Module {
  dModel: number;
  dFF: number;
  linear1: Linear;
  linear2: Linear;
  relu: ReLU;
  
  constructor(dModel: number, dFF: number);
  forward(X: Matrix): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): Record<string, Matrix>;
  grads(): Record<string, Matrix>;
}

export declare class TransformerBlock extends Module {
  dModel: number;
  nHeads: number;
  dFF: number;
  attn: MultiHeadAttention;
  ff: FeedForward;
  ln1: LayerNorm;
  ln2: LayerNorm;
  
  constructor(dModel: number, nHeads: number, dFF: number);
  forward(X: Matrix, B: number, T: number, mask?: Matrix | null): Matrix;
  backward(dZ: Matrix): Matrix;
  params(): Record<string, Matrix>;
  grads(): Record<string, Matrix>;
}

// GPT Model Types
export declare class GPT extends Module {
  vocabSize: number;
  maxLen: number;
  dModel: number;
  tokEmb: Embedding;
  posEmb: PositionalEncoding;
  blocks: TransformerBlock[];
  lnF: LayerNorm;
  lmHead: Linear;
  cache: any;
  
  constructor(
    vocabSize: number,
    maxLen: number,
    dModel: number,
    nHeads: number,
    nLayers: number,
    dFF: number
  );
  
  forward(X: Matrix, targets?: Matrix | null): Matrix | [Matrix, number];
  backward(): void;
  generate(
    idx: Matrix,
    maxNewTokens: number,
    temperature?: number,
    eosTokenId?: number | null
  ): Matrix;
  params(): Record<string, Matrix>;
  grads(): Record<string, Matrix>;
}

// Optimizer Types
export declare class Optimizer {
  params: Module[];
  lr: number;
  
  constructor(modules: Module[], lr: number);
  step(): void;
  zeroGrad(): void;
}

export declare class Adam extends Optimizer {
  beta1: number;
  beta2: number;
  eps: number;
  t: number;
  m: Record<string, Matrix>[];
  v: Record<string, Matrix>[];
  
  constructor(
    modules: Module[],
    lr?: number,
    betas?: [number, number],
    eps?: number
  );
  
  step(): void;
  getState(): {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    t: number;
    numParams: number;
  };
}

// Learning Rate Scheduler Types
export declare class LRScheduler {
  optimizer: Optimizer;
  
  constructor(optimizer: Optimizer);
  step(): void;
  getLr(): number;
}

export declare class StepLR extends LRScheduler {
  stepSize: number;
  gamma: number;
  lastEpoch: number;
  
  constructor(optimizer: Optimizer, stepSize: number, gamma?: number);
  step(): void;
}

export declare class WarmupCosineLR extends LRScheduler {
  warmupSteps: number;
  totalSteps: number;
  minLr: number;
  baseLr: number;
  currentStep: number;
  
  constructor(
    optimizer: Optimizer,
    warmupSteps: number,
    totalSteps: number,
    minLr?: number
  );
  
  step(): void;
}

// Tokenizer Types
export declare class CharTokenizer {
  chars: string[];
  stoi: Record<string, number>;
  itos: Record<number, string>;
  
  constructor();
  buildVocab(text: string): void;
  encode(text: string, addBos?: boolean, addEos?: boolean): number[];
  decode(tokens: number[]): string;
  
  get vocabSize(): number;
  get eosTokenId(): number;
  get bosTokenId(): number;
  get padTokenId(): number;
  get unkTokenId(): number;
  
  getState(): {
    chars: string[];
    stoi: Record<string, number>;
    itos: Record<number, string>;
    vocabSize: number;
  };
  
  loadState(state: {
    chars: string[];
    stoi: Record<string, number>;
    itos: Record<number, string>;
  }): void;
}

export declare class WordTokenizer {
  words: string[];
  stoi: Record<string, number>;
  itos: Record<number, string>;
  
  constructor();
  buildVocab(text: string): void;
  encode(text: string, addBos?: boolean, addEos?: boolean): number[];
  decode(tokens: number[]): string;
  
  get vocabSize(): number;
  get eosTokenId(): number;
  get bosTokenId(): number;
  get padTokenId(): number;
  get unkTokenId(): number;
  
  getState(): any;
  loadState(state: any): void;
}

export declare class BPETokenizer {
  merges: Array<[string, string]>;
  vocab: Record<string, number>;
  
  constructor();
  buildVocab(text: string, vocabSize: number): void;
  encode(text: string, addBos?: boolean, addEos?: boolean): number[];
  decode(tokens: number[]): string;
  
  get vocabSize(): number;
  get eosTokenId(): number;
  get bosTokenId(): number;
  get padTokenId(): number;
  get unkTokenId(): number;
  
  getState(): any;
  loadState(state: any): void;
}

// Utility Types
export declare class DataLoader {
  batchSize: number;
  blockSize: number;
  data: number[];
  vocabSize: number;
  
  constructor(dataDir: string, split: string, batchSize: number, blockSize: number);
  getBatch(): { X: Matrix; Y: Matrix };
}

export declare class TrainingMonitor {
  logInterval: number;
  stepTimes: number[];
  losses: number[];
  startTime: number;
  lastLogTime: number;
  
  constructor(logInterval?: number);
  logStep(iterNum: number, loss: number, lr: number, gradNorm?: number | null): string | null;
  getAvgLoss(window?: number): number;
  getStats(): {
    totalSteps: number;
    totalTime: number;
    avgStepTime: number;
    avgLoss: number;
    recentLoss: number;
    stepsPerSecond: number;
  };
  reset(): void;
  isConverging(window?: number, threshold?: number): boolean;
  isStalled(window?: number, threshold?: number): boolean;
}

export declare class EarlyStopping {
  patience: number;
  minDelta: number;
  restore: boolean;
  bestScore: number;
  counter: number;
  bestWeights: any;
  
  constructor(patience?: number, minDelta?: number, restore?: boolean);
  shouldStop(score: number, model?: Module | null): boolean;
  restoreBestWeights(model: Module): void;
}

export declare class LRFinder {
  model: Module;
  optimizer: Optimizer;
  startLr: number;
  endLr: number;
  numSteps: number;
  lrs: number[];
  losses: number[];
  
  constructor(
    model: Module,
    optimizer: Optimizer,
    startLr?: number,
    endLr?: number,
    numSteps?: number
  );
  
  find(trainStep: () => number): {
    learningRates: number[];
    losses: number[];
    suggestedLr: number;
  };
}

export declare class MetricsLogger {
  filepath: string;
  metrics: Array<{
    iter: number;
    timestamp: number;
    train_loss?: number;
    val_loss?: number;
    lr?: number;
    grad_norm?: number;
  }>;
  
  constructor(filepath: string);
  log(
    iterNum: number,
    trainLoss?: number | null,
    valLoss?: number | null,
    lr?: number | null,
    gradNorm?: number | null
  ): void;
}

export declare class PerformanceBenchmark {
  verbose: boolean;
  outputDir: string;
  warmupSteps: number;
  benchmarkSteps: number;
  
  constructor(options?: {
    verbose?: boolean;
    outputDir?: string;
    warmupSteps?: number;
    benchmarkSteps?: number;
  });
  
  benchmarkForwardPass(): Promise<any>;
  benchmarkTraining(): Promise<any>;
  benchmarkGeneration(): Promise<any>;
  benchmarkMemory(): Promise<any>;
  runBenchmarks(): Promise<any>;
}

// Functional Types
export declare function crossEntropyLoss(logits: Matrix, targets: number[]): number;
export declare function softmax(X: Matrix): Matrix;

// Training Utility Functions
export declare function setupLogger(name?: string, level?: string): {
  name: string;
  level: number;
  debug(message: string, ...args: any[]): void;
  info(message: string, ...args: any[]): void;
  warn(message: string, ...args: any[]): void;
  error(message: string, ...args: any[]): void;
};

export declare function clipGradNorm(model: Module, maxNorm: number): number;
export declare function getLr(optimizer: Optimizer): number;

// Data Generation Functions
export declare function generateData(
  text: string,
  tokenizerType: 'char' | 'word' | 'bpe',
  outputDir: string,
  options?: any
): void;

export declare function loadTokenizer(filepath: string): CharTokenizer | WordTokenizer | BPETokenizer;
export declare function loadData(filepath: string): number[];

// Main Script Functions
export declare function train(): Promise<void>;
export declare function sample(
  modelPath: string,
  prompt: string,
  maxNewTokens?: number,
  temperature?: number
): Promise<string>;

// Configuration Types
export interface TrainingConfig {
  dataDir: string;
  outDir: string;
  alwaysSaveCheckpoint: boolean;
  resume: boolean;
  batchSize: number;
  blockSize: number;
  lr: number;
  minLr: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dFF: number;
  maxIters: number;
  warmupIters: number;
  lrDecayIters: number;
  evalInterval: number;
  evalIters: number;
  logInterval: number;
  gradClip: number;
}

export interface ModelConfig {
  vocabSize: number;
  maxLen: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFF: number;
}

export interface BenchmarkResults {
  timestamp: string;
  nodeVersion: string;
  platform: string;
  arch: string;
  benchmarks: {
    forwardPass: any;
    training: any;
    generation: any;
    memory: any;
  };
  summary: any;
}