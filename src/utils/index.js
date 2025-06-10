/**
 * Utilities Module Index for numpyGPT JavaScript Implementation
 * 
 * Provides clean exports for all utility modules:
 * - Training utilities (logging, monitoring, gradient clipping)
 * - Visualization utilities (metrics logging, plotting)
 * - Data utilities (DataLoader, dataset analysis)
 */

// Training utilities
export {
  setupLogger,
  clipGradNorm,
  getLr,
  TrainingMonitor,
  EarlyStopping,
  LRFinder
} from './training.js';

// Visualization utilities
export {
  MetricsLogger,
  quickPlot,
  textHistogram
} from './vis.js';

// Data utilities
export {
  DataLoader,
  MultiDataLoader,
  createDataLoaders,
  analyzeDataset
} from './data/dataloader.js';