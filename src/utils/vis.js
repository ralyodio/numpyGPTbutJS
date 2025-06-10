/**
 * Visualization Utilities for numpyGPT JavaScript Implementation
 * JavaScript equivalent of numpyGPT/utils/vis.py
 * 
 * Provides advanced metrics logging and visualization capabilities:
 * - Comprehensive metrics collection and persistence
 * - Text-based plotting for terminal environments
 * - Data export for external visualization tools
 * - Training progress analysis and insights
 */

import fs from 'fs';
import path from 'path';

/**
 * Advanced Metrics Logger for training visualization
 * Provides comprehensive metrics collection with persistence and analysis
 */
export class MetricsLogger {
  /**
   * Initialize metrics logger
   * @param {string} logFile - Path to metrics log file
   */
  constructor(logFile = 'metrics.json') {
    this.logFile = logFile;
    this.metrics = {
      iterations: [],
      train_loss: [],
      val_loss: [],
      grad_norm: [],
      lr: [],
      timestamps: []
    };
    
    // Load existing metrics if file exists
    if (fs.existsSync(logFile)) {
      try {
        const data = fs.readFileSync(logFile, 'utf-8');
        this.metrics = JSON.parse(data);
        
        // Ensure all required fields exist
        if (!this.metrics.timestamps) {
          this.metrics.timestamps = new Array(this.metrics.iterations.length).fill(null);
        }
      } catch (error) {
        console.warn(`Warning: Could not load existing metrics from ${logFile}:`, error.message);
      }
    }
  }
  
  /**
   * Log training metrics
   * @param {number} iterNum - Current iteration number
   * @param {number} trainLoss - Training loss (optional)
   * @param {number} valLoss - Validation loss (optional)
   * @param {number} gradNorm - Gradient norm (optional)
   * @param {number} lr - Learning rate (optional)
   */
  log(iterNum, trainLoss = null, valLoss = null, gradNorm = null, lr = null) {
    this.metrics.iterations.push(iterNum);
    this.metrics.train_loss.push(trainLoss);
    this.metrics.val_loss.push(valLoss);
    this.metrics.grad_norm.push(gradNorm);
    this.metrics.lr.push(lr);
    this.metrics.timestamps.push(new Date().toISOString());
    
    // Save to file immediately for persistence
    try {
      fs.writeFileSync(this.logFile, JSON.stringify(this.metrics, null, 2));
    } catch (error) {
      console.warn(`Warning: Could not save metrics to ${this.logFile}:`, error.message);
    }
  }
  
  /**
   * Create comprehensive training plots
   * @param {string} savePath - Path to save plot data
   * @returns {string} - Path to saved plot data
   */
  plot(savePath = 'training_curves.json') {
    const plotData = this._generatePlotData();
    
    // Save plot data as JSON for external tools
    fs.writeFileSync(savePath, JSON.stringify(plotData, null, 2));
    
    // Generate text-based plots for terminal viewing
    this._generateTextPlots();
    
    // Generate analysis report
    this._generateAnalysisReport();
    
    return savePath;
  }
  
  /**
   * Generate plot data structure
   * @returns {Object} - Structured plot data
   */
  _generatePlotData() {
    const iterations = this.metrics.iterations;
    
    // Filter out null values and create coordinate pairs
    const trainLossData = iterations
      .map((iter, i) => ({ x: iter, y: this.metrics.train_loss[i] }))
      .filter(point => point.y !== null);
    
    const valLossData = iterations
      .map((iter, i) => ({ x: iter, y: this.metrics.val_loss[i] }))
      .filter(point => point.y !== null);
    
    const gradNormData = iterations
      .map((iter, i) => ({ x: iter, y: this.metrics.grad_norm[i] }))
      .filter(point => point.y !== null);
    
    const lrData = iterations
      .map((iter, i) => ({ x: iter, y: this.metrics.lr[i] }))
      .filter(point => point.y !== null);
    
    // Calculate validation loss differences for trend analysis
    const valLossDiff = [];
    for (let i = 1; i < valLossData.length; i++) {
      valLossDiff.push({
        x: valLossData[i].x,
        y: valLossData[i].y - valLossData[i - 1].y
      });
    }
    
    return {
      metadata: {
        total_iterations: iterations.length,
        start_time: this.metrics.timestamps[0],
        end_time: this.metrics.timestamps[this.metrics.timestamps.length - 1],
        generated_at: new Date().toISOString()
      },
      plots: {
        train_loss: {
          title: 'Training Loss',
          data: trainLossData,
          xlabel: 'Iteration',
          ylabel: 'Loss'
        },
        val_loss: {
          title: 'Validation Loss',
          data: valLossData,
          xlabel: 'Iteration',
          ylabel: 'Loss'
        },
        grad_norm: {
          title: 'Gradient Norm',
          data: gradNormData,
          xlabel: 'Iteration',
          ylabel: 'Gradient Norm'
        },
        learning_rate: {
          title: 'Learning Rate',
          data: lrData,
          xlabel: 'Iteration',
          ylabel: 'Learning Rate'
        },
        val_loss_diff: {
          title: 'Validation Loss Change',
          data: valLossDiff,
          xlabel: 'Iteration',
          ylabel: 'Loss Difference'
        }
      }
    };
  }
  
  /**
   * Generate text-based plots for terminal viewing
   */
  _generateTextPlots() {
    console.log('\n📊 Training Visualization');
    console.log('='.repeat(50));
    
    const plotData = this._generatePlotData();
    
    // Plot each metric
    for (const [plotName, plotInfo] of Object.entries(plotData.plots)) {
      if (plotInfo.data.length > 0) {
        this._textPlot(plotInfo.data, plotInfo.title, plotInfo.xlabel, plotInfo.ylabel);
      }
    }
  }
  
  /**
   * Create a simple text-based plot
   * @param {Array} data - Array of {x, y} points
   * @param {string} title - Plot title
   * @param {string} xlabel - X-axis label
   * @param {string} ylabel - Y-axis label
   * @param {number} width - Plot width in characters
   * @param {number} height - Plot height in characters
   */
  _textPlot(data, title, xlabel, ylabel, width = 60, height = 15) {
    if (data.length === 0) return;
    
    console.log(`\n${title}`);
    console.log('-'.repeat(title.length));
    
    // Find data ranges
    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    console.log(`${xlabel}: ${xMin} - ${xMax}`);
    console.log(`${ylabel}: ${yMin.toFixed(4)} - ${yMax.toFixed(4)}`);
    
    // Create plot grid
    const grid = Array(height).fill().map(() => Array(width).fill(' '));
    
    // Plot data points
    for (const point of data) {
      if (xMax === xMin || yMax === yMin) continue;
      
      const x = Math.floor(((point.x - xMin) / (xMax - xMin)) * (width - 1));
      const y = Math.floor(((yMax - point.y) / (yMax - yMin)) * (height - 1));
      
      if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[y][x] = '*';
      }
    }
    
    // Print grid with Y-axis labels
    for (let y = 0; y < height; y++) {
      const yValue = yMin + ((height - 1 - y) / (height - 1)) * (yMax - yMin);
      const yLabel = yValue.toFixed(3).padStart(8);
      console.log(`${yLabel} |${grid[y].join('')}|`);
    }
    
    // Print X-axis
    console.log(' '.repeat(9) + '+' + '-'.repeat(width) + '+');
    
    // Print X-axis labels
    const xLabelLine = ' '.repeat(10) + 
      xMin.toFixed(0).padEnd(Math.floor(width/2)) + 
      xMax.toFixed(0).padStart(Math.ceil(width/2));
    console.log(xLabelLine);
  }
  
  /**
   * Generate comprehensive analysis report
   */
  _generateAnalysisReport() {
    console.log('\n📈 Training Analysis Report');
    console.log('='.repeat(50));
    
    const analysis = this.analyzeTraining();
    
    console.log(`Training Duration: ${analysis.duration.toFixed(1)} minutes`);
    console.log(`Total Iterations: ${analysis.totalIterations}`);
    console.log(`Average Iteration Time: ${analysis.avgIterationTime.toFixed(2)}ms`);
    
    if (analysis.trainLoss.improvement !== null) {
      console.log(`\nTraining Loss:`);
      console.log(`  Initial: ${analysis.trainLoss.initial.toFixed(4)}`);
      console.log(`  Final: ${analysis.trainLoss.final.toFixed(4)}`);
      console.log(`  Improvement: ${analysis.trainLoss.improvement.toFixed(4)} (${analysis.trainLoss.improvementPercent.toFixed(1)}%)`);
      console.log(`  Trend: ${analysis.trainLoss.trend}`);
    }
    
    if (analysis.valLoss.improvement !== null) {
      console.log(`\nValidation Loss:`);
      console.log(`  Best: ${analysis.valLoss.best.toFixed(4)}`);
      console.log(`  Final: ${analysis.valLoss.final.toFixed(4)}`);
      console.log(`  Overfitting Risk: ${analysis.overfitting.risk}`);
      console.log(`  Gap: ${analysis.overfitting.gap.toFixed(4)}`);
    }
    
    if (analysis.gradients.avgNorm !== null) {
      console.log(`\nGradient Analysis:`);
      console.log(`  Average Norm: ${analysis.gradients.avgNorm.toFixed(4)}`);
      console.log(`  Max Norm: ${analysis.gradients.maxNorm.toFixed(4)}`);
      console.log(`  Health: ${analysis.gradients.health}`);
    }
    
    console.log(`\nRecommendations:`);
    analysis.recommendations.forEach(rec => console.log(`  • ${rec}`));
  }
  
  /**
   * Analyze training progress and provide insights
   * @returns {Object} - Comprehensive training analysis
   */
  analyzeTraining() {
    const trainLosses = this.metrics.train_loss.filter(l => l !== null);
    const valLosses = this.metrics.val_loss.filter(l => l !== null);
    const gradNorms = this.metrics.grad_norm.filter(g => g !== null);
    const timestamps = this.metrics.timestamps.filter(t => t !== null);
    
    // Calculate duration
    const duration = timestamps.length > 1 
      ? (new Date(timestamps[timestamps.length - 1]) - new Date(timestamps[0])) / (1000 * 60)
      : 0;
    
    const avgIterationTime = duration > 0 ? (duration * 60 * 1000) / this.metrics.iterations.length : 0;
    
    // Training loss analysis
    const trainLossAnalysis = {
      initial: trainLosses.length > 0 ? trainLosses[0] : null,
      final: trainLosses.length > 0 ? trainLosses[trainLosses.length - 1] : null,
      improvement: null,
      improvementPercent: null,
      trend: 'unknown'
    };
    
    if (trainLossAnalysis.initial !== null && trainLossAnalysis.final !== null) {
      trainLossAnalysis.improvement = trainLossAnalysis.initial - trainLossAnalysis.final;
      trainLossAnalysis.improvementPercent = (trainLossAnalysis.improvement / trainLossAnalysis.initial) * 100;
      
      if (trainLossAnalysis.improvement > 0.1) {
        trainLossAnalysis.trend = 'strongly decreasing';
      } else if (trainLossAnalysis.improvement > 0.01) {
        trainLossAnalysis.trend = 'decreasing';
      } else if (trainLossAnalysis.improvement > -0.01) {
        trainLossAnalysis.trend = 'stable';
      } else {
        trainLossAnalysis.trend = 'increasing';
      }
    }
    
    // Validation loss analysis
    const valLossAnalysis = {
      best: valLosses.length > 0 ? Math.min(...valLosses) : null,
      final: valLosses.length > 0 ? valLosses[valLosses.length - 1] : null,
      improvement: null
    };
    
    // Overfitting analysis
    const overfittingAnalysis = {
      risk: 'unknown',
      gap: 0
    };
    
    if (trainLossAnalysis.final !== null && valLossAnalysis.final !== null) {
      overfittingAnalysis.gap = valLossAnalysis.final - trainLossAnalysis.final;
      
      if (overfittingAnalysis.gap < 0.1) {
        overfittingAnalysis.risk = 'low';
      } else if (overfittingAnalysis.gap < 0.5) {
        overfittingAnalysis.risk = 'moderate';
      } else {
        overfittingAnalysis.risk = 'high';
      }
    }
    
    // Gradient analysis
    const gradientAnalysis = {
      avgNorm: gradNorms.length > 0 ? gradNorms.reduce((a, b) => a + b, 0) / gradNorms.length : null,
      maxNorm: gradNorms.length > 0 ? Math.max(...gradNorms) : null,
      health: 'unknown'
    };
    
    if (gradientAnalysis.avgNorm !== null) {
      if (gradientAnalysis.maxNorm > 10) {
        gradientAnalysis.health = 'exploding gradients detected';
      } else if (gradientAnalysis.avgNorm < 0.001) {
        gradientAnalysis.health = 'vanishing gradients possible';
      } else {
        gradientAnalysis.health = 'healthy';
      }
    }
    
    // Generate recommendations
    const recommendations = [];
    
    if (trainLossAnalysis.trend === 'increasing') {
      recommendations.push('Training loss is increasing - consider reducing learning rate');
    } else if (trainLossAnalysis.trend === 'stable') {
      recommendations.push('Training loss has plateaued - consider learning rate scheduling');
    }
    
    if (overfittingAnalysis.risk === 'high') {
      recommendations.push('High overfitting risk - consider regularization or early stopping');
    }
    
    if (gradientAnalysis.health === 'exploding gradients detected') {
      recommendations.push('Implement gradient clipping to prevent exploding gradients');
    } else if (gradientAnalysis.health === 'vanishing gradients possible') {
      recommendations.push('Consider residual connections or different initialization');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('Training appears healthy - continue monitoring');
    }
    
    return {
      duration,
      totalIterations: this.metrics.iterations.length,
      avgIterationTime,
      trainLoss: trainLossAnalysis,
      valLoss: valLossAnalysis,
      overfitting: overfittingAnalysis,
      gradients: gradientAnalysis,
      recommendations
    };
  }
  
  /**
   * Export metrics to CSV format
   * @param {string} csvPath - Path to save CSV file
   */
  exportCSV(csvPath = 'metrics.csv') {
    const headers = ['iteration', 'train_loss', 'val_loss', 'grad_norm', 'lr', 'timestamp'];
    const rows = [headers.join(',')];
    
    for (let i = 0; i < this.metrics.iterations.length; i++) {
      const row = [
        this.metrics.iterations[i],
        this.metrics.train_loss[i] || '',
        this.metrics.val_loss[i] || '',
        this.metrics.grad_norm[i] || '',
        this.metrics.lr[i] || '',
        this.metrics.timestamps[i] || ''
      ];
      rows.push(row.join(','));
    }
    
    fs.writeFileSync(csvPath, rows.join('\n'));
    return csvPath;
  }
  
  /**
   * Get summary statistics
   * @returns {Object} - Summary statistics
   */
  getSummary() {
    const trainLosses = this.metrics.train_loss.filter(l => l !== null);
    const valLosses = this.metrics.val_loss.filter(l => l !== null);
    
    return {
      totalIterations: this.metrics.iterations.length,
      trainLossCount: trainLosses.length,
      valLossCount: valLosses.length,
      finalTrainLoss: trainLosses.length > 0 ? trainLosses[trainLosses.length - 1] : null,
      finalValLoss: valLosses.length > 0 ? valLosses[valLosses.length - 1] : null,
      bestValLoss: valLosses.length > 0 ? Math.min(...valLosses) : null
    };
  }
}

/**
 * Simple plotting utility for quick visualizations
 * @param {Array} data - Array of {x, y} points
 * @param {string} title - Plot title
 * @param {Object} options - Plot options
 */
export function quickPlot(data, title = 'Plot', options = {}) {
  const {
    width = 60,
    height = 15,
    xlabel = 'X',
    ylabel = 'Y'
  } = options;
  
  const logger = new MetricsLogger();
  logger._textPlot(data, title, xlabel, ylabel, width, height);
}

/**
 * Create a simple histogram in text format
 * @param {Array} values - Array of values
 * @param {string} title - Histogram title
 * @param {number} bins - Number of bins
 */
export function textHistogram(values, title = 'Histogram', bins = 20) {
  if (values.length === 0) return;
  
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins;
  
  // Create bins
  const binCounts = new Array(bins).fill(0);
  const binLabels = [];
  
  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binWidth;
    const binEnd = min + (i + 1) * binWidth;
    binLabels.push(`${binStart.toFixed(2)}-${binEnd.toFixed(2)}`);
  }
  
  // Count values in each bin
  for (const value of values) {
    const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
    binCounts[binIndex]++;
  }
  
  // Find max count for scaling
  const maxCount = Math.max(...binCounts);
  const scale = 50 / maxCount; // Scale to 50 characters max
  
  console.log(`\n${title}`);
  console.log('='.repeat(title.length));
  console.log(`Values: ${values.length}, Range: ${min.toFixed(3)} - ${max.toFixed(3)}`);
  console.log('');
  
  for (let i = 0; i < bins; i++) {
    const barLength = Math.round(binCounts[i] * scale);
    const bar = '█'.repeat(barLength);
    const count = binCounts[i].toString().padStart(4);
    console.log(`${binLabels[i].padEnd(15)} |${bar} ${count}`);
  }
}