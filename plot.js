#!/usr/bin/env node

/**
 * Plotting Script for numpyGPT JavaScript Implementation
 * JavaScript equivalent of plot.py
 * 
 * This script:
 * 1. Loads training metrics from JSON file
 * 2. Displays training statistics
 * 3. Creates simple text-based plots
 * 4. Optionally exports data for external plotting tools
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Simple text-based plotting function
 * @param {Array} data - Array of {x, y} points
 * @param {string} title - Plot title
 * @param {number} width - Plot width in characters
 * @param {number} height - Plot height in characters
 */
function textPlot(data, title, width = 60, height = 20) {
  if (data.length === 0) {
    console.log(`${title}: No data to plot`);
    return;
  }
  
  // Find data ranges
  const xValues = data.map(d => d.x);
  const yValues = data.map(d => d.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  console.log(`\n${title}`);
  console.log('='.repeat(title.length));
  console.log(`X range: ${xMin.toFixed(2)} - ${xMax.toFixed(2)}`);
  console.log(`Y range: ${yMin.toFixed(4)} - ${yMax.toFixed(4)}`);
  
  // Create plot grid
  const grid = Array(height).fill().map(() => Array(width).fill(' '));
  
  // Plot data points
  for (const point of data) {
    const x = Math.floor(((point.x - xMin) / (xMax - xMin)) * (width - 1));
    const y = Math.floor(((yMax - point.y) / (yMax - yMin)) * (height - 1));
    
    if (x >= 0 && x < width && y >= 0 && y < height) {
      grid[y][x] = '*';
    }
  }
  
  // Print grid with Y-axis labels
  for (let y = 0; y < height; y++) {
    const yValue = yMin + ((height - 1 - y) / (height - 1)) * (yMax - yMin);
    const yLabel = yValue.toFixed(3).padStart(7);
    console.log(`${yLabel} |${grid[y].join('')}|`);
  }
  
  // Print X-axis
  const xAxisLine = ' '.repeat(8) + '+' + '-'.repeat(width) + '+';
  console.log(xAxisLine);
  
  // Print X-axis labels
  const xLabelLine = ' '.repeat(9) + 
    xMin.toFixed(0).padEnd(Math.floor(width/2)) + 
    xMax.toFixed(0).padStart(Math.ceil(width/2));
  console.log(xLabelLine);
}

/**
 * Plot training metrics
 * @param {string} outDir - Output directory containing metrics.json
 * @param {string} outputFile - Optional output file for data export
 */
function plotMetrics(outDir = 'out/char', outputFile = null) {
  const metricsFile = path.join(outDir, 'metrics.json');
  
  if (!fs.existsSync(metricsFile)) {
    console.error(`Metrics file not found: ${metricsFile}`);
    process.exit(1);
  }
  
  console.log(`Loading metrics from ${metricsFile}`);
  const metrics = JSON.parse(fs.readFileSync(metricsFile, 'utf-8'));
  
  if (metrics.length === 0) {
    console.log('No metrics found');
    return;
  }
  
  console.log(`\nTraining Metrics Summary`);
  console.log('========================');
  console.log(`Total iterations: ${metrics.length}`);
  
  // Extract different metric types
  const trainLossData = [];
  const valLossData = [];
  const lrData = [];
  const gradNormData = [];
  
  for (const entry of metrics) {
    const iter = entry.iter;
    
    if (entry.train_loss !== undefined && entry.train_loss !== null) {
      trainLossData.push({ x: iter, y: entry.train_loss });
    }
    
    if (entry.val_loss !== undefined && entry.val_loss !== null) {
      valLossData.push({ x: iter, y: entry.val_loss });
    }
    
    if (entry.lr !== undefined && entry.lr !== null) {
      lrData.push({ x: iter, y: entry.lr });
    }
    
    if (entry.grad_norm !== undefined && entry.grad_norm !== null) {
      gradNormData.push({ x: iter, y: entry.grad_norm });
    }
  }
  
  // Display statistics
  if (trainLossData.length > 0) {
    const latestTrainLoss = trainLossData[trainLossData.length - 1].y;
    const initialTrainLoss = trainLossData[0].y;
    console.log(`Latest train loss: ${latestTrainLoss.toFixed(4)}`);
    console.log(`Initial train loss: ${initialTrainLoss.toFixed(4)}`);
    console.log(`Train loss improvement: ${(initialTrainLoss - latestTrainLoss).toFixed(4)}`);
  }
  
  if (valLossData.length > 0) {
    const valLosses = valLossData.map(d => d.y);
    const bestValLoss = Math.min(...valLosses);
    const latestValLoss = valLossData[valLossData.length - 1].y;
    console.log(`Best validation loss: ${bestValLoss.toFixed(4)}`);
    console.log(`Latest validation loss: ${latestValLoss.toFixed(4)}`);
  }
  
  if (lrData.length > 0) {
    const latestLr = lrData[lrData.length - 1].y;
    console.log(`Latest learning rate: ${latestLr.toExponential(2)}`);
  }
  
  if (gradNormData.length > 0) {
    const latestGradNorm = gradNormData[gradNormData.length - 1].y;
    const avgGradNorm = gradNormData.reduce((sum, d) => sum + d.y, 0) / gradNormData.length;
    console.log(`Latest gradient norm: ${latestGradNorm.toFixed(4)}`);
    console.log(`Average gradient norm: ${avgGradNorm.toFixed(4)}`);
  }
  
  // Create text plots
  if (trainLossData.length > 0) {
    textPlot(trainLossData, 'Training Loss');
  }
  
  if (valLossData.length > 0) {
    textPlot(valLossData, 'Validation Loss');
  }
  
  if (lrData.length > 0) {
    textPlot(lrData, 'Learning Rate');
  }
  
  if (gradNormData.length > 0) {
    textPlot(gradNormData, 'Gradient Norm');
  }
  
  // Export data for external plotting if requested
  if (outputFile) {
    const exportData = {
      train_loss: trainLossData,
      val_loss: valLossData,
      learning_rate: lrData,
      grad_norm: gradNormData,
      metadata: {
        total_iterations: metrics.length,
        created_at: new Date().toISOString(),
        source_file: metricsFile
      }
    };
    
    fs.writeFileSync(outputFile, JSON.stringify(exportData, null, 2));
    console.log(`\nData exported to ${outputFile}`);
    console.log('You can use this data with external plotting tools like Python matplotlib, R, or online tools.');
  }
  
  // Generate CSV for easy import into spreadsheet tools
  const csvFile = path.join(outDir, 'metrics.csv');
  const csvLines = ['iteration,train_loss,val_loss,learning_rate,grad_norm'];
  
  for (const entry of metrics) {
    const line = [
      entry.iter,
      entry.train_loss || '',
      entry.val_loss || '',
      entry.lr || '',
      entry.grad_norm || ''
    ].join(',');
    csvLines.push(line);
  }
  
  fs.writeFileSync(csvFile, csvLines.join('\n'));
  console.log(`\nCSV data saved to ${csvFile}`);
  
  // Provide plotting suggestions
  console.log(`\nPlotting Suggestions:`);
  console.log(`1. Import ${csvFile} into Excel, Google Sheets, or similar`);
  console.log(`2. Use Python: pandas.read_csv('${csvFile}').plot()`);
  console.log(`3. Use R: plot(read.csv('${csvFile}'))`);
  console.log(`4. Use online tools like Plot.ly or Chart.js`);
}

/**
 * Parse command line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    outDir: 'out/char',
    outputFile: null
  };
  
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key === 'out_dir') options.outDir = value;
    else if (key === 'output_file') options.outputFile = value;
    else if (key === 'help') {
      console.log(`
Usage: node plot.js [options]

Options:
  --out_dir <path>          Output directory containing metrics.json (default: out/char)
  --output_file <path>      Export data to JSON file for external plotting (optional)
  --help                    Show this help message

Examples:
  node plot.js --out_dir out/char
  node plot.js --out_dir out/bpe --output_file plots/data.json
  node plot.js --output_file training_data.json
      `);
      process.exit(0);
    }
  }
  
  return options;
}

/**
 * Analyze training progress and provide insights
 * @param {Array} metrics - Training metrics array
 */
function analyzeTraining(metrics) {
  if (metrics.length < 2) {
    console.log('\nNot enough data for analysis');
    return;
  }
  
  console.log(`\nTraining Analysis:`);
  console.log('==================');
  
  // Analyze training loss trend
  const trainLossEntries = metrics.filter(m => m.train_loss !== undefined && m.train_loss !== null);
  if (trainLossEntries.length >= 2) {
    const recent = trainLossEntries.slice(-Math.min(10, trainLossEntries.length));
    const recentLosses = recent.map(m => m.train_loss);
    const avgRecentLoss = recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length;
    
    const early = trainLossEntries.slice(0, Math.min(10, trainLossEntries.length));
    const earlyLosses = early.map(m => m.train_loss);
    const avgEarlyLoss = earlyLosses.reduce((a, b) => a + b, 0) / earlyLosses.length;
    
    const improvement = avgEarlyLoss - avgRecentLoss;
    const improvementPercent = (improvement / avgEarlyLoss) * 100;
    
    console.log(`Training loss improvement: ${improvement.toFixed(4)} (${improvementPercent.toFixed(1)}%)`);
    
    if (improvement > 0) {
      console.log('✓ Model is learning (loss decreasing)');
    } else {
      console.log('⚠ Model may not be learning effectively (loss not decreasing)');
    }
  }
  
  // Analyze validation loss vs training loss
  const valLossEntries = metrics.filter(m => m.val_loss !== undefined && m.val_loss !== null);
  if (valLossEntries.length >= 2 && trainLossEntries.length >= 2) {
    const latestVal = valLossEntries[valLossEntries.length - 1].val_loss;
    const latestTrain = trainLossEntries[trainLossEntries.length - 1].train_loss;
    const gap = latestVal - latestTrain;
    
    console.log(`Train/Val gap: ${gap.toFixed(4)}`);
    
    if (gap < 0.1) {
      console.log('✓ Good generalization (small train/val gap)');
    } else if (gap < 0.5) {
      console.log('⚠ Some overfitting (moderate train/val gap)');
    } else {
      console.log('⚠ Possible overfitting (large train/val gap)');
    }
  }
  
  // Analyze gradient norms
  const gradNormEntries = metrics.filter(m => m.grad_norm !== undefined && m.grad_norm !== null);
  if (gradNormEntries.length >= 2) {
    const gradNorms = gradNormEntries.map(m => m.grad_norm);
    const avgGradNorm = gradNorms.reduce((a, b) => a + b, 0) / gradNorms.length;
    const maxGradNorm = Math.max(...gradNorms);
    
    console.log(`Average gradient norm: ${avgGradNorm.toFixed(4)}`);
    console.log(`Maximum gradient norm: ${maxGradNorm.toFixed(4)}`);
    
    if (maxGradNorm > 10) {
      console.log('⚠ High gradient norms detected - consider gradient clipping');
    } else if (avgGradNorm < 0.001) {
      console.log('⚠ Very small gradients - learning may be slow');
    } else {
      console.log('✓ Gradient norms look healthy');
    }
  }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const options = parseArgs();
    
    console.log('Training Metrics Visualization');
    console.log('==============================');
    console.log(`Output directory: ${options.outDir}`);
    if (options.outputFile) {
      console.log(`Export file: ${options.outputFile}`);
    }
    console.log('');
    
    plotMetrics(options.outDir, options.outputFile);
    
    // Load metrics for analysis
    const metricsFile = path.join(options.outDir, 'metrics.json');
    if (fs.existsSync(metricsFile)) {
      const metrics = JSON.parse(fs.readFileSync(metricsFile, 'utf-8'));
      analyzeTraining(metrics);
    }
    
  } catch (error) {
    console.error('Plotting failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Export for use as module
export { plotMetrics, textPlot, analyzeTraining };