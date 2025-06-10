/**
 * Real-time Training Visualization Utilities
 * Provides live charts and visualizations for training progress
 */

/**
 * Real-time training visualizer with live charts
 */
export class TrainingVisualizer {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container with id '${containerId}' not found`);
    }
    
    this.options = {
      width: options.width || 800,
      height: options.height || 400,
      maxPoints: options.maxPoints || 1000,
      updateInterval: options.updateInterval || 100,
      showGrid: options.showGrid !== false,
      showLegend: options.showLegend !== false,
      colors: options.colors || {
        loss: '#e74c3c',
        valLoss: '#3498db',
        lr: '#f39c12',
        gradNorm: '#9b59b6'
      },
      ...options
    };
    
    this.data = {
      loss: [],
      valLoss: [],
      lr: [],
      gradNorm: [],
      steps: []
    };
    
    this.charts = {};
    this.isRunning = false;
    
    this.initializeCharts();
  }
  
  /**
   * Initialize chart containers and canvases
   */
  initializeCharts() {
    this.container.innerHTML = `
      <div class="training-visualizer">
        <div class="chart-container">
          <h3>Training Loss</h3>
          <canvas id="loss-chart" width="${this.options.width}" height="${this.options.height}"></canvas>
        </div>
        <div class="chart-container">
          <h3>Learning Rate</h3>
          <canvas id="lr-chart" width="${this.options.width}" height="${this.options.height / 2}"></canvas>
        </div>
        <div class="chart-container">
          <h3>Gradient Norm</h3>
          <canvas id="grad-chart" width="${this.options.width}" height="${this.options.height / 2}"></canvas>
        </div>
      </div>
    `;
    
    // Add CSS styles
    const style = document.createElement('style');
    style.textContent = `
      .training-visualizer {
        font-family: Arial, sans-serif;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      .chart-container {
        margin: 20px 0;
        background: white;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      }
      .chart-container h3 {
        margin: 0 0 10px 0;
        color: #2c3e50;
        font-size: 16px;
        font-weight: 600;
      }
      .chart-container canvas {
        border: 1px solid #e1e8ed;
        border-radius: 4px;
      }
    `;
    document.head.appendChild(style);
    
    // Initialize chart contexts
    this.charts.loss = document.getElementById('loss-chart').getContext('2d');
    this.charts.lr = document.getElementById('lr-chart').getContext('2d');
    this.charts.grad = document.getElementById('grad-chart').getContext('2d');
    
    // Set up initial charts
    this.drawChart('loss', 'Training Loss', ['loss', 'valLoss']);
    this.drawChart('lr', 'Learning Rate', ['lr']);
    this.drawChart('grad', 'Gradient Norm', ['gradNorm']);
  }
  
  /**
   * Add a data point to the visualization
   */
  addDataPoint(step, metrics) {
    this.data.steps.push(step);
    
    // Add metrics
    this.data.loss.push(metrics.loss || null);
    this.data.valLoss.push(metrics.valLoss || null);
    this.data.lr.push(metrics.lr || null);
    this.data.gradNorm.push(metrics.gradNorm || null);
    
    // Limit data points to prevent memory issues
    if (this.data.steps.length > this.options.maxPoints) {
      const excess = this.data.steps.length - this.options.maxPoints;
      this.data.steps.splice(0, excess);
      this.data.loss.splice(0, excess);
      this.data.valLoss.splice(0, excess);
      this.data.lr.splice(0, excess);
      this.data.gradNorm.splice(0, excess);
    }
    
    // Update charts if running
    if (this.isRunning) {
      this.updateCharts();
    }
  }
  
  /**
   * Start real-time updates
   */
  start() {
    this.isRunning = true;
    this.updateLoop();
  }
  
  /**
   * Stop real-time updates
   */
  stop() {
    this.isRunning = false;
  }
  
  /**
   * Update loop for real-time rendering
   */
  updateLoop() {
    if (!this.isRunning) return;
    
    this.updateCharts();
    setTimeout(() => this.updateLoop(), this.options.updateInterval);
  }
  
  /**
   * Update all charts
   */
  updateCharts() {
    this.drawChart('loss', 'Training Loss', ['loss', 'valLoss']);
    this.drawChart('lr', 'Learning Rate', ['lr']);
    this.drawChart('grad', 'Gradient Norm', ['gradNorm']);
  }
  
  /**
   * Draw a chart with given metrics
   */
  drawChart(chartId, title, metrics) {
    const ctx = this.charts[chartId];
    const canvas = ctx.canvas;
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Set up drawing parameters
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    
    // Draw background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid if enabled
    if (this.options.showGrid) {
      this.drawGrid(ctx, padding, chartWidth, chartHeight);
    }
    
    // Draw axes
    this.drawAxes(ctx, padding, chartWidth, chartHeight);
    
    // Draw data lines
    for (const metric of metrics) {
      if (this.data[metric].length > 0) {
        this.drawLine(ctx, metric, padding, chartWidth, chartHeight);
      }
    }
    
    // Draw legend if enabled
    if (this.options.showLegend && metrics.length > 1) {
      this.drawLegend(ctx, metrics, width, height);
    }
    
    // Draw title
    ctx.fillStyle = '#2c3e50';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 20);
  }
  
  /**
   * Draw grid lines
   */
  drawGrid(ctx, padding, chartWidth, chartHeight) {
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i * chartWidth) / 10;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, padding + chartHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i * chartHeight) / 10;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }
  }
  
  /**
   * Draw chart axes
   */
  drawAxes(ctx, padding, chartWidth, chartHeight) {
    ctx.strokeStyle = '#34495e';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding + chartHeight);
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + chartHeight);
    ctx.stroke();
    
    // Add axis labels
    ctx.fillStyle = '#7f8c8d';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    // X-axis label
    ctx.fillText('Training Steps', padding + chartWidth / 2, padding + chartHeight + 40);
    
    // Y-axis label (rotated)
    ctx.save();
    ctx.translate(20, padding + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Value', 0, 0);
    ctx.restore();
  }
  
  /**
   * Draw a data line for a specific metric
   */
  drawLine(ctx, metric, padding, chartWidth, chartHeight) {
    const data = this.data[metric].filter(val => val !== null);
    const steps = this.data.steps.slice(-data.length);
    
    if (data.length < 2) return;
    
    // Calculate scales
    const minStep = Math.min(...steps);
    const maxStep = Math.max(...steps);
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    
    const stepRange = maxStep - minStep || 1;
    const valRange = maxVal - minVal || 1;
    
    // Set line style
    ctx.strokeStyle = this.options.colors[metric] || '#3498db';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Draw line
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = padding + ((steps[i] - minStep) / stepRange) * chartWidth;
      const y = padding + chartHeight - ((data[i] - minVal) / valRange) * chartHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    
    // Draw points
    ctx.fillStyle = this.options.colors[metric] || '#3498db';
    for (let i = 0; i < data.length; i++) {
      const x = padding + ((steps[i] - minStep) / stepRange) * chartWidth;
      const y = padding + chartHeight - ((data[i] - minVal) / valRange) * chartHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    // Draw scale labels
    this.drawScaleLabels(ctx, minStep, maxStep, minVal, maxVal, padding, chartWidth, chartHeight);
  }
  
  /**
   * Draw scale labels on axes
   */
  drawScaleLabels(ctx, minStep, maxStep, minVal, maxVal, padding, chartWidth, chartHeight) {
    ctx.fillStyle = '#7f8c8d';
    ctx.font = '10px Arial';
    
    // X-axis labels
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const step = minStep + (i * (maxStep - minStep)) / 5;
      const x = padding + (i * chartWidth) / 5;
      ctx.fillText(Math.round(step).toString(), x, padding + chartHeight + 15);
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const val = minVal + (i * (maxVal - minVal)) / 5;
      const y = padding + chartHeight - (i * chartHeight) / 5;
      ctx.fillText(val.toFixed(3), padding - 10, y + 3);
    }
  }
  
  /**
   * Draw legend
   */
  drawLegend(ctx, metrics, width, height) {
    const legendX = width - 150;
    const legendY = 40;
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(legendX - 10, legendY - 10, 140, metrics.length * 20 + 20);
    
    ctx.strokeStyle = '#ddd';
    ctx.strokeRect(legendX - 10, legendY - 10, 140, metrics.length * 20 + 20);
    
    for (let i = 0; i < metrics.length; i++) {
      const metric = metrics[i];
      const y = legendY + i * 20;
      
      // Draw color indicator
      ctx.fillStyle = this.options.colors[metric] || '#3498db';
      ctx.fillRect(legendX, y - 5, 15, 10);
      
      // Draw label
      ctx.fillStyle = '#2c3e50';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(metric, legendX + 20, y + 3);
    }
  }
  
  /**
   * Export chart as image
   */
  exportChart(chartId, filename) {
    const canvas = this.charts[chartId].canvas;
    const link = document.createElement('a');
    link.download = filename || `${chartId}-chart.png`;
    link.href = canvas.toDataURL();
    link.click();
  }
  
  /**
   * Clear all data
   */
  clear() {
    this.data = {
      loss: [],
      valLoss: [],
      lr: [],
      gradNorm: [],
      steps: []
    };
    this.updateCharts();
  }
  
  /**
   * Get current statistics
   */
  getStats() {
    const getMetricStats = (data) => {
      const validData = data.filter(val => val !== null && !isNaN(val));
      if (validData.length === 0) return null;
      
      return {
        current: validData[validData.length - 1],
        min: Math.min(...validData),
        max: Math.max(...validData),
        avg: validData.reduce((sum, val) => sum + val, 0) / validData.length,
        count: validData.length
      };
    };
    
    return {
      loss: getMetricStats(this.data.loss),
      valLoss: getMetricStats(this.data.valLoss),
      lr: getMetricStats(this.data.lr),
      gradNorm: getMetricStats(this.data.gradNorm),
      totalSteps: this.data.steps.length
    };
  }
}

/**
 * Simple metrics dashboard for training monitoring
 */
export class MetricsDashboard {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container with id '${containerId}' not found`);
    }
    
    this.metrics = {};
    this.initializeDashboard();
  }
  
  initializeDashboard() {
    this.container.innerHTML = `
      <div class="metrics-dashboard">
        <div class="metric-card" id="loss-card">
          <h4>Training Loss</h4>
          <div class="metric-value">--</div>
          <div class="metric-trend"></div>
        </div>
        <div class="metric-card" id="val-loss-card">
          <h4>Validation Loss</h4>
          <div class="metric-value">--</div>
          <div class="metric-trend"></div>
        </div>
        <div class="metric-card" id="lr-card">
          <h4>Learning Rate</h4>
          <div class="metric-value">--</div>
          <div class="metric-trend"></div>
        </div>
        <div class="metric-card" id="grad-card">
          <h4>Gradient Norm</h4>
          <div class="metric-value">--</div>
          <div class="metric-trend"></div>
        </div>
      </div>
    `;
    
    // Add CSS styles
    const style = document.createElement('style');
    style.textContent = `
      .metrics-dashboard {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        padding: 20px;
      }
      .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #3498db;
      }
      .metric-card h4 {
        margin: 0 0 10px 0;
        color: #2c3e50;
        font-size: 14px;
        font-weight: 600;
      }
      .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0;
      }
      .metric-trend {
        font-size: 12px;
        color: #7f8c8d;
      }
      .trend-up { color: #e74c3c; }
      .trend-down { color: #27ae60; }
      .trend-stable { color: #f39c12; }
    `;
    document.head.appendChild(style);
  }
  
  updateMetric(name, value, trend = null) {
    this.metrics[name] = value;
    
    const cardId = `${name.replace(/([A-Z])/g, '-$1').toLowerCase()}-card`;
    const card = document.getElementById(cardId);
    
    if (card) {
      const valueElement = card.querySelector('.metric-value');
      const trendElement = card.querySelector('.metric-trend');
      
      if (valueElement) {
        if (typeof value === 'number') {
          valueElement.textContent = value.toFixed(4);
        } else {
          valueElement.textContent = value || '--';
        }
      }
      
      if (trendElement && trend !== null) {
        trendElement.textContent = trend > 0 ? '↗ Increasing' : trend < 0 ? '↘ Decreasing' : '→ Stable';
        trendElement.className = `metric-trend ${trend > 0 ? 'trend-up' : trend < 0 ? 'trend-down' : 'trend-stable'}`;
      }
    }
  }
  
  getMetrics() {
    return { ...this.metrics };
  }
}