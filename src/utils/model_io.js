/**
 * Model Import/Export Utilities
 * Provides compatibility with Python version for model serialization
 */

import { Matrix } from 'ml-matrix';
import { GPT } from '../models/GPT.js';
import { CharTokenizer } from '../tokenizer/char_level.js';
import { WordTokenizer } from '../tokenizer/word_level.js';
import { BPETokenizer } from '../tokenizer/bpe.js';
import fs from 'fs';
import path from 'path';

/**
 * Model serialization and deserialization utilities
 */
export class ModelIO {
  /**
   * Export model to Python-compatible format
   * @param {GPT} model - Model to export
   * @param {Object} tokenizer - Tokenizer to export
   * @param {string} filepath - Path to save the model
   * @param {Object} metadata - Additional metadata
   */
  static exportModel(model, tokenizer, filepath, metadata = {}) {
    const modelData = {
      // Model metadata
      metadata: {
        framework: 'numpyGPT-js',
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        architecture: 'GPT',
        ...metadata
      },
      
      // Model configuration
      config: {
        vocabSize: model.vocabSize,
        maxLen: model.maxLen,
        dModel: model.dModel,
        nHeads: model.blocks[0]?.nHeads || 0,
        nLayers: model.blocks.length,
        dFF: model.blocks[0]?.dFF || 0
      },
      
      // Model parameters (converted to Python-compatible format)
      parameters: ModelIO._serializeParameters(model),
      
      // Tokenizer data
      tokenizer: ModelIO._serializeTokenizer(tokenizer),
      
      // Training state (if available)
      training_state: {
        step: metadata.step || 0,
        loss: metadata.loss || null,
        lr: metadata.lr || null
      }
    };
    
    // Save as JSON (Python can load with json.load)
    const jsonData = JSON.stringify(modelData, null, 2);
    fs.writeFileSync(filepath, jsonData);
    
    console.log(`Model exported to ${filepath}`);
    return modelData;
  }
  
  /**
   * Import model from Python-compatible format
   * @param {string} filepath - Path to the model file
   * @returns {Object} - {model, tokenizer, metadata}
   */
  static importModel(filepath) {
    if (!fs.existsSync(filepath)) {
      throw new Error(`Model file not found: ${filepath}`);
    }
    
    const jsonData = fs.readFileSync(filepath, 'utf-8');
    const modelData = JSON.parse(jsonData);
    
    // Validate format
    if (!modelData.config || !modelData.parameters) {
      throw new Error('Invalid model format: missing config or parameters');
    }
    
    // Create model
    const config = modelData.config;
    const model = new GPT(
      config.vocabSize,
      config.maxLen,
      config.dModel,
      config.nHeads,
      config.nLayers,
      config.dFF
    );
    
    // Load parameters
    ModelIO._deserializeParameters(model, modelData.parameters);
    
    // Create tokenizer
    const tokenizer = ModelIO._deserializeTokenizer(modelData.tokenizer);
    
    console.log(`Model imported from ${filepath}`);
    return {
      model,
      tokenizer,
      metadata: modelData.metadata,
      trainingState: modelData.training_state
    };
  }
  
  /**
   * Export model in NumPy-compatible format (.npz)
   * Note: This creates a JSON representation that can be converted to .npz in Python
   */
  static exportToNumPy(model, tokenizer, filepath) {
    const npzData = {
      // Model parameters as nested arrays (NumPy-compatible)
      'model/tok_emb/weight': ModelIO._matrixToArray(model.tokEmb.weight),
      'model/pos_emb/pe': ModelIO._matrixToArray(model.posEmb.pe),
      'model/ln_f/weight': ModelIO._matrixToArray(model.lnF.weight),
      'model/ln_f/bias': ModelIO._matrixToArray(model.lnF.bias),
      'model/lm_head/weight': ModelIO._matrixToArray(model.lmHead.W),
      'model/lm_head/bias': ModelIO._matrixToArray(model.lmHead.b),
    };
    
    // Add transformer block parameters
    for (let i = 0; i < model.blocks.length; i++) {
      const block = model.blocks[i];
      const prefix = `model/blocks/${i}`;
      
      // Attention parameters
      npzData[`${prefix}/attn/W_q/weight`] = ModelIO._matrixToArray(block.attn.WQ.W);
      npzData[`${prefix}/attn/W_q/bias`] = ModelIO._matrixToArray(block.attn.WQ.b);
      npzData[`${prefix}/attn/W_k/weight`] = ModelIO._matrixToArray(block.attn.WK.W);
      npzData[`${prefix}/attn/W_k/bias`] = ModelIO._matrixToArray(block.attn.WK.b);
      npzData[`${prefix}/attn/W_v/weight`] = ModelIO._matrixToArray(block.attn.WV.W);
      npzData[`${prefix}/attn/W_v/bias`] = ModelIO._matrixToArray(block.attn.WV.b);
      npzData[`${prefix}/attn/W_o/weight`] = ModelIO._matrixToArray(block.attn.WO.W);
      npzData[`${prefix}/attn/W_o/bias`] = ModelIO._matrixToArray(block.attn.WO.b);
      
      // Feed-forward parameters
      npzData[`${prefix}/ff/linear1/weight`] = ModelIO._matrixToArray(block.ff.linear1.W);
      npzData[`${prefix}/ff/linear1/bias`] = ModelIO._matrixToArray(block.ff.linear1.b);
      npzData[`${prefix}/ff/linear2/weight`] = ModelIO._matrixToArray(block.ff.linear2.W);
      npzData[`${prefix}/ff/linear2/bias`] = ModelIO._matrixToArray(block.ff.linear2.b);
      
      // Layer norm parameters
      npzData[`${prefix}/ln1/weight`] = ModelIO._matrixToArray(block.ln1.weight);
      npzData[`${prefix}/ln1/bias`] = ModelIO._matrixToArray(block.ln1.bias);
      npzData[`${prefix}/ln2/weight`] = ModelIO._matrixToArray(block.ln2.weight);
      npzData[`${prefix}/ln2/bias`] = ModelIO._matrixToArray(block.ln2.bias);
    }
    
    // Add metadata
    npzData['config'] = {
      vocabSize: model.vocabSize,
      maxLen: model.maxLen,
      dModel: model.dModel,
      nHeads: model.blocks[0]?.nHeads || 0,
      nLayers: model.blocks.length,
      dFF: model.blocks[0]?.dFF || 0
    };
    
    // Add tokenizer data
    npzData['tokenizer'] = ModelIO._serializeTokenizer(tokenizer);
    
    // Save as JSON (can be converted to .npz in Python)
    const jsonData = JSON.stringify(npzData, null, 2);
    fs.writeFileSync(filepath.replace('.npz', '.json'), jsonData);
    
    // Also create a Python script to convert to .npz
    const pythonScript = ModelIO._generateNpzConversionScript(filepath);
    fs.writeFileSync(filepath.replace('.npz', '_convert.py'), pythonScript);
    
    console.log(`NumPy-compatible data exported to ${filepath.replace('.npz', '.json')}`);
    console.log(`Python conversion script saved to ${filepath.replace('.npz', '_convert.py')}`);
  }
  
  /**
   * Import model from NumPy format
   */
  static importFromNumPy(filepath) {
    // Load the JSON representation of NumPy data
    const jsonPath = filepath.replace('.npz', '.json');
    if (!fs.existsSync(jsonPath)) {
      throw new Error(`NumPy JSON file not found: ${jsonPath}`);
    }
    
    const npzData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    
    // Extract config
    const config = npzData.config;
    
    // Create model
    const model = new GPT(
      config.vocabSize,
      config.maxLen,
      config.dModel,
      config.nHeads,
      config.nLayers,
      config.dFF
    );
    
    // Load parameters from NumPy format
    ModelIO._loadFromNumPyData(model, npzData);
    
    // Create tokenizer
    const tokenizer = ModelIO._deserializeTokenizer(npzData.tokenizer);
    
    return { model, tokenizer, config };
  }
  
  /**
   * Serialize model parameters to JSON-compatible format
   */
  static _serializeParameters(model) {
    const params = model.params();
    const serialized = {};
    
    for (const [name, param] of Object.entries(params)) {
      if (param instanceof Matrix) {
        serialized[name] = {
          data: param.to2DArray(),
          shape: [param.rows, param.columns],
          dtype: 'float32'
        };
      }
    }
    
    return serialized;
  }
  
  /**
   * Deserialize parameters from JSON format
   */
  static _deserializeParameters(model, serializedParams) {
    const params = model.params();
    
    for (const [name, paramData] of Object.entries(serializedParams)) {
      if (params[name] && paramData.data) {
        const matrix = new Matrix(paramData.data);
        
        // Copy values to existing parameter matrix
        for (let i = 0; i < matrix.rows; i++) {
          for (let j = 0; j < matrix.columns; j++) {
            params[name].set(i, j, matrix.get(i, j));
          }
        }
      }
    }
  }
  
  /**
   * Serialize tokenizer to JSON format
   */
  static _serializeTokenizer(tokenizer) {
    if (!tokenizer) return null;
    
    const tokenizerData = {
      type: tokenizer.constructor.name,
      state: tokenizer.getState()
    };
    
    return tokenizerData;
  }
  
  /**
   * Deserialize tokenizer from JSON format
   */
  static _deserializeTokenizer(tokenizerData) {
    if (!tokenizerData) return null;
    
    let tokenizer;
    
    switch (tokenizerData.type) {
      case 'CharTokenizer':
        tokenizer = new CharTokenizer();
        break;
      case 'WordTokenizer':
        tokenizer = new WordTokenizer();
        break;
      case 'BPETokenizer':
        tokenizer = new BPETokenizer();
        break;
      default:
        throw new Error(`Unknown tokenizer type: ${tokenizerData.type}`);
    }
    
    if (tokenizerData.state) {
      tokenizer.loadState(tokenizerData.state);
    }
    
    return tokenizer;
  }
  
  /**
   * Convert Matrix to nested array (NumPy-compatible)
   */
  static _matrixToArray(matrix) {
    if (!matrix) return null;
    return matrix.to2DArray();
  }
  
  /**
   * Load parameters from NumPy data format
   */
  static _loadFromNumPyData(model, npzData) {
    // Load token embedding
    if (npzData['model/tok_emb/weight']) {
      const data = npzData['model/tok_emb/weight'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.tokEmb.weight);
    }
    
    // Load positional encoding
    if (npzData['model/pos_emb/pe']) {
      const data = npzData['model/pos_emb/pe'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.posEmb.pe);
    }
    
    // Load final layer norm
    if (npzData['model/ln_f/weight']) {
      const data = npzData['model/ln_f/weight'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.lnF.weight);
    }
    
    if (npzData['model/ln_f/bias']) {
      const data = npzData['model/ln_f/bias'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.lnF.bias);
    }
    
    // Load language modeling head
    if (npzData['model/lm_head/weight']) {
      const data = npzData['model/lm_head/weight'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.lmHead.W);
    }
    
    if (npzData['model/lm_head/bias']) {
      const data = npzData['model/lm_head/bias'];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, model.lmHead.b);
    }
    
    // Load transformer blocks
    for (let i = 0; i < model.blocks.length; i++) {
      const block = model.blocks[i];
      const prefix = `model/blocks/${i}`;
      
      // Load attention parameters
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_q/weight`, block.attn.WQ.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_q/bias`, block.attn.WQ.b);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_k/weight`, block.attn.WK.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_k/bias`, block.attn.WK.b);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_v/weight`, block.attn.WV.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_v/bias`, block.attn.WV.b);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_o/weight`, block.attn.WO.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/attn/W_o/bias`, block.attn.WO.b);
      
      // Load feed-forward parameters
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ff/linear1/weight`, block.ff.linear1.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ff/linear1/bias`, block.ff.linear1.b);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ff/linear2/weight`, block.ff.linear2.W);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ff/linear2/bias`, block.ff.linear2.b);
      
      // Load layer norm parameters
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ln1/weight`, block.ln1.weight);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ln1/bias`, block.ln1.bias);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ln2/weight`, block.ln2.weight);
      ModelIO._loadMatrixFromNpz(npzData, `${prefix}/ln2/bias`, block.ln2.bias);
    }
  }
  
  /**
   * Helper to load matrix from NPZ data
   */
  static _loadMatrixFromNpz(npzData, key, targetMatrix) {
    if (npzData[key]) {
      const data = npzData[key];
      const matrix = new Matrix(data);
      ModelIO._copyMatrix(matrix, targetMatrix);
    }
  }
  
  /**
   * Copy values from source matrix to target matrix
   */
  static _copyMatrix(source, target) {
    for (let i = 0; i < source.rows; i++) {
      for (let j = 0; j < source.columns; j++) {
        target.set(i, j, source.get(i, j));
      }
    }
  }
  
  /**
   * Generate Python script to convert JSON to .npz format
   */
  static _generateNpzConversionScript(npzPath) {
    const jsonPath = npzPath.replace('.npz', '.json');
    
    return `#!/usr/bin/env python3
"""
Convert numpyGPT JavaScript model to NumPy .npz format
Generated automatically by numpyGPT-js ModelIO
"""

import json
import numpy as np

def convert_to_npz(json_path, npz_path):
    """Convert JSON model data to NumPy .npz format"""
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to NumPy arrays
    npz_data = {}
    
    for key, value in data.items():
        if key in ['config', 'tokenizer']:
            # Keep metadata as-is
            npz_data[key] = value
        elif isinstance(value, list):
            # Convert nested lists to NumPy arrays
            npz_data[key] = np.array(value, dtype=np.float32)
        else:
            npz_data[key] = value
    
    # Save as .npz file
    np.savez_compressed(npz_path, **npz_data)
    print(f"Model converted to {npz_path}")

def load_npz_model(npz_path):
    """Load and inspect .npz model"""
    
    data = np.load(npz_path, allow_pickle=True)
    
    print("Model contents:")
    for key in data.files:
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
        else:
            print(f"  {key}: {type(data[key])}")
    
    return data

if __name__ == "__main__":
    json_path = "${jsonPath}"
    npz_path = "${npzPath}"
    
    # Convert JSON to NPZ
    convert_to_npz(json_path, npz_path)
    
    # Verify the conversion
    print("\\nVerifying conversion:")
    model_data = load_npz_model(npz_path)
    
    # Print config
    if 'config' in model_data:
        config = model_data['config'].item()
        print(f"\\nModel config: {config}")
    
    # Print tokenizer info
    if 'tokenizer' in model_data:
        tokenizer = model_data['tokenizer'].item()
        print(f"\\nTokenizer type: {tokenizer.get('type', 'unknown')}")
`;
  }
  
  /**
   * Create a checkpoint with full training state
   */
  static saveCheckpoint(model, optimizer, tokenizer, step, loss, filepath) {
    const checkpoint = {
      metadata: {
        framework: 'numpyGPT-js',
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        step: step,
        loss: loss
      },
      
      model_config: {
        vocabSize: model.vocabSize,
        maxLen: model.maxLen,
        dModel: model.dModel,
        nHeads: model.blocks[0]?.nHeads || 0,
        nLayers: model.blocks.length,
        dFF: model.blocks[0]?.dFF || 0
      },
      
      model_state: ModelIO._serializeParameters(model),
      
      optimizer_state: {
        type: optimizer.constructor.name,
        lr: optimizer.lr,
        step: optimizer.t || 0,
        // Note: Full optimizer state serialization would require more work
        // For now, just save the essential parameters
      },
      
      tokenizer_state: ModelIO._serializeTokenizer(tokenizer)
    };
    
    fs.writeFileSync(filepath, JSON.stringify(checkpoint, null, 2));
    console.log(`Checkpoint saved to ${filepath}`);
    
    return checkpoint;
  }
  
  /**
   * Load a checkpoint
   */
  static loadCheckpoint(filepath) {
    if (!fs.existsSync(filepath)) {
      throw new Error(`Checkpoint file not found: ${filepath}`);
    }
    
    const checkpoint = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
    
    // Create model
    const config = checkpoint.model_config;
    const model = new GPT(
      config.vocabSize,
      config.maxLen,
      config.dModel,
      config.nHeads,
      config.nLayers,
      config.dFF
    );
    
    // Load model state
    ModelIO._deserializeParameters(model, checkpoint.model_state);
    
    // Create tokenizer
    const tokenizer = ModelIO._deserializeTokenizer(checkpoint.tokenizer_state);
    
    console.log(`Checkpoint loaded from ${filepath}`);
    
    return {
      model,
      tokenizer,
      metadata: checkpoint.metadata,
      optimizerState: checkpoint.optimizer_state
    };
  }
}

/**
 * Utility functions for model conversion between frameworks
 */
export class ModelConverter {
  /**
   * Convert JavaScript model to PyTorch-compatible format
   */
  static toPyTorch(model, tokenizer, filepath) {
    const pytorchData = {
      model_state_dict: {},
      config: {
        vocab_size: model.vocabSize,
        max_position_embeddings: model.maxLen,
        hidden_size: model.dModel,
        num_attention_heads: model.blocks[0]?.nHeads || 0,
        num_hidden_layers: model.blocks.length,
        intermediate_size: model.blocks[0]?.dFF || 0
      },
      tokenizer_config: ModelIO._serializeTokenizer(tokenizer)
    };
    
    // Convert parameter names to PyTorch convention
    const params = model.params();
    for (const [jsName, param] of Object.entries(params)) {
      const pytorchName = ModelConverter._jsToPyTorchName(jsName);
      pytorchData.model_state_dict[pytorchName] = param.to2DArray();
    }
    
    fs.writeFileSync(filepath, JSON.stringify(pytorchData, null, 2));
    console.log(`PyTorch-compatible model saved to ${filepath}`);
    
    return pytorchData;
  }
  
  /**
   * Convert parameter names from JavaScript to PyTorch convention
   */
  static _jsToPyTorchName(jsName) {
    // Convert naming convention
    // e.g., "tok_emb.weight" -> "embeddings.word_embeddings.weight"
    const nameMap = {
      'tok_emb.weight': 'embeddings.word_embeddings.weight',
      'pos_emb.pe': 'embeddings.position_embeddings.weight',
      'ln_f.weight': 'layernorm.weight',
      'ln_f.bias': 'layernorm.bias',
      'lm_head.W': 'lm_head.weight',
      'lm_head.b': 'lm_head.bias'
    };
    
    // Handle transformer blocks
    if (jsName.includes('blocks.')) {
      const parts = jsName.split('.');
      const blockNum = parts[1];
      const component = parts.slice(2).join('.');
      
      const componentMap = {
        'attn.W_q.W': 'attention.self.query.weight',
        'attn.W_q.b': 'attention.self.query.bias',
        'attn.W_k.W': 'attention.self.key.weight',
        'attn.W_k.b': 'attention.self.key.bias',
        'attn.W_v.W': 'attention.self.value.weight',
        'attn.W_v.b': 'attention.self.value.bias',
        'attn.W_o.W': 'attention.output.dense.weight',
        'attn.W_o.b': 'attention.output.dense.bias',
        'ff.linear1.W': 'intermediate.dense.weight',
        'ff.linear1.b': 'intermediate.dense.bias',
        'ff.linear2.W': 'output.dense.weight',
        'ff.linear2.b': 'output.dense.bias',
        'ln1.weight': 'attention.output.LayerNorm.weight',
        'ln1.bias': 'attention.output.LayerNorm.bias',
        'ln2.weight': 'output.LayerNorm.weight',
        'ln2.bias': 'output.LayerNorm.bias'
      };
      
      const pytorchComponent = componentMap[component] || component;
      return `encoder.layer.${blockNum}.${pytorchComponent}`;
    }
    
    return nameMap[jsName] || jsName;
  }
}