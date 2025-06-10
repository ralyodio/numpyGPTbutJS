/**
 * Character-level Tokenizer
 * JavaScript equivalent of numpyGPT/tokenizer/char_level.py
 * 
 * Simple character-based tokenization where each character is a token
 */
export class CharTokenizer {
  constructor() {
    this.chars = [];
    this.stoi = {}; // string to index
    this.itos = {}; // index to string
  }

  /**
   * Build vocabulary from text
   * @param {string} text - Training text to build vocabulary from
   */
  buildVocab(text) {
    // Get unique characters and sort them
    const uniqueChars = [...new Set(text)].sort();
    
    // Add special tokens at the beginning
    this.chars = ['<pad>', '<unk>', '<bos>', '<eos>', ...uniqueChars];
    
    // Build string-to-index and index-to-string mappings
    this.stoi = {};
    this.itos = {};
    
    for (let i = 0; i < this.chars.length; i++) {
      const char = this.chars[i];
      this.stoi[char] = i;
      this.itos[i] = char;
    }
  }

  /**
   * Encode text to token indices
   * @param {string} text - Text to encode
   * @param {boolean} addBos - Whether to add beginning-of-sequence token
   * @param {boolean} addEos - Whether to add end-of-sequence token
   * @returns {Array} - Array of token indices
   */
  encode(text, addBos = true, addEos = true) {
    // Convert each character to its index, use <unk> (index 1) for unknown chars
    const tokens = [];
    
    for (const char of text) {
      const index = this.stoi[char] !== undefined ? this.stoi[char] : 1; // 1 is <unk>
      tokens.push(index);
    }
    
    // Add special tokens if requested
    if (addBos) {
      tokens.unshift(this.stoi['<bos>']); // Add at beginning
    }
    
    if (addEos) {
      tokens.push(this.stoi['<eos>']); // Add at end
    }
    
    return tokens;
  }

  /**
   * Decode token indices to text
   * @param {Array} tokens - Array of token indices
   * @returns {string} - Decoded text
   */
  decode(tokens) {
    const chars = [];
    
    for (const token of tokens) {
      if (this.itos[token] !== undefined) {
        chars.push(this.itos[token]);
      }
    }
    
    // Join characters and remove special tokens
    let text = chars.join('');
    
    // Remove special tokens from the output
    const specialTokens = ['<bos>', '<eos>', '<pad>', '<unk>'];
    for (const special of specialTokens) {
      text = text.replaceAll(special, '');
    }
    
    return text;
  }

  /**
   * Get vocabulary size
   * @returns {number} - Size of vocabulary
   */
  get vocabSize() {
    return this.chars.length;
  }

  /**
   * Get end-of-sequence token ID
   * @returns {number} - EOS token ID
   */
  get eosTokenId() {
    return 3; // <eos> is at index 3
  }

  /**
   * Get beginning-of-sequence token ID
   * @returns {number} - BOS token ID
   */
  get bosTokenId() {
    return 2; // <bos> is at index 2
  }

  /**
   * Get pad token ID
   * @returns {number} - PAD token ID
   */
  get padTokenId() {
    return 0; // <pad> is at index 0
  }

  /**
   * Get unknown token ID
   * @returns {number} - UNK token ID
   */
  get unkTokenId() {
    return 1; // <unk> is at index 1
  }

  /**
   * Get tokenizer state for saving/loading
   * @returns {Object} - Tokenizer state
   */
  getState() {
    return {
      chars: this.chars,
      stoi: this.stoi,
      itos: this.itos,
      vocabSize: this.vocabSize
    };
  }

  /**
   * Load tokenizer state
   * @param {Object} state - Tokenizer state to load
   */
  loadState(state) {
    this.chars = state.chars;
    this.stoi = state.stoi;
    this.itos = state.itos;
  }
}