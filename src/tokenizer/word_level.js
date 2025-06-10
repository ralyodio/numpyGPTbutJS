/**
 * Word-level Tokenizer
 * JavaScript equivalent of numpyGPT/tokenizer/word_level.py
 * 
 * Tokenizes text into words, punctuation, and special tokens
 */
export class WordTokenizer {
  constructor(minFreq = 1, maxVocabSize = null) {
    this.minFreq = minFreq;
    this.maxVocabSize = maxVocabSize;
    this.words = [];
    this.stoi = {}; // string to index
    this.itos = {}; // index to string
  }

  /**
   * Build vocabulary from text
   * @param {string} text - Training text to build vocabulary from
   */
  buildVocab(text) {
    // Replace special characters with tokens
    text = text.replace(/\n/g, ' <|newline|> ');
    text = text.replace(/\t/g, ' <|tab|> ');
    text = text.replace(/\r/g, ' <|carriage_return|> ');

    // Tokenize text into:
    // 1. Special tokens with format <|...|>   -> <\|[^|]+\|>
    // 2. Words (alphanumerics/underscores)    -> \b\w+\b
    // 3. Punctuation/non-word characters      -> [^\s\w]
    const tokenRegex = /<\|[^|]+\|>|\b\w+\b|[^\s\w]/g;
    const tokens = text.match(tokenRegex) || [];
    
    // Convert to lowercase except for special tokens
    const processedTokens = tokens.map(token => {
      if (token.startsWith('<|') && token.endsWith('|>')) {
        return token; // Keep special tokens as-is
      }
      return token.toLowerCase();
    });

    // Count word frequencies
    const wordCounts = {};
    for (const token of processedTokens) {
      wordCounts[token] = (wordCounts[token] || 0) + 1;
    }

    // Filter by minimum frequency
    const filteredWords = Object.keys(wordCounts).filter(
      word => wordCounts[word] >= this.minFreq
    );

    // Sort by frequency (descending) then alphabetically
    const sortedWords = filteredWords.sort((a, b) => {
      const freqDiff = wordCounts[b] - wordCounts[a];
      if (freqDiff !== 0) return freqDiff;
      return a.localeCompare(b);
    });

    // Limit vocabulary size if specified
    let finalWords = sortedWords;
    if (this.maxVocabSize) {
      finalWords = sortedWords.slice(0, this.maxVocabSize - 4); // Reserve 4 for special tokens
    }

    // Add special tokens at the beginning
    this.words = ['<pad>', '<unk>', '<bos>', '<eos>', ...finalWords];
    
    // Build mappings
    this.stoi = {};
    this.itos = {};
    
    for (let i = 0; i < this.words.length; i++) {
      const word = this.words[i];
      this.stoi[word] = i;
      this.itos[i] = word;
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
    // Replace special characters with tokens
    text = text.replace(/\n/g, ' <|newline|> ');
    text = text.replace(/\t/g, ' <|tab|> ');
    text = text.replace(/\r/g, ' <|carriage_return|> ');

    // Tokenize using the same regex as in buildVocab
    const tokenRegex = /<\|[^|]+\|>|\b\w+\b|[^\s\w]/g;
    const tokens = text.match(tokenRegex) || [];
    
    // Convert to lowercase except for special tokens
    const processedTokens = tokens.map(token => {
      if (token.startsWith('<|') && token.endsWith('|>')) {
        return token; // Keep special tokens as-is
      }
      return token.toLowerCase();
    });

    // Convert tokens to indices
    const indices = processedTokens.map(token => 
      this.stoi[token] !== undefined ? this.stoi[token] : 1 // 1 is <unk>
    );

    // Add special tokens if requested
    if (addBos) {
      indices.unshift(2); // <bos> is at index 2
    }
    
    if (addEos) {
      indices.push(3); // <eos> is at index 3
    }

    return indices;
  }

  /**
   * Decode token indices to text
   * @param {Array} tokens - Array of token indices
   * @returns {string} - Decoded text
   */
  decode(tokens) {
    const words = tokens
      .map(token => this.itos[token])
      .filter(word => word !== undefined);

    // Remove special tokens
    const specialTokens = ['<bos>', '<eos>', '<pad>', '<unk>'];
    const filteredWords = words.filter(word => !specialTokens.includes(word));

    const result = [];
    
    for (let i = 0; i < filteredWords.length; i++) {
      const word = filteredWords[i];
      
      if (word === '<|newline|>') {
        result.push('\n');
      } else if (word === '<|tab|>') {
        result.push('\t');
      } else if (word === '<|carriage_return|>') {
        result.push('\r');
      } else {
        result.push(word);
        
        // Add space after word if next word is not punctuation
        if (i < filteredWords.length - 1) {
          const nextWord = filteredWords[i + 1];
          if (!nextWord.startsWith('<|') && !'\'.,;!?)"'.includes(nextWord)) {
            result.push(' ');
          }
        }
      }
    }

    let text = result.join('');
    
    // Clean up spacing and punctuation
    text = text.replace(/ +/g, ' '); // Multiple spaces to single space
    text = text.replace(/ ([\'.,;:!?)])/g, '$1'); // Remove space before punctuation
    text = text.replace(/\b([A-Za-z]+) ?' ?(ll|re|ve|d|s|t|m)\b/g, "$1'$2"); // Fix contractions
    text = text.replace(/([a-z]) :/g, '$1:'); // Fix colons
    text = text.replace(/ *\n */g, '\n'); // Clean newlines
    text = text.replace(/ *\t */g, '\t'); // Clean tabs
    text = text.replace(/ *\r */g, '\r'); // Clean carriage returns
    
    return text.trim();
  }

  /**
   * Get vocabulary size
   * @returns {number} - Size of vocabulary
   */
  get vocabSize() {
    return this.words.length;
  }

  /**
   * Get end-of-sequence token ID
   * @returns {number} - EOS token ID
   */
  get eosTokenId() {
    return 3;
  }

  /**
   * Get beginning-of-sequence token ID
   * @returns {number} - BOS token ID
   */
  get bosTokenId() {
    return 2;
  }

  /**
   * Get pad token ID
   * @returns {number} - PAD token ID
   */
  get padTokenId() {
    return 0;
  }

  /**
   * Get unknown token ID
   * @returns {number} - UNK token ID
   */
  get unkTokenId() {
    return 1;
  }

  /**
   * Get tokenizer state for saving/loading
   * @returns {Object} - Tokenizer state
   */
  getState() {
    return {
      words: this.words,
      stoi: this.stoi,
      itos: this.itos,
      minFreq: this.minFreq,
      maxVocabSize: this.maxVocabSize,
      vocabSize: this.vocabSize
    };
  }

  /**
   * Load tokenizer state
   * @param {Object} state - Tokenizer state to load
   */
  loadState(state) {
    this.words = state.words;
    this.stoi = state.stoi;
    this.itos = state.itos;
    this.minFreq = state.minFreq;
    this.maxVocabSize = state.maxVocabSize;
  }
}