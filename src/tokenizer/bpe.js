/**
 * BPE (Byte Pair Encoding) Tokenizer
 * JavaScript equivalent of numpyGPT/tokenizer/bpe.py
 * 
 * BPE learns subword vocabulary by iteratively merging most frequent adjacent pairs
 * Start: "hello" -> ['h','e','l','l','o'] 
 * Find 'l','l' is most frequent pair -> merge to 'll'
 * Result: "hello" -> ['h','e','ll','o']
 * Repeat until vocab_size reached => meaningful subwords emerge naturally.
 */
export class BPETokenizer {
  constructor(vocabSize = 1000) {
    this.vocabSize = vocabSize;
    this.stoi = {}; // string to index
    this.itos = {}; // index to string
    this.merges = []; // list of merge pairs
    this.mergeRanks = {}; // merge pair to rank mapping
  }

  /**
   * Get word frequencies from texts
   * @param {string|Array} texts - Text(s) to analyze
   * @returns {Object} - Word frequency mapping
   */
  _getWordFreqs(texts) {
    if (typeof texts === 'string') {
      texts = [texts];
    }

    const wordFreqs = {};
    
    for (const text of texts) {
      // Replace special characters with tokens
      let processedText = text.replace(/\n/g, ' <|newline|> ');
      processedText = processedText.replace(/\t/g, ' <|tab|> ');
      processedText = processedText.replace(/\r/g, ' <|carriage_return|> ');

      // Tokenize text into:
      // 1. Special tokens with format <|...|>   -> <\|[^|]+\|>
      // 2. Words (alphanumerics/underscores)    -> \b\w+\b
      // 3. Punctuation/non-word characters      -> [^\s\w]
      const tokenRegex = /<\|[^|]+\|>|\b\w+\b|[^\s\w]/g;
      const tokens = processedText.match(tokenRegex) || [];
      
      // Keep special tokens as-is, don't lowercase them
      const processedTokens = tokens.map(token => {
        if (token.startsWith('<|') && token.endsWith('|>')) {
          return token;
        }
        return token;
      });

      for (const token of processedTokens) {
        wordFreqs[token] = (wordFreqs[token] || 0) + 1;
      }
    }

    return wordFreqs;
  }

  /**
   * Get all adjacent pairs in a word
   * @param {Array} word - Array of tokens/characters
   * @returns {Set} - Set of adjacent pairs
   */
  _getPairs(word) {
    const pairs = new Set();
    
    for (let i = 0; i < word.length - 1; i++) {
      pairs.add([word[i], word[i + 1]]);
    }
    
    return pairs;
  }

  /**
   * Merge a specific pair in a word
   * @param {Array} wordTokens - Array of word tokens
   * @param {Array} pair - Pair to merge [token1, token2]
   * @returns {Array} - Word tokens with pair merged
   */
  _mergeWord(wordTokens, pair) {
    const newTokens = [];
    let i = 0;
    
    while (i < wordTokens.length) {
      // If we find the pair, merge it
      if (i < wordTokens.length - 1 && 
          wordTokens[i] === pair[0] && 
          wordTokens[i + 1] === pair[1]) {
        newTokens.push(pair[0] + pair[1]);
        i += 2; // Skip both tokens
      } else {
        newTokens.push(wordTokens[i]);
        i += 1;
      }
    }
    
    return newTokens;
  }

  /**
   * Build vocabulary using BPE algorithm
   * @param {string|Array} texts - Training text(s)
   */
  buildVocab(texts) {
    const wordFreqs = this._getWordFreqs(texts);

    // Step 1: Initialize with character-level tokens
    let vocab = {};
    const baseChars = new Set();

    for (const [word, freq] of Object.entries(wordFreqs)) {
      if (word.startsWith('<|') && word.endsWith('|>')) {
        // Special tokens stay as single units
        vocab[word + ' </w>'] = freq;
        baseChars.add(word);
      } else {
        // Split into characters + end-of-word marker
        const charList = [...word, '</w>'];
        vocab[charList.join(' ')] = freq;
        for (const char of word) {
          baseChars.add(char);
        }
      }
    }

    baseChars.add('</w>');

    // Step 2: Calculate how many merges we can do
    const specialTokens = ['<pad>', '<unk>', '<bos>', '<eos>'];
    const baseVocabSize = specialTokens.length + baseChars.size;
    const availableForMerges = Math.max(0, this.vocabSize - baseVocabSize);

    // Step 3: Iteratively find and apply merges
    for (let i = 0; i < availableForMerges; i++) {
      const pairs = {};
      
      // Count all adjacent pairs across all words
      for (const [word, freq] of Object.entries(vocab)) {
        const wordTokens = word.split(' ');
        const wordPairs = this._getPairs(wordTokens);
        
        for (const pair of wordPairs) {
          // Don't merge pairs involving special tokens
          if (!pair.some(p => p.startsWith('<|') && p.includes('|>'))) {
            const pairKey = pair.join('|'); // Use | as separator to avoid conflicts
            pairs[pairKey] = (pairs[pairKey] || 0) + freq;
          }
        }
      }

      if (Object.keys(pairs).length === 0) {
        break;
      }

      // Find most frequent pair
      const bestPairKey = Object.keys(pairs).reduce((a, b) => 
        pairs[a] > pairs[b] ? a : b
      );
      const bestPair = bestPairKey.split('|');

      // Update vocab with the merged pair
      const newVocab = {};
      for (const [word, freq] of Object.entries(vocab)) {
        const wordTokens = word.split(' ');
        const mergedTokens = this._mergeWord(wordTokens, bestPair);
        newVocab[mergedTokens.join(' ')] = freq;
      }

      vocab = newVocab;
      this.merges.push(bestPair);
      this.mergeRanks[bestPair.join('|')] = i; // Remember when this was learned
    }

    // Step 4: Build final vocabulary
    const allTokens = [...specialTokens, ...Array.from(baseChars).sort()];
    
    for (const pair of this.merges) {
      allTokens.push(pair.join(''));
    }

    // Limit to vocab size
    const finalTokens = allTokens.slice(0, this.vocabSize);

    this.stoi = {};
    this.itos = {};
    
    for (let i = 0; i < finalTokens.length; i++) {
      this.stoi[finalTokens[i]] = i;
      this.itos[i] = finalTokens[i];
    }
    
    this.vocabSize = Object.keys(this.stoi).length;
  }

  /**
   * Tokenize a single word using learned merges
   * @param {string} word - Word to tokenize
   * @returns {Array} - Array of subword tokens
   */
  _tokenizeWord(word) {
    if (word.startsWith('<|') && word.endsWith('|>')) {
      return [word, '</w>'];
    }

    let wordTokens = [...word, '</w>'];

    while (wordTokens.length > 1) {
      const pairs = this._getPairs(wordTokens);
      
      if (pairs.size === 0) {
        break;
      }

      // Find the pair with the lowest merge rank (learned earliest)
      let bestPair = null;
      let bestRank = Infinity;
      
      for (const pair of pairs) {
        const pairKey = pair.join('|');
        const rank = this.mergeRanks[pairKey];
        
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }

      if (bestPair === null) {
        break;
      }

      wordTokens = this._mergeWord(wordTokens, bestPair);
    }

    return wordTokens;
  }

  /**
   * Encode text to token indices
   * @param {string} text - Text to encode
   * @param {boolean} addBos - Whether to add beginning-of-sequence token
   * @param {boolean} addEos - Whether to add end-of-sequence token
   * @returns {Array} - Array of token indices
   */
  encode(text, addBos = false, addEos = false) {
    // Replace special characters with tokens
    text = text.replace(/\n/g, ' <|newline|> ');
    text = text.replace(/\t/g, ' <|tab|> ');
    text = text.replace(/\r/g, ' <|carriage_return|> ');

    // Tokenize using the same regex as in buildVocab
    const tokenRegex = /<\|[^|]+\|>|\b\w+\b|[^\s\w]/g;
    const tokens = text.match(tokenRegex) || [];
    
    // Keep special tokens as-is
    const processedTokens = tokens.map(token => {
      if (token.startsWith('<|') && token.endsWith('|>')) {
        return token;
      }
      return token;
    });

    const indices = [];
    
    if (addBos) {
      indices.push(2); // <bos> index
    }

    for (const token of processedTokens) {
      const wordTokens = this._tokenizeWord(token);
      
      for (const wordToken of wordTokens) {
        const index = this.stoi[wordToken] !== undefined ? this.stoi[wordToken] : 1; // 1 is <unk>
        indices.push(index);
      }
    }

    if (addEos) {
      indices.push(3); // <eos> index
    }

    return indices;
  }

  /**
   * Decode token indices to text
   * @param {Array} indices - Array of token indices
   * @returns {string} - Decoded text
   */
  decode(indices) {
    const tokens = indices.map(idx => this.itos[idx] || '<unk>');

    // Remove special tokens
    const specialTokens = ['<bos>', '<eos>', '<pad>', '<unk>'];
    const filteredTokens = tokens.filter(token => !specialTokens.includes(token));

    const result = [];

    for (let i = 0; i < filteredTokens.length; i++) {
      const token = filteredTokens[i];
      
      if (token === '<|newline|>') {
        result.push('\n');
      } else if (token === '<|tab|>') {
        result.push('\t');
      } else if (token === '<|carriage_return|>') {
        result.push('\r');
      } else {
        // Check if token ends with </w> (end of word marker)
        if (token.endsWith('</w>')) {
          const word = token.slice(0, -4); // Remove '</w>'
          result.push(word);
          
          if (i < filteredTokens.length - 1) {
            result.push(' ');
          }
        } else {
          // It's a subword token
          result.push(token);
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
      vocabSize: this.vocabSize,
      stoi: this.stoi,
      itos: this.itos,
      merges: this.merges,
      mergeRanks: this.mergeRanks
    };
  }

  /**
   * Load tokenizer state
   * @param {Object} state - Tokenizer state to load
   */
  loadState(state) {
    this.vocabSize = state.vocabSize;
    this.stoi = state.stoi;
    this.itos = state.itos;
    this.merges = state.merges;
    this.mergeRanks = state.mergeRanks;
  }
}