# Tokenizers: From Text to Numbers

*How neural networks understand language.*

## the problem

Neural networks work with numbers. Text is strings. You need to convert "Hello world" into `[1, 42, 7]` somehow. But which conversion? 

* Character-level: `['H','e','l','l','o',' ','w','o','r','l','d']`
* Word-level: `['Hello', 'world']`  
* BPE: `['Hel', 'lo', 'world']`

Each choice shapes what your model learns.

## character-level: simple and universal

```javascript
class CharTokenizer {
    constructor() {
        this.chars = [];
        this.stoi = {}; // string to index
        this.itos = {}; // index to string
    }
    
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
    
    get vocabSize() {
        return this.chars.length;
    }
}
```

**Pros:** Never sees unknown words. Works for any language.<br>
**Cons:** Long sequences. Hard to learn word meanings.

## word-level: semantic units

```javascript
class WordTokenizer {
    constructor(minFreq = 1, maxVocabSize = 10000) {
        this.minFreq = minFreq;
        this.maxVocabSize = maxVocabSize;
        this.vocab = [];
        this.stoi = {};
        this.itos = {};
    }
    
    tokenize(text) {
        // Simple tokenization: split on whitespace and punctuation
        return text.toLowerCase()
                  .match(/\b\w+\b|[^\s\w]/g) || [];
    }
    
    buildVocab(text) {
        // Tokenize all text
        const allTokens = this.tokenize(text);
        
        // Count token frequencies
        const tokenCounts = {};
        for (const token of allTokens) {
            tokenCounts[token] = (tokenCounts[token] || 0) + 1;
        }
        
        // Filter by minimum frequency and sort by frequency
        const filteredTokens = Object.entries(tokenCounts)
            .filter(([token, count]) => count >= this.minFreq)
            .sort(([, a], [, b]) => b - a)
            .slice(0, this.maxVocabSize - 4) // Reserve space for special tokens
            .map(([token]) => token);
        
        // Add special tokens at the beginning
        this.vocab = ['<pad>', '<unk>', '<bos>', '<eos>', ...filteredTokens];
        
        // Build mappings
        this.stoi = {};
        this.itos = {};
        
        for (let i = 0; i < this.vocab.length; i++) {
            const token = this.vocab[i];
            this.stoi[token] = i;
            this.itos[i] = token;
        }
    }
    
    encode(text, addBos = true, addEos = true) {
        const tokens = this.tokenize(text);
        const indices = [];
        
        for (const token of tokens) {
            const index = this.stoi[token] !== undefined ? this.stoi[token] : 1; // 1 is <unk>
            indices.push(index);
        }
        
        // Add special tokens if requested
        if (addBos) {
            indices.unshift(this.stoi['<bos>']);
        }
        
        if (addEos) {
            indices.push(this.stoi['<eos>']);
        }
        
        return indices;
    }
    
    decode(indices) {
        const tokens = [];
        
        for (const index of indices) {
            if (this.itos[index] !== undefined) {
                const token = this.itos[index];
                // Skip special tokens in output
                if (!['<pad>', '<unk>', '<bos>', '<eos>'].includes(token)) {
                    tokens.push(token);
                }
            }
        }
        
        return tokens.join(' ');
    }
    
    get vocabSize() {
        return this.vocab.length;
    }
}
```

**Pros:** Captures word meanings. Shorter sequences.<br>
**Cons:** Large vocabularies. Unknown words become `<unk>`.

## bpe: best of both worlds

Start with characters. Merge frequent pairs iteratively.

```javascript
// Initial: "hello" -> ['h', 'e', 'l', 'l', 'o']
// Count pairs: ('l','l') appears most
// Merge: "hello" -> ['h', 'e', 'll', 'o'] 
// Repeat until vocab_size reached
```

The algorithm:

```javascript
class BPETokenizer {
    constructor(maxVocabSize = 1000) {
        this.maxVocabSize = maxVocabSize;
        this.vocab = [];
        this.merges = [];
        this.stoi = {};
        this.itos = {};
    }
    
    buildVocab(text) {
        // Step 1: Get word frequencies
        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        const wordFreqs = {};
        
        for (const word of words) {
            wordFreqs[word] = (wordFreqs[word] || 0) + 1;
        }
        
        // Step 2: Initialize vocabulary with characters
        const chars = new Set();
        for (const word in wordFreqs) {
            for (const char of word) {
                chars.add(char);
            }
        }
        
        // Start with special tokens and characters
        this.vocab = ['<pad>', '<unk>', '<bos>', '<eos>', ...Array.from(chars).sort()];
        
        // Step 3: Create initial word representations
        let vocab = {};
        for (const [word, freq] of Object.entries(wordFreqs)) {
            vocab[word.split('').join(' ') + ' </w>'] = freq;
        }
        
        // Step 4: Iteratively merge most frequent pairs
        const numMerges = this.maxVocabSize - this.vocab.length;
        
        for (let i = 0; i < numMerges; i++) {
            const pairs = this._getPairs(vocab);
            
            if (pairs.size === 0) break;
            
            // Find most frequent pair
            let bestPair = null;
            let maxCount = 0;
            
            for (const [pair, count] of pairs) {
                if (count > maxCount) {
                    maxCount = count;
                    bestPair = pair;
                }
            }
            
            if (!bestPair) break;
            
            // Merge the best pair
            vocab = this._mergeVocab(bestPair, vocab);
            this.merges.push(bestPair);
            
            // Add merged token to vocabulary
            const mergedToken = bestPair[0] + bestPair[1];
            if (!this.vocab.includes(mergedToken)) {
                this.vocab.push(mergedToken);
            }
        }
        
        // Build mappings
        this._buildMappings();
    }
    
    _getPairs(vocab) {
        const pairs = new Map();
        
        for (const [word, freq] of Object.entries(vocab)) {
            const symbols = word.split(' ');
            
            for (let i = 0; i < symbols.length - 1; i++) {
                const pair = [symbols[i], symbols[i + 1]];
                const pairKey = pair.join(',');
                pairs.set(pairKey, (pairs.get(pairKey) || 0) + freq);
            }
        }
        
        return pairs;
    }
    
    _mergeVocab(pair, vocab) {
        const newVocab = {};
        const pattern = new RegExp(pair[0] + ' ' + pair[1], 'g');
        
        for (const [word, freq] of Object.entries(vocab)) {
            const newWord = word.replace(pattern, pair[0] + pair[1]);
            newVocab[newWord] = freq;
        }
        
        return newVocab;
    }
    
    _buildMappings() {
        this.stoi = {};
        this.itos = {};
        
        for (let i = 0; i < this.vocab.length; i++) {
            const token = this.vocab[i];
            this.stoi[token] = i;
            this.itos[i] = token;
        }
    }
    
    encode(text, addBos = true, addEos = true) {
        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        const tokens = [];
        
        for (const word of words) {
            const wordTokens = this._encodeWord(word);
            tokens.push(...wordTokens);
        }
        
        // Convert tokens to indices
        const indices = tokens.map(token => this.stoi[token] || 1); // 1 is <unk>
        
        // Add special tokens if requested
        if (addBos) {
            indices.unshift(this.stoi['<bos>']);
        }
        
        if (addEos) {
            indices.push(this.stoi['<eos>']);
        }
        
        return indices;
    }
    
    _encodeWord(word) {
        // Start with character-level split
        let wordTokens = word.split('').join(' ') + ' </w>';
        
        // Apply learned merges
        for (const [first, second] of this.merges) {
            const pattern = new RegExp(first + ' ' + second, 'g');
            wordTokens = wordTokens.replace(pattern, first + second);
        }
        
        return wordTokens.split(' ').filter(token => token.length > 0);
    }
    
    decode(indices) {
        const tokens = [];
        
        for (const index of indices) {
            if (this.itos[index] !== undefined) {
                const token = this.itos[index];
                // Skip special tokens in output
                if (!['<pad>', '<unk>', '<bos>', '<eos>'].includes(token)) {
                    tokens.push(token);
                }
            }
        }
        
        // Join tokens and handle word boundaries
        let text = tokens.join('');
        text = text.replace(/</w>/g, ' '); // Replace end-of-word markers with spaces
        
        return text.trim();
    }
    
    get vocabSize() {
        return this.vocab.length;
    }
}
```

**Example walkthrough:**
```
Text: "hello hello world"
Initial vocab: {'h e l l o </w>': 2, 'w o r l d </w>': 1}

Iteration 1: Most frequent pair is ('l', 'l')
Merge: {'h e ll o </w>': 2, 'w o r l d </w>': 1}

Iteration 2: Most frequent pair is ('h', 'e') 
Merge: {'he ll o </w>': 2, 'w o r l d </w>': 1}

Continue until vocab_size reached...
```

## the three approaches compared

| Method | Sequence Length | Unknown Words | Use Case |
|--------|-----------------|---------------|----------|
| Char   | Very Long       | Never         | Small data, multilingual |
| Word   | Short           | Common        | Large vocab, semantic tasks |
| BPE    | Medium          | Rare          | Production systems |

## special tokens

Every tokenizer needs these:

```javascript
const specialTokens = ['<pad>', '<unk>', '<bos>', '<eos>'];
// <pad>: padding for batches
// <unk>: unknown words  
// <bos>: beginning of sequence
// <eos>: end of sequence
```

## implementation pattern

All tokenizers follow this interface:

```javascript
class Tokenizer {
    buildVocab(text) {
        // Build token <-> index mappings
        throw new Error('buildVocab must be implemented');
    }
    
    encode(text, addBos = true, addEos = true) {
        // text -> list of indices
        const tokens = this.tokenize(text);
        return tokens.map(token => this.stoi[token] || this.stoi['<unk>']);
    }
    
    decode(indices) {
        // indices -> text (join method depends on tokenizer type)
        const tokens = indices.map(i => this.itos[i]).filter(Boolean);
        
        // Remove special tokens
        const filteredTokens = tokens.filter(token => 
            !['<pad>', '<unk>', '<bos>', '<eos>'].includes(token)
        );
        
        // Join method depends on tokenizer type:
        return filteredTokens.join('');     // char-level: no spaces
        // return filteredTokens.join(' ');  // word-level: spaces between words
    }
    
    get vocabSize() {
        return this.vocab.length;
    }
}
```

## usage examples

**Character-level tokenization:**
```javascript
import { CharTokenizer } from './src/tokenizer/char_level.js';

const tokenizer = new CharTokenizer();
const text = "Hello, world! How are you?";

tokenizer.buildVocab(text);
console.log(`Vocabulary size: ${tokenizer.vocabSize}`);

const encoded = tokenizer.encode(text);
console.log('Encoded:', encoded);

const decoded = tokenizer.decode(encoded);
console.log('Decoded:', decoded);
```

**Word-level tokenization:**
```javascript
import { WordTokenizer } from './src/tokenizer/word_level.js';

const tokenizer = new WordTokenizer(minFreq=2, maxVocabSize=1000);
const text = "The quick brown fox jumps over the lazy dog. The dog was very lazy.";

tokenizer.buildVocab(text);
console.log(`Vocabulary size: ${tokenizer.vocabSize}`);
console.log('Sample vocab:', tokenizer.vocab.slice(0, 10));

const encoded = tokenizer.encode("The fox jumps");
console.log('Encoded:', encoded);

const decoded = tokenizer.decode(encoded);
console.log('Decoded:', decoded);
```

**BPE tokenization:**
```javascript
import { BPETokenizer } from './src/tokenizer/bpe.js';

const tokenizer = new BPETokenizer(maxVocabSize=500);
const text = "hello world hello universe world peace";

tokenizer.buildVocab(text);
console.log(`Vocabulary size: ${tokenizer.vocabSize}`);
console.log('Learned merges:', tokenizer.merges.slice(0, 5));

const encoded = tokenizer.encode("hello world");
console.log('Encoded:', encoded);

const decoded = tokenizer.decode(encoded);
console.log('Decoded:', decoded);
```

## choosing the right tokenizer

**Use Character-level when:**
- Working with small datasets
- Dealing with multiple languages
- Need to handle any possible input
- Memory/compute is limited

**Use Word-level when:**
- Have large, clean datasets
- Working with well-defined languages
- Semantic understanding is crucial
- Can handle large vocabularies

**Use BPE when:**
- Building production systems
- Need balance between efficiency and coverage
- Working with diverse text sources
- Want subword information

## tokenizer state management

Save and load tokenizer state:

```javascript
class TokenizerState {
    static save(tokenizer, filepath) {
        const state = {
            type: tokenizer.constructor.name,
            vocab: tokenizer.vocab,
            stoi: tokenizer.stoi,
            itos: tokenizer.itos,
            merges: tokenizer.merges || null, // BPE only
            minFreq: tokenizer.minFreq || null, // Word only
            maxVocabSize: tokenizer.maxVocabSize || null
        };
        
        const fs = require('fs');
        fs.writeFileSync(filepath, JSON.stringify(state, null, 2));
    }
    
    static load(filepath) {
        const fs = require('fs');
        const state = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
        
        let tokenizer;
        switch (state.type) {
            case 'CharTokenizer':
                tokenizer = new CharTokenizer();
                break;
            case 'WordTokenizer':
                tokenizer = new WordTokenizer(state.minFreq, state.maxVocabSize);
                break;
            case 'BPETokenizer':
                tokenizer = new BPETokenizer(state.maxVocabSize);
                tokenizer.merges = state.merges;
                break;
            default:
                throw new Error(`Unknown tokenizer type: ${state.type}`);
        }
        
        tokenizer.vocab = state.vocab;
        tokenizer.stoi = state.stoi;
        tokenizer.itos = state.itos;
        
        return tokenizer;
    }
}

// Usage
TokenizerState.save(tokenizer, 'tokenizer.json');
const loadedTokenizer = TokenizerState.load('tokenizer.json');
```

## performance considerations

**Memory usage:**
```javascript
// Character-level: ~100 tokens
// Word-level: ~10,000-50,000 tokens  
// BPE: ~1,000-5,000 tokens

console.log(`Memory usage: ${tokenizer.vocabSize * 4} bytes for vocab indices`);
```

**Encoding speed:**
```javascript
function benchmarkTokenizer(tokenizer, text, iterations = 1000) {
    const start = Date.now();
    
    for (let i = 0; i < iterations; i++) {
        tokenizer.encode(text);
    }
    
    const end = Date.now();
    const timePerEncoding = (end - start) / iterations;
    
    console.log(`Average encoding time: ${timePerEncoding.toFixed(3)}ms`);
}
```

## key takeaways

1. **Character-level** is simple and universal but creates long sequences
2. **Word-level** captures semantics but struggles with unknown words
3. **BPE** provides the best balance for most applications
4. **Special tokens** are essential for proper sequence handling
5. **Vocabulary size** directly impacts model size and memory usage

Choose your tokenizer based on your data, computational resources, and task requirements. When in doubt, start with BPE!
