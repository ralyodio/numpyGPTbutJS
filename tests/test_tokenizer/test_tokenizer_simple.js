import { CharTokenizer, WordTokenizer, BPETokenizer } from '../../src/tokenizer/index.js';

/**
 * Test suite for Tokenizers
 */

// Test utilities
function assertArrayEqual(actual, expected, message = '') {
  if (actual.length !== expected.length) {
    throw new Error(`${message} Array lengths differ: expected ${expected.length}, got ${actual.length}`);
  }
  
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(`${message} Arrays differ at index ${i}: expected ${expected[i]}, got ${actual[i]}`);
    }
  }
}

function assertEqual(actual, expected, message = '') {
  if (actual !== expected) {
    throw new Error(`${message} Expected ${expected}, got ${actual}`);
  }
}

// Test Character Tokenizer
function testCharTokenizerBasic() {
  console.log('Testing CharTokenizer basic functionality...');
  
  const tokenizer = new CharTokenizer();
  const text = "Hello, world!";
  
  // Build vocabulary
  tokenizer.buildVocab(text);
  
  // Check vocabulary properties
  assertEqual(tokenizer.vocabSize, 4 + 10, 'Vocab size'); // 4 special + 10 unique chars
  assertEqual(tokenizer.eosTokenId, 3, 'EOS token ID');
  assertEqual(tokenizer.bosTokenId, 2, 'BOS token ID');
  assertEqual(tokenizer.padTokenId, 0, 'PAD token ID');
  assertEqual(tokenizer.unkTokenId, 1, 'UNK token ID');
  
  // Test encoding with text that's in the vocabulary
  const encoded = tokenizer.encode("Hello", true, true);
  assertEqual(encoded[0], 2, 'First token should be BOS');
  assertEqual(encoded[encoded.length - 1], 3, 'Last token should be EOS');
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  assertEqual(decoded, "Hello", 'Decoded text should match original');
  
  console.log('✓ CharTokenizer basic test passed');
}

function testCharTokenizerSpecialChars() {
  console.log('Testing CharTokenizer with special characters...');
  
  const tokenizer = new CharTokenizer();
  const text = "Hello\nworld\t!";
  
  tokenizer.buildVocab(text);
  
  // Test encoding and decoding with special characters
  const encoded = tokenizer.encode(text, false, false);
  const decoded = tokenizer.decode(encoded);
  
  assertEqual(decoded, text, 'Should handle special characters correctly');
  
  console.log('✓ CharTokenizer special characters test passed');
}

function testCharTokenizerUnknownChars() {
  console.log('Testing CharTokenizer with unknown characters...');
  
  const tokenizer = new CharTokenizer();
  tokenizer.buildVocab("abc");
  
  // Encode text with unknown character
  const encoded = tokenizer.encode("abx", false, false);
  
  // 'x' should be encoded as <unk> (index 1)
  assertEqual(encoded[2], 1, 'Unknown character should be encoded as UNK');
  
  console.log('✓ CharTokenizer unknown characters test passed');
}

// Test Word Tokenizer
function testWordTokenizerBasic() {
  console.log('Testing WordTokenizer basic functionality...');
  
  const tokenizer = new WordTokenizer();
  const text = "Hello world! How are you?";
  
  // Build vocabulary
  tokenizer.buildVocab(text);
  
  // Check vocabulary properties
  if (tokenizer.vocabSize < 4) {
    throw new Error('Vocab size should be at least 4 (special tokens)');
  }
  
  assertEqual(tokenizer.eosTokenId, 3, 'EOS token ID');
  assertEqual(tokenizer.bosTokenId, 2, 'BOS token ID');
  
  // Test encoding
  const encoded = tokenizer.encode("Hello world", true, true);
  assertEqual(encoded[0], 2, 'First token should be BOS');
  assertEqual(encoded[encoded.length - 1], 3, 'Last token should be EOS');
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  if (!decoded.includes("hello") || !decoded.includes("world")) {
    throw new Error('Decoded text should contain original words');
  }
  
  console.log('✓ WordTokenizer basic test passed');
}

function testWordTokenizerPunctuation() {
  console.log('Testing WordTokenizer with punctuation...');
  
  const tokenizer = new WordTokenizer();
  const text = "Hello, world! How's it going?";
  
  tokenizer.buildVocab(text);
  
  // Test encoding and decoding
  const encoded = tokenizer.encode(text, false, false);
  const decoded = tokenizer.decode(encoded);
  
  // Should handle punctuation reasonably
  if (!decoded.includes("hello") || !decoded.includes("world")) {
    throw new Error('Should preserve main words');
  }
  
  console.log('✓ WordTokenizer punctuation test passed');
}

function testWordTokenizerSpecialTokens() {
  console.log('Testing WordTokenizer with special tokens...');
  
  const tokenizer = new WordTokenizer();
  const text = "Line 1\nLine 2\tTabbed";
  
  tokenizer.buildVocab(text);
  
  // Test encoding and decoding with newlines and tabs
  const encoded = tokenizer.encode(text, false, false);
  const decoded = tokenizer.decode(encoded);
  
  // Should preserve newlines and tabs
  if (!decoded.includes('\n') || !decoded.includes('\t')) {
    throw new Error('Should preserve special characters');
  }
  
  console.log('✓ WordTokenizer special tokens test passed');
}

function testWordTokenizerVocabLimit() {
  console.log('Testing WordTokenizer with vocabulary limit...');
  
  const tokenizer = new WordTokenizer(1, 10); // min_freq=1, max_vocab=10
  const text = "the quick brown fox jumps over the lazy dog and cat";
  
  tokenizer.buildVocab(text);
  
  // Vocabulary should be limited to 10
  assertEqual(tokenizer.vocabSize, 10, 'Vocab size should be limited to 10');
  
  console.log('✓ WordTokenizer vocab limit test passed');
}

// Test BPE Tokenizer
function testBPETokenizerBasic() {
  console.log('Testing BPETokenizer basic functionality...');
  
  const tokenizer = new BPETokenizer(50); // Small vocab for testing
  const text = "hello world hello";
  
  // Build vocabulary
  tokenizer.buildVocab(text);
  
  // Check vocabulary properties
  if (tokenizer.vocabSize < 4) {
    throw new Error('Vocab size should be at least 4 (special tokens)');
  }
  
  assertEqual(tokenizer.eosTokenId, 3, 'EOS token ID');
  assertEqual(tokenizer.bosTokenId, 2, 'BOS token ID');
  
  // Test encoding
  const encoded = tokenizer.encode("hello", false, false);
  if (encoded.length === 0) {
    throw new Error('Encoding should produce tokens');
  }
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  if (!decoded.includes("hello")) {
    throw new Error('Decoded text should contain original word');
  }
  
  console.log('✓ BPETokenizer basic test passed');
}

function testBPETokenizerMerges() {
  console.log('Testing BPETokenizer merge learning...');
  
  const tokenizer = new BPETokenizer(100);
  const text = "hello hello hello world world";
  
  tokenizer.buildVocab(text);
  
  // Should have learned some merges
  if (tokenizer.merges.length === 0) {
    throw new Error('Should have learned some merges');
  }
  
  // Test that repeated patterns are handled efficiently
  const encoded1 = tokenizer.encode("hello", false, false);
  const encoded2 = tokenizer.encode("world", false, false);
  
  if (encoded1.length === 0 || encoded2.length === 0) {
    throw new Error('Should encode words to tokens');
  }
  
  console.log('✓ BPETokenizer merges test passed');
}

function testBPETokenizerSpecialTokens() {
  console.log('Testing BPETokenizer with special tokens...');
  
  const tokenizer = new BPETokenizer(100);
  const text = "Hello\nworld\ttab";
  
  tokenizer.buildVocab(text);
  
  // Test encoding and decoding with special characters
  const encoded = tokenizer.encode(text, false, false);
  const decoded = tokenizer.decode(encoded);
  
  // Should preserve newlines and tabs
  if (!decoded.includes('\n') || !decoded.includes('\t')) {
    throw new Error('Should preserve special characters');
  }
  
  console.log('✓ BPETokenizer special tokens test passed');
}

function testBPETokenizerSubwords() {
  console.log('Testing BPETokenizer subword handling...');
  
  const tokenizer = new BPETokenizer(50);
  const text = "running runner run runs";
  
  tokenizer.buildVocab(text);
  
  // Test that it can handle related words
  const encoded = tokenizer.encode("running", false, false);
  const decoded = tokenizer.decode(encoded);
  
  if (!decoded.includes("run")) {
    throw new Error('Should handle subword patterns');
  }
  
  console.log('✓ BPETokenizer subwords test passed');
}

// Test Tokenizer State Management
function testTokenizerStateSaving() {
  console.log('Testing tokenizer state saving/loading...');
  
  // Test CharTokenizer
  const charTokenizer1 = new CharTokenizer();
  charTokenizer1.buildVocab("hello world");
  const charState = charTokenizer1.getState();
  
  const charTokenizer2 = new CharTokenizer();
  charTokenizer2.loadState(charState);
  
  assertEqual(charTokenizer2.vocabSize, charTokenizer1.vocabSize, 'CharTokenizer vocab size should match');
  
  // Test WordTokenizer
  const wordTokenizer1 = new WordTokenizer();
  wordTokenizer1.buildVocab("hello world");
  const wordState = wordTokenizer1.getState();
  
  const wordTokenizer2 = new WordTokenizer();
  wordTokenizer2.loadState(wordState);
  
  assertEqual(wordTokenizer2.vocabSize, wordTokenizer1.vocabSize, 'WordTokenizer vocab size should match');
  
  // Test BPETokenizer
  const bpeTokenizer1 = new BPETokenizer(50);
  bpeTokenizer1.buildVocab("hello world hello");
  const bpeState = bpeTokenizer1.getState();
  
  const bpeTokenizer2 = new BPETokenizer();
  bpeTokenizer2.loadState(bpeState);
  
  assertEqual(bpeTokenizer2.vocabSize, bpeTokenizer1.vocabSize, 'BPETokenizer vocab size should match');
  
  console.log('✓ Tokenizer state saving test passed');
}

// Test Round-trip Consistency
function testTokenizerRoundTrip() {
  console.log('Testing tokenizer round-trip consistency...');
  
  const testTexts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Testing\nnewlines\tand\ttabs.",
    "Numbers 123 and symbols @#$%"
  ];
  
  for (const text of testTexts) {
    // Test CharTokenizer
    const charTokenizer = new CharTokenizer();
    charTokenizer.buildVocab(text);
    const charEncoded = charTokenizer.encode(text, false, false);
    const charDecoded = charTokenizer.decode(charEncoded);
    
    if (charDecoded !== text) {
      throw new Error(`CharTokenizer round-trip failed for: "${text}"`);
    }
    
    // Test WordTokenizer
    const wordTokenizer = new WordTokenizer();
    wordTokenizer.buildVocab(text);
    const wordEncoded = wordTokenizer.encode(text, false, false);
    const wordDecoded = wordTokenizer.decode(wordEncoded);
    
    // Word tokenizer may not preserve exact spacing/punctuation, so just check key words
    const originalWords = text.toLowerCase().match(/\w+/g) || [];
    const decodedWords = wordDecoded.toLowerCase().match(/\w+/g) || [];
    
    for (const word of originalWords) {
      if (!decodedWords.includes(word)) {
        throw new Error(`WordTokenizer lost word "${word}" in round-trip`);
      }
    }
    
    // Test BPETokenizer
    const bpeTokenizer = new BPETokenizer(200);
    bpeTokenizer.buildVocab(text);
    const bpeEncoded = bpeTokenizer.encode(text, false, false);
    const bpeDecoded = bpeTokenizer.decode(bpeEncoded);
    
    // BPE may not preserve exact spacing, so check key words
    for (const word of originalWords) {
      if (!bpeDecoded.toLowerCase().includes(word)) {
        throw new Error(`BPETokenizer lost word "${word}" in round-trip`);
      }
    }
  }
  
  console.log('✓ Tokenizer round-trip test passed');
}

// Run all tests
function runAllTests() {
  console.log('=== Testing Tokenizers ===\n');
  
  try {
    // CharTokenizer tests
    testCharTokenizerBasic();
    testCharTokenizerSpecialChars();
    testCharTokenizerUnknownChars();
    
    // WordTokenizer tests
    testWordTokenizerBasic();
    testWordTokenizerPunctuation();
    testWordTokenizerSpecialTokens();
    testWordTokenizerVocabLimit();
    
    // BPETokenizer tests
    testBPETokenizerBasic();
    testBPETokenizerMerges();
    testBPETokenizerSpecialTokens();
    testBPETokenizerSubwords();
    
    // General tests
    testTokenizerStateSaving();
    testTokenizerRoundTrip();
    
    console.log('\n🎉 All tokenizer tests passed!');
    return true;
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    return false;
  }
}

// Export for use in other test files
export { runAllTests };

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests();
}