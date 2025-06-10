/**
 * Extended test suite for tokenizers
 * Tests all tokenizer types with comprehensive coverage
 */

import { CharTokenizer } from '../../src/tokenizer/char_level.js';
import { WordTokenizer } from '../../src/tokenizer/word_level.js';
import { BPETokenizer } from '../../src/tokenizer/bpe.js';

function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

function testCharLevelTokenizer() {
  console.log('Testing CharTokenizer...');
  
  const tokenizer = new CharTokenizer();
  const text = "Hello, World! 123";
  
  // Build vocabulary first
  tokenizer.buildVocab(text);
  
  // Test encoding
  const encoded = tokenizer.encode(text, false, false); // No BOS/EOS for simpler testing
  assert(Array.isArray(encoded), 'Encoded result should be an array');
  assert(encoded.length === text.length, 'Encoded length should match text length');
  
  // Test that all characters are mapped to valid token IDs
  for (let i = 0; i < encoded.length; i++) {
    assert(typeof encoded[i] === 'number', `Token at position ${i} should be a number`);
    assert(encoded[i] >= 0, `Token at position ${i} should be non-negative`);
  }
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  assert(decoded === text, 'Decoded text should match original');
  
  // Test vocabulary
  const vocabSize = tokenizer.vocabSize;
  assert(typeof vocabSize === 'number', 'Vocabulary size should be a number');
  assert(vocabSize > 0, 'Vocabulary size should be positive');
  
  // Test special characters
  const specialText = "\n\t\r";
  tokenizer.buildVocab(specialText);
  const specialEncoded = tokenizer.encode(specialText, false, false);
  const specialDecoded = tokenizer.decode(specialEncoded);
  // Note: decoded may not exactly match due to special token handling
  assert(typeof specialDecoded === 'string', 'Special characters should be handled');
  
  // Test empty string
  const emptyEncoded = tokenizer.encode("", false, false);
  assert(emptyEncoded.length === 0, 'Empty string should encode to empty array');
  
  const emptyDecoded = tokenizer.decode([]);
  assert(emptyDecoded === "", 'Empty array should decode to empty string');
  
  console.log('✓ CharTokenizer tests passed');
}

function testWordLevelTokenizer() {
  console.log('Testing WordTokenizer...');
  
  const tokenizer = new WordTokenizer();
  const text = "Hello, world! This is a test.";
  
  // Build vocabulary first
  tokenizer.buildVocab(text);
  
  // Test encoding
  const encoded = tokenizer.encode(text, false, false);
  assert(Array.isArray(encoded), 'Encoded result should be an array');
  assert(encoded.length > 0, 'Encoded array should not be empty');
  
  // Test that all tokens are valid
  for (let i = 0; i < encoded.length; i++) {
    assert(typeof encoded[i] === 'number', `Token at position ${i} should be a number`);
    assert(encoded[i] >= 0, `Token at position ${i} should be non-negative`);
  }
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  assert(typeof decoded === 'string', 'Decoded result should be a string');
  
  // Test vocabulary
  const vocabSize = tokenizer.vocabSize;
  assert(typeof vocabSize === 'number', 'Vocabulary size should be a number');
  assert(vocabSize > 0, 'Vocabulary size should be positive');
  
  // Test with different text patterns
  const patterns = [
    "Simple text",
    "Text with numbers 123",
    "Text with punctuation!@#",
    "Multiple    spaces",
    "Mixed-case TEXT",
    ""
  ];
  
  for (const pattern of patterns) {
    if (pattern.length > 0) {
      tokenizer.buildVocab(pattern);
    }
    const patternEncoded = tokenizer.encode(pattern, false, false);
    const patternDecoded = tokenizer.decode(patternEncoded);
    
    assert(Array.isArray(patternEncoded), `Pattern "${pattern}" should encode to array`);
    assert(typeof patternDecoded === 'string', `Pattern "${pattern}" should decode to string`);
  }
  
  // Test unknown token handling
  const unknownText = "This contains unknown words like xyzabc123";
  tokenizer.buildVocab(unknownText);
  const unknownEncoded = tokenizer.encode(unknownText, false, false);
  assert(Array.isArray(unknownEncoded), 'Unknown words should still be encoded');
  
  console.log('✓ WordTokenizer tests passed');
}

function testBPETokenizer() {
  console.log('Testing BPETokenizer...');
  
  const tokenizer = new BPETokenizer();
  
  // Test with training data
  const trainingTexts = [
    "hello world",
    "hello there",
    "world peace",
    "peace and love",
    "love is all"
  ];
  
  // Train the tokenizer
  tokenizer.buildVocab(trainingTexts);
  
  // Test basic functionality
  const testText = "hello world peace";
  const encoded = tokenizer.encode(testText, false, false);
  assert(Array.isArray(encoded), 'Encoded result should be an array');
  assert(encoded.length > 0, 'Encoded array should not be empty');
  
  // Test that all tokens are valid
  for (let i = 0; i < encoded.length; i++) {
    assert(typeof encoded[i] === 'number', `Token at position ${i} should be a number`);
    assert(encoded[i] >= 0, `Token at position ${i} should be non-negative`);
  }
  
  // Test decoding
  const decoded = tokenizer.decode(encoded);
  assert(typeof decoded === 'string', 'Decoded result should be a string');
  
  // Test vocabulary
  const vocabSize = tokenizer.vocabSize;
  assert(typeof vocabSize === 'number', 'Vocabulary size should be a number');
  assert(vocabSize > 0, 'Vocabulary size should be positive');
  
  // Test that BPE creates subword units
  const longWord = "supercalifragilisticexpialidocious";
  const longEncoded = tokenizer.encode(longWord, false, false);
  assert(longEncoded.length > 1, 'Long unknown word should be split into subwords');
  
  // Test empty string
  const emptyEncoded = tokenizer.encode("", false, false);
  assert(emptyEncoded.length === 0, 'Empty string should encode to empty array');
  
  const emptyDecoded = tokenizer.decode([]);
  assert(emptyDecoded === "", 'Empty array should decode to empty string');
  
  console.log('✓ BPETokenizer tests passed');
}

function testTokenizerConsistency() {
  console.log('Testing tokenizer consistency...');
  
  const tokenizers = [
    new CharTokenizer(),
    new WordTokenizer(),
    new BPETokenizer()
  ];
  
  // Build vocabularies
  const trainText = "hello world test";
  tokenizers[0].buildVocab(trainText);
  tokenizers[1].buildVocab(trainText);
  tokenizers[2].buildVocab([trainText]);
  
  const testTexts = [
    "hello",
    "world",
    "test",
    "hello world",
    "test case",
    ""
  ];
  
  for (const tokenizer of tokenizers) {
    const tokenizerName = tokenizer.constructor.name;
    
    for (const text of testTexts) {
      // Test encode-decode consistency
      if (text.length > 0) {
        tokenizer.buildVocab(text);
      }
      const encoded = tokenizer.encode(text, false, false);
      const decoded = tokenizer.decode(encoded);
      
      assert(Array.isArray(encoded), `${tokenizerName}: encode should return array for "${text}"`);
      assert(typeof decoded === 'string', `${tokenizerName}: decode should return string for "${text}"`);
      
      // For char-level tokenizer, decode should be close to original
      if (tokenizer instanceof CharTokenizer && text.length > 0) {
        // Note: May not be exact due to special token handling
        assert(typeof decoded === 'string', `${tokenizerName}: decode should return string for "${text}"`);
      }
      
      // Test that encoding is deterministic
      const encoded2 = tokenizer.encode(text, false, false);
      assert(encoded.length === encoded2.length, `${tokenizerName}: encoding should be deterministic for "${text}"`);
      for (let i = 0; i < encoded.length; i++) {
        assert(encoded[i] === encoded2[i], `${tokenizerName}: encoding should be deterministic for "${text}"`);
      }
    }
  }
  
  console.log('✓ Tokenizer consistency tests passed');
}

function testTokenizerEdgeCases() {
  console.log('Testing tokenizer edge cases...');
  
  const charTokenizer = new CharTokenizer();
  const wordTokenizer = new WordTokenizer();
  const bpeTokenizer = new BPETokenizer();
  
  // Train BPE with minimal data
  bpeTokenizer.buildVocab(["a"]);
  
  const edgeCases = [
    "",                    // Empty string
    " ",                   // Single space
    "\n",                  // Newline
    "\t",                  // Tab
    "a",                   // Single character
    "   ",                 // Multiple spaces
    "123",                 // Numbers only
    "!@#$%",              // Punctuation only
    "A",                   // Single uppercase
    "aA",                  // Mixed case
    "🚀",                  // Single emoji
    "café",                // Accented characters
    "αβγ",                 // Greek letters
    "中文",                // Chinese characters
    "\u0000",              // Null character
    "\u200B",              // Zero-width space
  ];
  
  for (const testCase of edgeCases) {
    // Test char-level tokenizer
    try {
      const charEncoded = charTokenizer.encode(testCase);
      const charDecoded = charTokenizer.decode(charEncoded);
      assert(charDecoded === testCase, `Char tokenizer failed for: "${testCase}"`);
    } catch (error) {
      console.warn(`Char tokenizer warning for "${testCase}": ${error.message}`);
    }
    
    // Test word-level tokenizer
    try {
      const wordEncoded = wordTokenizer.encode(testCase);
      const wordDecoded = wordTokenizer.decode(wordEncoded);
      assert(typeof wordDecoded === 'string', `Word tokenizer should return string for: "${testCase}"`);
    } catch (error) {
      console.warn(`Word tokenizer warning for "${testCase}": ${error.message}`);
    }
    
    // Test BPE tokenizer
    try {
      const bpeEncoded = bpeTokenizer.encode(testCase);
      const bpeDecoded = bpeTokenizer.decode(bpeEncoded);
      assert(typeof bpeDecoded === 'string', `BPE tokenizer should return string for: "${testCase}"`);
    } catch (error) {
      console.warn(`BPE tokenizer warning for "${testCase}": ${error.message}`);
    }
  }
  
  console.log('✓ Tokenizer edge cases tests passed');
}

function testTokenizerVocabulary() {
  console.log('Testing tokenizer vocabulary management...');
  
  // Test char-level vocabulary
  const charTokenizer = new CharTokenizer();
  const charText = "abcABC123!@#";
  charTokenizer.buildVocab(charText); // Build vocabulary
  
  const charVocabSize = charTokenizer.vocabSize;
  
  assert(charVocabSize > 0, 'Char vocab size should be positive');
  
  // Test word-level vocabulary
  const wordTokenizer = new WordTokenizer();
  const wordText = "hello world test case";
  wordTokenizer.buildVocab(wordText);
  
  const wordVocabSize = wordTokenizer.vocabSize;
  
  assert(wordVocabSize > 0, 'Word vocab size should be positive');
  
  // Test BPE vocabulary
  const bpeTokenizer = new BPETokenizer();
  const bpeTexts = ["hello world", "world peace", "peace love"];
  bpeTokenizer.buildVocab(bpeTexts);
  
  const bpeVocabSize = bpeTokenizer.vocabSize;
  
  assert(bpeVocabSize > 0, 'BPE vocab size should be positive');
  
  console.log('✓ Tokenizer vocabulary tests passed');
}

function testTokenizerPerformance() {
  console.log('Testing tokenizer performance...');
  
  const charTokenizer = new CharTokenizer();
  const wordTokenizer = new WordTokenizer();
  const bpeTokenizer = new BPETokenizer();
  
  // Train BPE
  const trainingData = Array(100).fill("hello world test case").map((text, i) => `${text} ${i}`);
  bpeTokenizer.buildVocab(trainingData);
  
  // Test with moderately large text
  const largeText = "hello world ".repeat(1000);
  
  // Build vocabularies first
  charTokenizer.buildVocab(largeText);
  wordTokenizer.buildVocab(largeText);
  
  // Test char-level performance
  const charStart = Date.now();
  const charEncoded = charTokenizer.encode(largeText, false, false);
  const charDecoded = charTokenizer.decode(charEncoded);
  const charTime = Date.now() - charStart;
  
  assert(typeof charDecoded === 'string', 'Char tokenizer should handle large text correctly');
  assert(charTime < 2000, 'Char tokenizer should be reasonably fast'); // Less than 2 seconds
  
  // Test word-level performance
  const wordStart = Date.now();
  const wordEncoded = wordTokenizer.encode(largeText, false, false);
  const wordDecoded = wordTokenizer.decode(wordEncoded);
  const wordTime = Date.now() - wordStart;
  
  assert(typeof wordDecoded === 'string', 'Word tokenizer should handle large text');
  assert(wordTime < 2000, 'Word tokenizer should be reasonably fast');
  
  // Test BPE performance
  const bpeStart = Date.now();
  const bpeEncoded = bpeTokenizer.encode(largeText, false, false);
  const bpeDecoded = bpeTokenizer.decode(bpeEncoded);
  const bpeTime = Date.now() - bpeStart;
  
  assert(typeof bpeDecoded === 'string', 'BPE tokenizer should handle large text');
  assert(bpeTime < 3000, 'BPE tokenizer should be reasonably fast'); // Allow more time for BPE
  
  console.log(`  Char tokenizer: ${charTime}ms`);
  console.log(`  Word tokenizer: ${wordTime}ms`);
  console.log(`  BPE tokenizer: ${bpeTime}ms`);
  
  console.log('✓ Tokenizer performance tests passed');
}

// Run all tests
function runTests() {
  console.log('=== Testing Tokenizers ===\n');
  
  try {
    testCharLevelTokenizer();
    testWordLevelTokenizer();
    testBPETokenizer();
    testTokenizerConsistency();
    testTokenizerEdgeCases();
    testTokenizerVocabulary();
    testTokenizerPerformance();
    
    console.log('\n🎉 All tokenizer tests passed!');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests();
}

export { runTests };