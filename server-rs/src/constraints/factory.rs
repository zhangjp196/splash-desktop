//! The compiled matchers and the token masks they produce, ported from
//! `server/constraints.py`'s `ConstraintFactory` and `TokenConstraint`.

use std::sync::{Arc, Mutex};

use llguidance::api::GrammarWithLexer;
use llguidance::{Matcher, ParserFactory};
use toktrie::SimpleVob;

use crate::chat_format::ChatFormat;
use crate::constraints::tokenizer_env::ConstraintTokenizer;
use crate::support::lru::LruCache;

/// A grammar the factory compiles: GBNF and regex go through llguidance's
/// own converters, which `GrammarWithLexer` builds from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Grammar {
    Regex(String),
    Gbnf(String),
    Json(serde_json::Value),
}

impl Grammar {
    fn to_grammar_with_lexer(&self) -> Result<GrammarWithLexer, GrammarError> {
        match self {
            Grammar::Regex(regex) => Ok(GrammarWithLexer::from_regex(regex)),
            // A GBNF grammar reads as llguidance's own grammar language,
            // which TopLevelGrammar::from_lark_or_grammar_list parses.
            Grammar::Gbnf(gbnf) => {
                let top = llguidance::api::TopLevelGrammar::from_lark_or_grammar_list(gbnf)
                    .map_err(|error| GrammarError(error.to_string()))?;
                Ok(top.grammars.into_iter().next().unwrap_or_default())
            }
            Grammar::Json(schema) => Ok(GrammarWithLexer::from_json_schema(schema.clone())),
        }
    }
}

/// A grammar the converter refused, or one whose compilation failed: the
/// request's error, not the server's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarError(pub String);

/// The compiled matchers, within a budget of grammar bytes.
pub struct ConstraintFactory {
    factory: ParserFactory,
    chat_format: ChatFormat,
    cache: Mutex<LruCache<Grammar, Arc<Mutex<Matcher>>>>,
    pub hits: Mutex<usize>,
    pub misses: Mutex<usize>,
}

/// The cache's bounds, as the Python's are.
const CACHE_SIZE: usize = 32;
const CACHE_SOURCE_BYTES: usize = 8 * 1024 * 1024;

impl ConstraintFactory {
    /// The factory over one model's grammar tokenizer.
    pub fn new(
        tokenizer: &ConstraintTokenizer,
        chat_format: &ChatFormat,
    ) -> Result<ConstraintFactory, String> {
        let factory = ParserFactory::new_simple(&tokenizer.env)
            .map_err(|error| format!("grammar factory: {error}"))?;
        Ok(ConstraintFactory {
            factory,
            chat_format: chat_format.clone(),
            cache: Mutex::new(LruCache::new(CACHE_SOURCE_BYTES, Some(CACHE_SIZE))),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        })
    }

    /// Compile one grammar, or the error it raises: a grammar can compile
    /// and still exceed the parser's limits where generation reaches a
    /// construct.
    pub fn create(&self, grammar: &Grammar) -> Result<TokenConstraint, GrammarError> {
        let matcher = self.matcher(grammar)?;
        // Each constraint consumes its matcher independently: the cached
        // one is deep-cloned, as the Python's deep_copy does.
        let cloned = matcher.lock().expect("matcher lock").deep_clone();
        Ok(TokenConstraint::new(Arc::new(Mutex::new(cloned)), self.chat_format.clone()))
    }

    fn matcher(
        &self,
        grammar: &Grammar,
    ) -> Result<Arc<Mutex<Matcher>>, GrammarError> {
        {
            let mut cache = self.cache.lock().expect("cache lock");
            if let Some(matcher) = cache.get(grammar) {
                *self.hits.lock().expect("hits lock") += 1;
                return Ok(matcher.clone());
            }
        }
        *self.misses.lock().expect("misses lock") += 1;
        let compiled = self.compile(grammar)?;
        let size = match grammar {
            Grammar::Regex(text) | Grammar::Gbnf(text) => text.len(),
            Grammar::Json(schema) => serde_json::to_string(schema).map(|s| s.len()).unwrap_or(0),
        };
        let mut cache = self.cache.lock().expect("cache lock");
        // Oversized grammars remain usable without displacing the cache.
        if size <= CACHE_SOURCE_BYTES {
            cache.put(grammar.clone(), compiled.clone(), size);
        }
        Ok(compiled)
    }

    fn compile(&self, grammar: &Grammar) -> Result<Arc<Mutex<Matcher>>, GrammarError> {
        let with_lexer = grammar.to_grammar_with_lexer()?;
        let parser = self
            .factory
            .create_parser(llguidance::api::TopLevelGrammar::from_grammar(
                with_lexer.clone(),
            ))
            .map_err(|error| GrammarError(error.to_string()))?;
        let matcher = Matcher::new(Ok(parser));
        if matcher.is_error() {
            return Err(GrammarError(matcher.get_error().unwrap_or_default()));
        }
        Ok(Arc::new(Mutex::new(matcher)))
    }
}

/// One request's constraint: the matcher its grammar compiled, the tokens
/// it has consumed, and the masks it produces.
pub struct TokenConstraint {
    matcher: Arc<Mutex<Matcher>>,
    chat_format: ChatFormat,
    committed: Mutex<Vec<Vec<u32>>>,
}

/// A mask request simulates at most the engine's target verify rows (the
/// pending anchor and seven draft proposals) and takes a mask before and
/// after each.
const MAX_ROWS: usize = 9;

impl TokenConstraint {
    fn new(matcher: Arc<Mutex<Matcher>>, chat_format: ChatFormat) -> TokenConstraint {
        TokenConstraint {
            matcher,
            chat_format,
            committed: Mutex::new(Vec::new()),
        }
    }

    /// The tokens the grammar has generated, which the next mask consumes.
    pub fn commit(&self, token_ids: &[u32]) -> Result<(), String> {
        if token_ids
            .iter()
            .any(|token| *token >= self.chat_format.vocabulary)
        {
            return Err("generated token is out of range".into());
        }
        self.committed
            .lock()
            .expect("committed lock")
            .push(token_ids.to_vec());
        Ok(())
    }

    /// Another constraint over the same grammar, for another choice of one
    /// request: the choices consume it independently.
    pub fn clone_constraint(&self) -> TokenConstraint {
        let matcher = self.matcher.lock().expect("matcher lock").deep_clone();
        TokenConstraint::new(Arc::new(Mutex::new(matcher)), self.chat_format.clone())
    }

    /// The masks for one mask request: one row before and after each of
    /// the simulated tokens, each row the vocabulary's bits, a bit set
    /// where the token is allowed. The rows past the simulated tokens the
    /// grammar could not consume repeat the last valid row, as the
    /// Python's do.
    pub fn masks(&self, simulation_tokens: &[u32]) -> Result<Vec<u8>, String> {
        if simulation_tokens.len() >= MAX_ROWS {
            return Err("too many simulation tokens".into());
        }
        self.consume_committed()?;
        let vocabulary = self.chat_format.vocabulary as usize;
        let mut matcher = self.matcher.lock().expect("matcher lock");
        let valid_count = matcher
            .validate_tokens(simulation_tokens)
            .map_err(|error| format!("grammar rejected the simulation: {error}"))?;
        let mut rows: Vec<SimpleVob> = Vec::new();
        for row in 0..=valid_count {
            let mut step_matcher = matcher.deep_clone();
            let prefix = &simulation_tokens[..row];
            if !prefix.is_empty() {
                step_matcher
                    .consume_tokens(prefix)
                    .map_err(|error| format!("grammar rejected the simulation: {error}"))?;
            }
            // A matcher that has stopped allows its ends, as the Python's
            // fill does.
            let mask = step_matcher
                .compute_mask_or_eos()
                .map_err(|error| format!("grammar mask failed: {error}"))?;
            rows.push(mask);
        }
        drop(matcher);
        // The rows past the grammar's acceptance repeat the last valid
        // row's bits.
        while rows.len() < simulation_tokens.len() + 1 {
            let last = rows[rows.len() - 1].clone();
            rows.push(last);
        }
        // The words of each row, little-endian u32s, a bit set where the
        // token is allowed.
        let words_per_row = (vocabulary + 31) / 32;
        let mut out = Vec::with_capacity(rows.len() * words_per_row * 4);
        for row in &rows {
            let words = row.as_slice();
            for word_index in 0..words_per_row {
                let word = words.get(word_index).copied().unwrap_or(0);
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// The tokens generated after the last mask, consumed.
    fn consume_committed(&self) -> Result<(), String> {
        let batches: Vec<Vec<u32>> =
            std::mem::take(&mut self.committed.lock().expect("committed lock"));
        let mut matcher = self.matcher.lock().expect("matcher lock");
        for token_ids in batches {
            // LLGuidance's bulk API rejects EOS after a NoExtension stop.
            let stopped_eos = token_ids.len() == 1
                && self.chat_format.stop_ids().contains(&token_ids[0])
                && !matcher.is_error()
                && matcher.is_stopped()
                && matcher.is_accepting().unwrap_or(false);
            let valid = if stopped_eos {
                matcher.consume_token(token_ids[0])
            } else {
                matcher.consume_tokens(&token_ids)
            };
            valid.map_err(|error| {
                // The first line is the reason; the rest dump the parser
                // state, output included.
                error
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or("grammar rejected the generated token")
                    .to_string()
            })?;
        }
        Ok(())
    }
}
