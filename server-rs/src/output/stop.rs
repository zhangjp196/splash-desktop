//! The first of a request's stop sequences that the output contains, as
//! the output arrives. Ported from `server/output.py`'s `StopSearch` and
//! `hold_partial`.

/// `text` split before its longest end that begins one of `markers`, which
/// more text may complete.
pub fn hold_partial<'a>(text: &'a str, markers: &[&str]) -> (&'a str, &'a str) {
    let longest = markers.iter().map(|marker| marker.len()).max().unwrap_or(0);
    let bytes = text.len();
    for length in (1..=(bytes.min(longest.saturating_sub(1)))).rev() {
        let (head, tail) = text.split_at(bytes - length);
        if markers
            .iter()
            .any(|marker| length < marker.len() && marker.starts_with(tail))
        {
            return (head, tail);
        }
    }
    (text, "")
}

/// For each prefix of `text`, the length of its longest proper end that is
/// also a prefix of `text` (the Knuth-Morris-Pratt fallback).
fn prefix_function(text: &str) -> Vec<usize> {
    let characters: Vec<char> = text.chars().collect();
    let mut lengths = vec![0usize; characters.len()];
    let mut length = 0usize;
    for index in 1..characters.len() {
        while length > 0 && characters[index] != characters[length] {
            length = lengths[length - 1];
        }
        if characters[index] == characters[length] {
            length += 1;
        }
        lengths[index] = length;
    }
    lengths
}

/// The first of a request's stop sequences that the output contains, as
/// the output arrives, and how much of its end may begin one. Each stop
/// keeps the length of the output's longest end that begins it (a
/// Knuth-Morris-Pratt state), so a character costs the same however long
/// the stops are: rescanning every length of a 50,000-character stop on
/// every token would copy about a gigabyte per token.
pub struct StopSearch {
    stops: Vec<Vec<char>>,
    fallbacks: Vec<Vec<usize>>,
    states: Vec<usize>,
}

/// What `search` found: the stop the output contains first, and how many
/// characters before the end of the searched text it begins (more than the
/// text holds where it began earlier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopMatch {
    pub stop: String,
    pub back: usize,
}

impl StopSearch {
    pub fn new(stops: &[&str]) -> StopSearch {
        StopSearch {
            stops: stops.iter().map(|stop| stop.chars().collect()).collect(),
            fallbacks: stops.iter().map(|stop| prefix_function(stop)).collect(),
            states: vec![0; stops.len()],
        }
    }

    /// The stop that the output, `text` appended, contains first, and how
    /// many characters before the end of `text` it begins; `None` where it
    /// contains none.
    pub fn search(&mut self, text: &str) -> Option<StopMatch> {
        let characters: Vec<char> = text.chars().collect();
        let mut first: Option<StopMatch> = None;
        for index in 0..self.stops.len() {
            let stop = &self.stops[index];
            let fallback = &self.fallbacks[index];
            let mut state = self.states[index];
            'stop: for (position, character) in characters.iter().enumerate() {
                while state > 0 && stop[state] != *character {
                    state = fallback[state - 1];
                }
                if stop[state] == *character {
                    state += 1;
                }
                if state == stop.len() {
                    let back = characters.len() - position - 1 + stop.len();
                    if first.as_ref().map(|found| back > found.back).unwrap_or(true) {
                        first = Some(StopMatch {
                            stop: stop.iter().collect(),
                            back,
                        });
                    }
                    break 'stop;
                }
            }
            self.states[index] = state;
        }
        first
    }

    /// The length of the output's longest end that begins a stop.
    pub fn held(&self) -> usize {
        self.states.iter().copied().max().unwrap_or(0)
    }
}
