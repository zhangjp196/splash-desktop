//! The chat format of each model family Splash serves: how its model
//! frames reasoning and lays out tool calls, which the server reads its
//! output by and constrains it to, and the tokens it shares with the
//! engine and the tokenizer. Ported from `server/chat_format.py`.

/// The tags every family's template frames reasoning and calls with, each
/// one token of its tokenizer: a think block, and a call's opening and
/// close.
pub const THINK_START: &str = "\u{003c}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}";
pub const THINK_END: &str = "\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}";
pub const TOOL_CALL_OPEN: &str = "\u{003c}\u{0074}\u{006f}\u{006f}\u{006c}\u{005f}\u{0063}\u{0061}\u{006c}\u{006c}\u{003e}";
pub const TOOL_CALL_CLOSE: &str = "\u{003c}\u{002f}\u{0074}\u{006f}\u{006f}\u{006c}\u{005f}\u{0063}\u{0061}\u{006c}\u{006c}\u{003e}";

/// How a family's chat template lays a tool call out between
/// TOOL_CALL_OPEN and TOOL_CALL_CLOSE: name_open, the function's name and
/// name_close, then each argument, then arguments_close. An argument is
/// key_open, its name and key_close, then value_open, its value and
/// value_close. Grammars write calls that way, and the projector reads
/// them that way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallSyntax {
    pub name_open: &'static str,
    pub name_close: &'static str,
    pub key_open: &'static str,
    pub key_close: &'static str,
    pub value_open: &'static str,
    pub value_close: &'static str,
    pub arguments_close: &'static str,
    /// Whether each of its tags is a token of the family's tokenizer, as
    /// TOOL_CALL_OPEN and TOOL_CALL_CLOSE are in every family's, which
    /// grammars then write as that token; otherwise they write the tags as
    /// text.
    pub tags_are_tokens: bool,
}

impl ToolCallSyntax {
    /// The text a call opens with, up to its function's name.
    pub fn call_open(&self) -> String {
        format!("{TOOL_CALL_OPEN}{}", self.name_open)
    }

    /// What follows TOOL_CALL_OPEN in a call of tool `name`, up to its
    /// arguments.
    pub fn function_opening(&self, name: &str) -> String {
        format!("{}{name}{}", self.name_open, self.name_close)
    }

    /// The character a function's name ends at: the first of name_close,
    /// or where it has none, of the tags that follow the name, each of
    /// which opens with it.
    pub fn name_end(&self) -> char {
        let candidate = if self.name_close.is_empty() {
            self.key_open
        } else {
            self.name_close
        };
        candidate.chars().next().unwrap_or('\0')
    }

    /// The tags that may follow an argument: the next one's opening, the
    /// arguments' close without the space its layout sets around it, and
    /// the call's close a model may write without the arguments'.
    pub fn argument_tags(&self) -> Vec<&'static str> {
        [self.key_open, self.arguments_close.trim(), TOOL_CALL_CLOSE]
            .into_iter()
            .filter(|tag| !tag.is_empty())
            .collect()
    }

    /// Where a value ends: at its close where a tag that may follow an
    /// argument follows it, so a value may hold the tags in any other
    /// order.
    pub fn value_ends(&self) -> Vec<String> {
        self.argument_tags()
            .iter()
            .map(|tag| format!("{}{tag}", self.value_close))
            .collect()
    }
}

/// Qwen3-Coder's XML, as Qwen's templates lay a call out: a tag around the
/// function and around each parameter, on lines of their own.
pub const QWEN_XML: ToolCallSyntax = ToolCallSyntax {
    name_open: "\n<function=",
    name_close: ">\n",
    key_open: "<parameter=",
    key_close: ">",
    value_open: "\n",
    value_close: "\n</parameter>\n",
    arguments_close: "</function>\n",
    tags_are_tokens: false,
};

/// GLM-4.7's argument tags, as Flint's template lays a call out: the
/// function's name right after TOOL_CALL_OPEN, then a key tag and a value
/// tag for each argument, with nothing between the tags, each tag one
/// token. The tags are Unicode group separators U+11D3..U+11D6.
pub const GLM_ARGS: ToolCallSyntax = ToolCallSyntax {
    name_open: "",
    name_close: "",
    key_open: "<arg_key>",
    key_close: "</arg_key>",
    value_open: "<arg_value>",
    value_close: "</arg_value>",
    arguments_close: "",
    tags_are_tokens: true,
};

/// What the server reads, writes and checks in a family's tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatFormat {
    /// The model's vocabulary, the rows of its logits that a constraint's
    /// masks cover: its target layout's vocabularySize, which may exceed
    /// its tokenizer's.
    pub vocabulary: u32,
    /// The engine's stop tokens, each with the text its tokenizer encodes
    /// as that token; the tokenizer's EOS is one.
    pub stop_tokens: Vec<(&'static str, u32)>,
    /// The token of THINK_END, which closes the reasoning.
    pub think_end_id: u32,
    /// The token of THINK_START where the model opens its reasoning
    /// itself, as its first token; `None` where the generation prompt
    /// opens it, as Qwen's do by the request's reasoning effort.
    pub think_start_id: Option<u32>,
    /// The token of TOOL_CALL_OPEN.
    pub call_open_id: u32,
    pub call_syntax: ToolCallSyntax,
    /// The template's special token after which prompt tokenization
    /// reuses a cached prompt's tokens.
    pub message_boundary: &'static str,
}

impl ChatFormat {
    pub fn stop_ids(&self) -> Vec<u32> {
        self.stop_tokens.iter().map(|(_text, id)| *id).collect()
    }

    /// Whether a request turns the model's reasoning on and off: its
    /// reasoning effort or enable_thinking opens the think block in the
    /// generation prompt, or leaves it out. A model that opens its
    /// reasoning itself decides each turn whether to reason, whatever the
    /// request says.
    pub fn reasoning_switch(&self) -> bool {
        self.think_start_id.is_none()
    }

    /// Whether a call's opening also ends the reasoning where a call may
    /// follow. A Qwen model may call a tool in the think block its
    /// generation prompt opened, without closing it; a model that opens
    /// its reasoning itself closes it before it calls one, so there a
    /// call's opening is reasoning, as Flint's own parser reads it.
    pub fn calls_end_reasoning(&self) -> bool {
        self.think_start_id.is_none()
    }
}

/// The Qwen families': a padded vocabulary, the think close and the turn
/// end as stop tokens, and a generation prompt that opens the think block
/// when the request thinks.
/// The Qwen families' format.
pub fn qwen() -> ChatFormat {
    ChatFormat {
    vocabulary: 248320,
    stop_tokens: vec![
        ("\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}", 248044),
        ("<|im_end|>", 248046),
    ],
    think_end_id: 248069,
    think_start_id: None,
    call_open_id: 248058,
    call_syntax: QWEN_XML,
    message_boundary: "<|im_end|>",
    }
}

/// Flint's: a vocabulary its tokenizer fills, the end-of-text and the role
/// tokens as stop tokens, and a generation prompt that leaves reasoning to
/// the model. Every chat prompt ends with the assistant's role token.
/// Flint's format.
pub fn flint() -> ChatFormat {
    ChatFormat {
    vocabulary: 150144,
    stop_tokens: vec![
        ("<|end_of_text|>", 150001),
        ("\u{003c}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}", 150003),
        ("\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}", 150005),
        ("<|developer|>", 150018),
    ],
    think_end_id: 150007,
    think_start_id: Some(150006),
    call_open_id: 150010,
    call_syntax: GLM_ARGS,
    message_boundary: "\u{003c}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}",
    }
}
