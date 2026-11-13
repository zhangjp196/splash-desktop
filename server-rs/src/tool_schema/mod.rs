//! Tool and response-format schema normalization and llguidance grammars,
//! ported from `server/tool_schema.py` and the parts of
//! `server/schema_validation.py` it and the response validation use
//! (module 11).
//!
//! The pieces, by file: [`schema`] reads schemas — their subschema walk,
//! depth and reference checks, the type and literal-text readers, and the
//! argument projection a tool's fields frame through; [`policy`] holds
//! [`ToolPolicy`], a request's tool calls' names and schemas, and
//! [`normalize_tools`]; [`grammar`] builds the llguidance grammars the
//! constraint factory compiles; [`validate`] checks schemas against their
//! dialect's meta-schema and answers against their schema, on the
//! `jsonschema` crate.

pub mod grammar;
pub mod policy;
pub mod schema;
pub mod validate;

pub use grammar::{
    json_grammar, tool_grammar, whitespace_rule, MAX_GRAMMAR_BOUND, MAX_WHITESPACE,
    PATTERN_CHECK_CACHE_BYTES, PATTERN_CHECK_CACHE_SIZE,
};
pub use policy::{normalize_response_format, normalize_tools, ToolParameterTypes, ToolPolicy};
pub use schema::{
    is_tag_name, schema_types, tool_argument_schema, MAX_FRAMED_SCHEMA_BYTES, MAX_NAME_LENGTH,
    MAX_SCHEMA_DEPTH, NAME_SPACE, TAG_NAME_RULE,
};
pub use validate::{
    build_validator, check_schema, validate_response_content, ResponseValidator,
    SchemaError, ValidationFailure,
};
