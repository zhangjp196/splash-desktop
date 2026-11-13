use minijinja::{Environment, Value};

fn to_minijinja(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::from(()),
        serde_json::Value::Bool(b) => Value::from(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::from(i) }
            else if let Some(u) = n.as_u64() { Value::from(u) }
            else { Value::from(n.as_f64().unwrap_or(0.0)) }
        }
        serde_json::Value::String(s) => Value::from(s.clone()),
        serde_json::Value::Array(items) => {
            Value::from(items.iter().map(to_minijinja).collect::<Vec<_>>())
        }
        serde_json::Value::Object(map) => Value::from_pairs(
            map.iter().map(|(k, v)| (k.clone(), to_minijinja(v))),
        ),
    }
}

fn main() {
    let mut env = Environment::new();
    env.add_template("j", "{{ x|tojson }}").unwrap();
    let cases = vec![
        serde_json::json!({"a": 1, "b": [1, 2.5], "c": null, "d": true, "e": "x\"y\n"}),
        serde_json::json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]),
    ];
    for case in cases {
        let out = env.get_template("j").unwrap().render(minijinja::context! {
            x => to_minijinja(&case),
        }).unwrap();
        println!("minijinja: {out}");
    }
}
