//! Emit the JSON Schema for the binding types. Run with: --features schema
//! cargo run -p gfold-ffi --features schema --bin emit-schema > ../schemas/gfold.schema.json
fn main() {
    let schema = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "gfold",
        "definitions": {
            "Config": schemars::schema_for!(gfold_core::config::Config).schema,
            "Trajectory": schemars::schema_for!(gfold_core::solve::Trajectory).schema,
        }
    });
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}
