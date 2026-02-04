extern crate cbindgen;

use std::env;
use std::path::Path;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src_path = Path::new(&crate_dir).join("src");
    let output_path = Path::new(&crate_dir).join("include").join("egg_bridge.h");
    
    // Use with_src instead of with_crate to avoid parsing Cargo.toml metadata
    // which can fail with workspace-level dependencies
    cbindgen::Builder::new()
        .with_src(src_path.join("lib.rs"))
        .with_language(cbindgen::Language::C)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&output_path);
}
