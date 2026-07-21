fn main() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../src/generated/bindings.generated.ts");
    if let Err(error) = app_lib::export_bindings(&path) {
        eprintln!("failed to export native bindings: {error}");
        std::process::exit(1);
    }
}
