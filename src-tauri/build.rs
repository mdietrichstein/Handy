fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_apple_intelligence_bridge();

    build_voxtral();

    generate_tray_translations();

    tauri_build::build()
}

/// Build the voxtral C library for speech-to-text inference.
///
/// Compiles all voxtral C source files into a static library.
/// On Linux: links Vulkan + OpenBLAS for GPU acceleration.
/// On macOS aarch64: links Metal + Accelerate for GPU acceleration.
/// On other platforms: links BLAS only (CPU).
fn build_voxtral() {
    use std::env;
    use std::path::Path;

    let voxtral_dir = Path::new("voxtral");
    println!("cargo:rerun-if-changed=voxtral/");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Common C source files (no main.c, no mic files)
    let c_sources = [
        "voxtral.c",
        "voxtral_encoder.c",
        "voxtral_decoder.c",
        "voxtral_kernels.c",
        "voxtral_audio.c",
        "voxtral_tokenizer.c",
        "voxtral_safetensors.c",
    ];

    let mut build = cc::Build::new();
    build
        .warnings(false) // voxtral uses -Wextra style, lots of noise in cc
        .opt_level_str("3")
        .flag("-ffast-math");

    // Add common sources
    for src in &c_sources {
        build.file(voxtral_dir.join(src));
    }

    // Platform-specific configuration
    if target_os == "linux" {
        // Linux: Vulkan + OpenBLAS
        build
            .define("USE_BLAS", None)
            .define("USE_OPENBLAS", None)
            .define("USE_VULKAN", None)
            .define("USE_GPU", None)
            .flag("-Wno-missing-field-initializers");

        // Check for local OpenBLAS with AVX-512
        let local_blas = Path::new("openblas-local");
        if local_blas.join("lib/libopenblas.so").exists() {
            build.include(local_blas.join("include"));
        } else {
            build.include("/usr/include/openblas");
        }

        // Add Vulkan source
        build.file(voxtral_dir.join("voxtral_vulkan.c"));

        // Link libraries
        if local_blas.join("lib/libopenblas.so").exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                local_blas.join("lib").display()
            );
        }
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-lib=vulkan");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    } else if target_os == "macos" && target_arch == "aarch64" {
        // macOS Apple Silicon: Metal + Accelerate
        build
            .define("USE_BLAS", None)
            .define("USE_METAL", None)
            .define("USE_GPU", None)
            .define("ACCELERATE_NEW_LAPACK", None);

        // Generate the embedded Metal shader source header
        let out_dir = env::var("OUT_DIR").unwrap();
        let shader_source = std::fs::read_to_string(voxtral_dir.join("voxtral_shaders.metal"))
            .expect("Failed to read voxtral_shaders.metal");
        let shader_header_path =
            Path::new(&out_dir).join("voxtral_shaders_source.h");

        // Generate xxd-style C array
        let mut header = String::new();
        header.push_str("unsigned char voxtral_shaders_metal[] = {\n");
        for (i, byte) in shader_source.as_bytes().iter().enumerate() {
            if i > 0 {
                header.push_str(", ");
            }
            if i % 12 == 0 {
                header.push_str("\n  ");
            }
            header.push_str(&format!("0x{:02x}", byte));
        }
        header.push_str("\n};\n");
        header.push_str(&format!(
            "unsigned int voxtral_shaders_metal_len = {};\n",
            shader_source.len()
        ));
        std::fs::write(&shader_header_path, header)
            .expect("Failed to write shader header");

        // Add Metal ObjC source with include path for generated header
        build.include(&out_dir);

        // Metal .m file needs to be compiled separately as ObjC
        let mut metal_build = cc::Build::new();
        metal_build
            .warnings(false)
            .opt_level_str("3")
            .flag("-ffast-math")
            .flag("-fobjc-arc")
            .define("USE_BLAS", None)
            .define("USE_METAL", None)
            .define("USE_GPU", None)
            .define("ACCELERATE_NEW_LAPACK", None)
            .include(voxtral_dir)
            .include(&out_dir)
            .file(voxtral_dir.join("voxtral_metal.m"));
        metal_build.compile("voxtral_metal");

        // Link frameworks
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=m");
    } else if target_os == "macos" {
        // macOS x86_64: Accelerate only (no Metal GPU)
        build
            .define("USE_BLAS", None)
            .define("ACCELERATE_NEW_LAPACK", None);

        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=m");
    } else if target_os == "windows" {
        // Windows: Vulkan + OpenBLAS (TODO: test this)
        build
            .define("USE_BLAS", None)
            .define("USE_OPENBLAS", None)
            .define("USE_VULKAN", None)
            .define("USE_GPU", None);

        build.file(voxtral_dir.join("voxtral_vulkan.c"));

        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-lib=vulkan");
    } else {
        // Fallback: CPU only
        build.define("USE_BLAS", None);
        println!("cargo:rustc-link-lib=m");
    }

    // Compile the main voxtral static library
    build.include(voxtral_dir);
    build.compile("voxtral");

    println!("cargo:warning=Built voxtral C library for {}-{}", target_os, target_arch);
}

/// Generate tray menu translations from frontend locale files.
///
/// Source of truth: src/i18n/locales/*/translation.json
/// The English "tray" section defines the struct fields.
fn generate_tray_translations() {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let locales_dir = Path::new("../src/i18n/locales");

    println!("cargo:rerun-if-changed=../src/i18n/locales");

    // Collect all locale translations
    let mut translations: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for entry in fs::read_dir(locales_dir).unwrap().flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let lang = path.file_name().unwrap().to_str().unwrap().to_string();
        let json_path = path.join("translation.json");

        println!("cargo:rerun-if-changed={}", json_path.display());

        let content = fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        if let Some(tray) = parsed.get("tray").cloned() {
            translations.insert(lang, tray);
        }
    }

    // English defines the schema
    let english = translations.get("en").unwrap().as_object().unwrap();
    let fields: Vec<_> = english
        .keys()
        .map(|k| (camel_to_snake(k), k.clone()))
        .collect();

    // Generate code
    let mut out = String::from(
        "// Auto-generated from src/i18n/locales/*/translation.json - do not edit\n\n",
    );

    // Struct
    out.push_str("#[derive(Debug, Clone)]\npub struct TrayStrings {\n");
    for (rust_field, _) in &fields {
        out.push_str(&format!("    pub {rust_field}: String,\n"));
    }
    out.push_str("}\n\n");

    // Static map
    out.push_str(
        "pub static TRANSLATIONS: Lazy<HashMap<&'static str, TrayStrings>> = Lazy::new(|| {\n",
    );
    out.push_str("    let mut m = HashMap::new();\n");

    for (lang, tray) in &translations {
        out.push_str(&format!("    m.insert(\"{lang}\", TrayStrings {{\n"));
        for (rust_field, json_key) in &fields {
            let val = tray.get(json_key).and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!(
                "        {rust_field}: \"{}\".to_string(),\n",
                escape_string(val)
            ));
        }
        out.push_str("    });\n");
    }

    out.push_str("    m\n});\n");

    fs::write(Path::new(&out_dir).join("tray_translations.rs"), out).unwrap();

    println!(
        "cargo:warning=Generated tray translations: {} languages, {} fields",
        translations.len(),
        fields.len()
    );
}

fn camel_to_snake(s: &str) -> String {
    s.chars()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if c.is_uppercase() && i > 0 {
                acc.push('_');
            }
            acc.push(c.to_lowercase().next().unwrap());
            acc
        })
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_apple_intelligence_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/apple_intelligence.swift";
    const STUB_SWIFT_FILE: &str = "swift/apple_intelligence_stub.swift";
    const BRIDGE_HEADER: &str = "swift/apple_intelligence_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("apple_intelligence.o");
    let static_lib_path = out_dir.join("libapple_intelligence.a");

    let sdk_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("Failed to locate macOS SDK")
            .stdout,
    )
    .expect("SDK path is not valid UTF-8")
    .trim()
    .to_string();

    // Check if the SDK supports FoundationModels (required for Apple Intelligence)
    let framework_path =
        Path::new(&sdk_path).join("System/Library/Frameworks/FoundationModels.framework");
    let has_foundation_models = framework_path.exists();

    let source_file = if has_foundation_models {
        println!("cargo:warning=Building with Apple Intelligence support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=Apple Intelligence SDK not found. Building with stubs.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    let swiftc_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--find", "swiftc"])
            .output()
            .expect("Failed to locate swiftc")
            .stdout,
    )
    .expect("swiftc path is not valid UTF-8")
    .trim()
    .to_string();

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Use macOS 11.0 as deployment target for compatibility
    // The @available(macOS 26.0, *) checks in Swift handle runtime availability
    // Weak linking for FoundationModels is handled via cargo:rustc-link-arg below
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-target",
            "arm64-apple-macosx11.0",
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for Apple Intelligence bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for Apple Intelligence bridge");

    if !status.success() {
        panic!("libtool failed for Apple Intelligence bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apple_intelligence");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");

    if has_foundation_models {
        // Use weak linking so the app can launch on systems without FoundationModels
        println!("cargo:rustc-link-arg=-weak_framework");
        println!("cargo:rustc-link-arg=FoundationModels");
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
