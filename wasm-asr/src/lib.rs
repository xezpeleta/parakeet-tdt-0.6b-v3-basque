use js_sys::{Array, Float32Array, Promise, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

// =========================================================
// Constants
// =========================================================

const HF_MODEL_BASE: &str =
    "https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx/resolve/main";
const HF_WASM_BASE: &str =
    "https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-en/resolve/main";
const OPFS_DIR: &str = "parakeet-basque-asr";

struct ModelFile {
    name: &'static str,
    wasm_path: &'static str,
    size: u64,
}

const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        name: "encoder.int8.onnx",
        wasm_path: "/encoder.int8.onnx",
        size: 652_000_000,
    },
    ModelFile {
        name: "decoder.int8.onnx",
        wasm_path: "/decoder.int8.onnx",
        size: 11_800_000,
    },
    ModelFile {
        name: "joiner.int8.onnx",
        wasm_path: "/joiner.int8.onnx",
        size: 6_360_000,
    },
    ModelFile {
        name: "tokens.txt",
        wasm_path: "/tokens.txt",
        size: 94_000,
    },
];

// =========================================================
// Bridge: call window.jsBridge* functions dynamically
// =========================================================

fn get_window() -> web_sys::Window {
    web_sys::window().expect("no global window")
}

/// Call a global bridge function synchronously.
fn bridge(name: &str, args: &Array) -> Result<JsValue, JsValue> {
    let win = get_window();
    let func = js_sys::Reflect::get(&win, &JsValue::from_str(name))?;
    if func.is_undefined() || func.is_null() {
        return Err(JsValue::from_str(&format!(
            "bridge: '{}' not found on window",
            name
        )));
    }
    let func: js_sys::Function = func
        .dyn_into()
        .map_err(|_| JsValue::from_str(&format!("bridge: '{}' is not a function", name)))?;
    func.apply(&JsValue::NULL, args)
}

/// Call a global bridge function that returns a Promise and await it.
async fn bridge_async(name: &str, args: &Array) -> Result<JsValue, JsValue> {
    let result = bridge(name, args)?;
    let promise: Promise = result.dyn_into().map_err(|_| {
        JsValue::from_str(&format!("bridge: '{}' did not return a Promise", name))
    })?;
    JsFuture::from(promise).await
}

// =========================================================
// App state
// =========================================================

use std::cell::{Cell, RefCell};

thread_local! {
    static MODEL_READY: Cell<bool> = Cell::new(false);
    static AUDIO_DATA: RefCell<Option<Vec<u8>>> = RefCell::new(None);
}

// =========================================================
// Exported WASM functions
// =========================================================

/// Called once on page load. Loads the sherpa-onnx WASM engine and checks
/// the OPFS cache, then reports status back to bridge.js / index.html.
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    // Load sherpa-onnx WASM runtime
    let args = Array::new();
    args.push(&JsValue::from_str(HF_WASM_BASE));
    bridge_async("jsBridgeLoadWasm", &args).await?;

    // Check whether all model files are already in OPFS
    let all_cached = all_files_cached().await;

    let args2 = Array::new();
    args2.push(&JsValue::from_bool(all_cached));
    bridge("jsBridgeOnInitComplete", &args2)?;

    Ok(())
}

/// Download model files (or load from OPFS cache), write them to the
/// Emscripten virtual filesystem, then create the OfflineRecognizer.
#[wasm_bindgen]
pub async fn download_or_load_model() -> Result<(), JsValue> {
    if MODEL_READY.with(|r| r.get()) {
        return Ok(());
    }

    let total_size: u64 = MODEL_FILES.iter().map(|f| f.size).sum();
    let mut loaded_size: u64 = 0;

    for (i, mf) in MODEL_FILES.iter().enumerate() {
        // Mark file as in-progress in the UI
        {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_str("loading"));
            bridge("jsBridgeSetFileState", &args)?;
        }

        // Try OPFS cache first
        let cached = opfs_read(OPFS_DIR, mf.name).await;

        let file_data: Uint8Array = match cached {
            Some(ua) => {
                let args = Array::new();
                args.push(&JsValue::from_str(&format!("📦 {} (cache)", mf.name)));
                bridge("jsBridgeSetFileLabel", &args)?;
                ua
            }
            None => {
                // Download from HuggingFace
                let url = format!("{}/{}", HF_MODEL_BASE, mf.name);
                {
                    let args = Array::new();
                    args.push(&JsValue::from_str(&format!("⬇️ {}…", mf.name)));
                    bridge("jsBridgeSetFileLabel", &args)?;
                }

                let fetch_args = Array::new();
                fetch_args.push(&JsValue::from_str(&url));
                let result = bridge_async("jsBridgeFetch", &fetch_args).await?;
                let ua: Uint8Array = result.dyn_into().map_err(|_| {
                    JsValue::from_str("jsBridgeFetch did not return Uint8Array")
                })?;

                // Save to OPFS so next visit is instant
                let write_args = Array::new();
                write_args.push(&JsValue::from_str(OPFS_DIR));
                write_args.push(&JsValue::from_str(mf.name));
                write_args.push(&ua);
                // Ignore write errors (e.g. private browsing, quota exceeded)
                let _ = bridge_async("jsBridgeOpfsWrite", &write_args).await;

                ua
            }
        };

        // Write file into Emscripten's virtual filesystem
        {
            let args = Array::new();
            args.push(&JsValue::from_str(mf.wasm_path));
            args.push(&file_data);
            bridge("jsBridgeWasmFsWrite", &args)?;
        }

        // Update progress bar
        {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_str("done"));
            bridge("jsBridgeSetFileState", &args)?;
        }

        loaded_size += mf.size;
        let pct = ((loaded_size as f64 / total_size as f64) * 70.0 + 10.0) as u8;
        {
            let args = Array::new();
            args.push(&JsValue::from_f64(pct as f64));
            bridge("jsBridgeSetProgress", &args)?;
        }
    }

    // Create the OfflineRecognizer
    bridge("jsBridgeCreateRecognizer", &Array::new())?;

    MODEL_READY.with(|r| r.set(true));

    bridge("jsBridgeOnModelReady", &Array::new())?;
    Ok(())
}

/// Store audio bytes provided by the user (file upload or microphone).
/// Called from JavaScript before calling transcribe().
#[wasm_bindgen]
pub fn set_audio_data(data: Uint8Array) {
    let len = data.length() as usize;
    let mut vec = vec![0u8; len];
    data.copy_to(&mut vec);
    AUDIO_DATA.with(|a| *a.borrow_mut() = Some(vec));
    bridge("jsBridgeOnAudioLoaded", &Array::new()).ok();
}

/// Decode the stored audio, run inference, return transcript text.
#[wasm_bindgen]
pub async fn transcribe() -> Result<String, JsValue> {
    if !MODEL_READY.with(|r| r.get()) {
        return Err(JsValue::from_str("Model not ready"));
    }

    let audio = AUDIO_DATA.with(|a| a.borrow().clone());
    let audio = audio.ok_or_else(|| JsValue::from_str("No audio data loaded"))?;

    // Copy bytes into a JS Uint8Array and let bridge.js decode audio
    let ua = Uint8Array::new_with_length(audio.len() as u32);
    ua.copy_from(&audio);

    let decode_args = Array::new();
    decode_args.push(&ua);
    let samples_val = bridge_async("jsBridgeDecodeAudio", &decode_args).await?;
    let samples: Float32Array = samples_val.dyn_into().map_err(|_| {
        JsValue::from_str("jsBridgeDecodeAudio did not return Float32Array")
    })?;

    // Run offline ASR inference
    let infer_args = Array::new();
    infer_args.push(&samples);
    let text_val = bridge("jsBridgeRunInference", &infer_args)?;

    Ok(text_val.as_string().unwrap_or_default())
}

/// Delete the OPFS cache directory and reset recognizer state.
#[wasm_bindgen]
pub async fn clear_cache() -> Result<(), JsValue> {
    let args = Array::new();
    args.push(&JsValue::from_str(OPFS_DIR));
    bridge_async("jsBridgeOpfsClearDir", &args).await?;

    MODEL_READY.with(|r| r.set(false));
    bridge("jsBridgeOnCacheCleared", &Array::new())?;
    Ok(())
}

/// Returns true if the model has been loaded and a recognizer created.
#[wasm_bindgen]
pub fn is_model_ready() -> bool {
    MODEL_READY.with(|r| r.get())
}

// =========================================================
// OPFS helpers (thin wrappers around JS bridge)
// =========================================================

async fn opfs_read(dir_name: &str, filename: &str) -> Option<Uint8Array> {
    let args = Array::new();
    args.push(&JsValue::from_str(dir_name));
    args.push(&JsValue::from_str(filename));
    let result = bridge_async("jsBridgeOpfsRead", &args).await.ok()?;
    if result.is_null() || result.is_undefined() {
        None
    } else {
        result.dyn_into::<Uint8Array>().ok()
    }
}

async fn all_files_cached() -> bool {
    for mf in MODEL_FILES {
        if opfs_read(OPFS_DIR, mf.name).await.is_none() {
            return false;
        }
    }
    true
}
