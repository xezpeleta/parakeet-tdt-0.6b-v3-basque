/**
 * bridge.js — browser API bridge for parakeet-wasm
 *
 * All functions are exposed as globals (window.jsBridge*) so
 * the Rust WASM module can call them via js_sys::Reflect.
 *
 * Load this script BEFORE the Rust WASM pkg/parakeet_wasm.js.
 */
'use strict';

/* ============================================================
   State
   ============================================================ */
let _sherpaModule = null;   // Emscripten module after init
let _recognizer   = null;   // OfflineRecognizer instance

const NUM_FILES = 4; // encoder, decoder, joiner, tokens

/* ============================================================
   1. Load the sherpa-onnx Emscripten WASM runtime
   ============================================================ */
window.jsBridgeLoadWasm = function(baseUrl) {
  return new Promise((resolve, reject) => {
    const wasmCdn = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;

    // Pre-configure Module BEFORE the script runs
    window.Module = {
      locateFile: (path) => wasmCdn + '/' + path,

      // Return an empty ArrayBuffer to skip the 191 MB English model .data
      // (Emscripten checks this before making the XHR)
      getPreloadedPackage: (_name, _size) => {
        console.log('[bridge] getPreloadedPackage → returning empty buffer');
        return new ArrayBuffer(0);
      },

      setStatus: (s) => {
        if (s) console.log('[bridge] WASM status:', s);
      },

      onRuntimeInitialized: function() {
        console.log('[bridge] WASM runtime ready');
        _sherpaModule = window.Module;
        resolve();
      },

      printErr: (msg) => {
        // Suppress noisy warnings about the empty .data file
        if (msg.includes('.data') || msg.includes('Assertion') || msg.includes('abort')) {
          console.warn('[bridge] WASM warn:', msg);
        } else {
          console.error('[bridge] WASM err:', msg);
        }
      },
    };

    // Load high-level JS API first, then the WASM engine
    const apiScript = document.createElement('script');
    apiScript.src = wasmCdn + '/sherpa-onnx-asr.js';
    apiScript.onerror = () => reject(new Error('Failed to load sherpa-onnx-asr.js'));
    apiScript.onload  = () => {
      const wasmScript = document.createElement('script');
      wasmScript.src = wasmCdn + '/sherpa-onnx-wasm-main-asr.js';
      wasmScript.onerror = () => reject(new Error('Failed to load sherpa-onnx-wasm-main-asr.js'));
      document.head.appendChild(wasmScript);
    };
    document.head.appendChild(apiScript);
  });
};

/* ============================================================
   2. Emscripten virtual filesystem write
   ============================================================ */
window.jsBridgeWasmFsWrite = function(path, uint8array) {
  const fs = window.Module.FS || window.FS;
  if (!fs) throw new Error('WASM FS not available');
  fs.writeFile(path, uint8array);
  console.log('[bridge] FS.writeFile', path, uint8array.byteLength, 'bytes');
};

/* ============================================================
   3. Create OfflineRecognizer
   ============================================================ */
window.jsBridgeCreateRecognizer = function() {
  if (!_sherpaModule) throw new Error('Sherpa WASM not initialised');
  _recognizer = new OfflineRecognizer({
    modelConfig: {
      transducer: {
        encoder: '/encoder.int8.onnx',
        decoder: '/decoder.int8.onnx',
        joiner:  '/joiner.int8.onnx',
      },
      tokens:     '/tokens.txt',
      numThreads: 2,
      provider:   'cpu',
      debug:      0,
      modelType:  '',
    },
    decodingMethod: 'greedy_search',
  }, _sherpaModule);
  console.log('[bridge] OfflineRecognizer created');
};

/* ============================================================
   4. Audio decoding — returns a Float32Array at 16 kHz
   ============================================================ */
window.jsBridgeDecodeAudio = async function(uint8array) {
  const TARGET_SR = 16000;
  const arrayBuffer = uint8array.buffer.slice(
    uint8array.byteOffset,
    uint8array.byteOffset + uint8array.byteLength
  );

  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
  let decoded;
  try {
    decoded = await audioCtx.decodeAudioData(arrayBuffer);
  } finally {
    await audioCtx.close();
  }

  // Resample if needed (most browsers honour sampleRate in the constructor,
  // but this is a safety net)
  let samples = decoded.getChannelData(0);
  if (decoded.sampleRate !== TARGET_SR) {
    const offCtx = new OfflineAudioContext(1, Math.round(decoded.duration * TARGET_SR), TARGET_SR);
    const src = offCtx.createBufferSource();
    src.buffer = decoded;
    src.connect(offCtx.destination);
    src.start();
    const resampled = await offCtx.startRendering();
    samples = resampled.getChannelData(0);
  }
  return samples; // Float32Array
};

/* ============================================================
   5. Run ASR inference (chunks audio into 30-second segments)
   ============================================================ */
window.jsBridgeRunInference = function(samplesFloat32) {
  if (!_recognizer) throw new Error('Recognizer not created');

  const TARGET_SR  = 16000;
  const CHUNK_SECS = 30;
  const chunkSize  = CHUNK_SECS * TARGET_SR;
  const total      = samplesFloat32.length;
  const results    = [];

  for (let offset = 0; offset < total; offset += chunkSize) {
    const chunk  = samplesFloat32.subarray(offset, Math.min(offset + chunkSize, total));
    const stream = _recognizer.createStream();
    stream.acceptWaveform(TARGET_SR, chunk);
    _recognizer.decode(stream);
    const res = _recognizer.getResult(stream);
    results.push(res.text || '');
    stream.free();
  }

  return results.join(' ').trim();
};

/* ============================================================
   6. Fetch a file with CORS (returns Promise<Uint8Array>)
   ============================================================ */
window.jsBridgeFetch = async function(url) {
  const resp = await fetch(url, { mode: 'cors' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const ab = await resp.arrayBuffer();
  return new Uint8Array(ab);
};

/* ============================================================
   7. OPFS helpers
   ============================================================ */
async function _opfsGetDir(dirName, create) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(dirName, { create });
}

window.jsBridgeOpfsRead = async function(dirName, filename) {
  try {
    const dir  = await _opfsGetDir(dirName, false);
    const fh   = await dir.getFileHandle(filename);
    const file = await fh.getFile();
    const ab   = await file.arrayBuffer();
    return new Uint8Array(ab);
  } catch {
    return null; // not cached
  }
};

window.jsBridgeOpfsWrite = async function(dirName, filename, uint8array) {
  try {
    const dir      = await _opfsGetDir(dirName, true);
    const fh       = await dir.getFileHandle(filename, { create: true });
    const writable = await fh.createWritable();
    await writable.write(uint8array);
    await writable.close();
    console.log('[bridge] OPFS cached:', filename, uint8array.byteLength, 'B');
  } catch (err) {
    console.warn('[bridge] OPFS write failed:', err);
  }
};

window.jsBridgeOpfsClearDir = async function(dirName) {
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(dirName, { recursive: true });
    console.log('[bridge] OPFS cleared:', dirName);
  } catch (err) {
    console.warn('[bridge] OPFS clear failed:', err);
  }
};

/* ============================================================
   8. UI callbacks — called by Rust to update the page
   ============================================================ */

// Called when init() completes: cached = true if all files in OPFS
window.jsBridgeOnInitComplete = function(cached) {
  setWasmReady();
  const btn   = document.getElementById('downloadBtn');
  const label = document.getElementById('downloadLabel');
  if (cached) {
    label.textContent = translations[currentLang].downloadBtnCached || 'Load saved model';
    document.getElementById('clearCacheBtn').classList.remove('hidden');
  }
  document.getElementById('downloadBtn').disabled = false;
};

// Progress bar (0–100)
window.jsBridgeSetProgress = function(pct) {
  const bar = document.getElementById('progressBar');
  if (bar) bar.style.width = pct + '%';
};

// Individual model file status: state = "loading" | "done" | "pending"
window.jsBridgeSetFileState = function(idx, state) {
  const el = document.getElementById('file' + idx);
  if (!el) return;
  if (state === 'done')    { el.textContent = '✓'; el.className = 'check'; }
  else if (state === 'loading') { el.textContent = '⏳'; el.className = ''; }
  else                     { el.textContent = '○'; el.className = 'pending'; }
};

// Update the "currently downloading" label
window.jsBridgeSetFileLabel = function(text) {
  const el = document.getElementById('downloadFileStatus');
  if (el) el.textContent = text;
};

// Called when model loading/download phase is completely done
window.jsBridgeOnModelReady = function() {
  setModelStatus('ready');
  document.getElementById('downloadBtn').classList.add('hidden');
  document.getElementById('clearCacheBtn').classList.remove('hidden');
  document.getElementById('downloadFileStatus').textContent = '';
  document.getElementById('transcribeBtn').disabled = false;
  document.getElementById('micBtn').disabled = false;
  window.jsBridgeSetProgress(100);
};

// Called when audio file / recording is ready
window.jsBridgeOnAudioLoaded = function() {
  if (document.getElementById('transcribeBtn')) {
    document.getElementById('transcribeBtn').disabled = !window.parakeetWasm.is_model_ready();
  }
};

// Called after cache is cleared
window.jsBridgeOnCacheCleared = function() {
  document.getElementById('clearCacheBtn').classList.add('hidden');
  document.getElementById('downloadBtn').classList.remove('hidden');
  document.getElementById('downloadLabel').textContent =
    translations[currentLang].downloadBtn || 'Download model';
  document.getElementById('downloadBtn').disabled = false;
  for (let i = 0; i < NUM_FILES; i++) {
    window.jsBridgeSetFileState(i, 'pending');
  }
};
