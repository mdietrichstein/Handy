//! FFI bindings for the voxtral C library.
//!
//! These bindings expose the streaming API from voxtral.h which is sufficient
//! for integrating voxtral as a transcription engine in Handy.

use std::ffi::{c_char, c_float, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr;

// ── Raw C bindings ──────────────────────────────────────────────────────────

extern "C" {
    /// Load model from directory containing consolidated.safetensors + tekken.json
    fn vox_load(model_dir: *const c_char) -> *mut c_void;

    /// Free all resources
    fn vox_free(ctx: *mut c_void);

    /// Create a streaming transcription context
    fn vox_stream_init(ctx: *mut c_void) -> *mut c_void;

    /// Feed audio samples (mono float32, 16kHz, [-1,1])
    fn vox_stream_feed(
        stream: *mut c_void,
        samples: *const c_float,
        n_samples: c_int,
    ) -> c_int;

    /// Signal end of audio
    fn vox_stream_finish(stream: *mut c_void) -> c_int;

    /// Retrieve pending decoded token strings
    fn vox_stream_get(
        stream: *mut c_void,
        out_tokens: *mut *const c_char,
        max: c_int,
    ) -> c_int;

    /// Free streaming context
    fn vox_stream_free(stream: *mut c_void);
}

// ── Safe Rust wrappers ──────────────────────────────────────────────────────

/// Opaque handle to a loaded voxtral model context.
pub struct VoxtralContext {
    ptr: *mut c_void,
}

// Safety: The C library uses no thread-local state for the context.
// All mutable access goes through &mut self.
unsafe impl Send for VoxtralContext {}

impl VoxtralContext {
    /// Load a voxtral model from a directory.
    ///
    /// The directory must contain `consolidated.safetensors`, `tekken.json`,
    /// and `params.json`.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let dir_str = model_dir
            .to_str()
            .ok_or_else(|| "Model path is not valid UTF-8".to_string())?;
        let c_dir =
            CString::new(dir_str).map_err(|e| format!("Invalid model path: {}", e))?;

        let ptr = unsafe { vox_load(c_dir.as_ptr()) };
        if ptr.is_null() {
            return Err(format!(
                "Failed to load voxtral model from {:?}",
                model_dir
            ));
        }

        Ok(Self { ptr })
    }

    /// Transcribe audio samples to text.
    ///
    /// `samples` must be mono float32 at 16 kHz, in the range [-1, 1].
    /// Returns the full transcription text.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Create stream
        let stream = unsafe { vox_stream_init(self.ptr) };
        if stream.is_null() {
            return Err("Failed to create voxtral stream".to_string());
        }

        // Feed all audio
        let rc = unsafe {
            vox_stream_feed(stream, samples.as_ptr(), samples.len() as c_int)
        };
        if rc != 0 {
            unsafe { vox_stream_free(stream) };
            return Err("Failed to feed audio to voxtral".to_string());
        }

        // Signal end of audio
        let rc = unsafe { vox_stream_finish(stream) };
        if rc != 0 {
            unsafe { vox_stream_free(stream) };
            return Err("Failed to finish voxtral stream".to_string());
        }

        // Drain all tokens
        let mut result = String::new();
        let mut token_ptrs: [*const c_char; 32] = [ptr::null(); 32];
        loop {
            let n = unsafe {
                vox_stream_get(stream, token_ptrs.as_mut_ptr(), 32)
            };
            if n <= 0 {
                break;
            }
            for i in 0..n as usize {
                if !token_ptrs[i].is_null() {
                    let token_str = unsafe { CStr::from_ptr(token_ptrs[i]) };
                    if let Ok(s) = token_str.to_str() {
                        result.push_str(s);
                    }
                }
            }
        }

        unsafe { vox_stream_free(stream) };

        Ok(result.trim().to_string())
    }
}

impl Drop for VoxtralContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { vox_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}
