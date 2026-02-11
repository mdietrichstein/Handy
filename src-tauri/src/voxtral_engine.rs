//! Voxtral transcription engine for Handy.
//!
//! Wraps the voxtral C library (Mistral AI Voxtral Realtime 4B) to provide
//! speech-to-text transcription. Uses Vulkan GPU acceleration on Linux and
//! Metal (MPS) on macOS Apple Silicon.

use crate::voxtral_ffi::VoxtralContext;
use log::{debug, info};
use std::path::{Path, PathBuf};

/// Voxtral transcription engine.
///
/// Manages the lifecycle of a loaded voxtral model and provides
/// a simple transcribe interface compatible with Handy's transcription manager.
pub struct VoxtralEngine {
    context: Option<VoxtralContext>,
    model_path: Option<PathBuf>,
}

impl VoxtralEngine {
    /// Create a new unloaded engine.
    pub fn new() -> Self {
        Self {
            context: None,
            model_path: None,
        }
    }

    /// Load a voxtral model from a directory.
    ///
    /// The directory must contain:
    /// - `consolidated.safetensors` (~8.9 GB)
    /// - `tekken.json` (tokenizer)
    /// - `params.json` (model config)
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        info!("Loading voxtral model from {:?}", model_path);
        let ctx = VoxtralContext::load(model_path)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        self.context = Some(ctx);
        self.model_path = Some(model_path.to_path_buf());

        info!("Voxtral model loaded successfully");
        Ok(())
    }

    /// Unload the model and free resources.
    pub fn unload_model(&mut self) {
        if self.context.is_some() {
            debug!("Unloading voxtral model");
            self.context = None;
            self.model_path = None;
        }
    }

    /// Transcribe audio samples to text.
    ///
    /// `samples` must be mono float32 at 16 kHz, range [-1, 1].
    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let ctx = self
            .context
            .as_ref()
            .ok_or("Voxtral model not loaded")?;

        debug!(
            "Voxtral: transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / 16000.0
        );

        let text = ctx
            .transcribe(&samples)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        Ok(text)
    }
}

impl Default for VoxtralEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VoxtralEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}
