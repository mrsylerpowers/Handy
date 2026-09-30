//! "Transcribe File": decodes a user-chosen audio or video file, splits it into
//! chunks and runs them through the shared [`TranscriptionManager`] on a
//! background thread, reporting progress to the frontend as events.

use crate::audio_toolkit::chunking::{plan_chunks, transcribe_chunks, ChunkedTranscription};
use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;
use crate::audio_toolkit::decode_audio_file;
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::get_settings;
use anyhow::{anyhow, Result};
use log::{error, info};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Manager};
use tauri_specta::Event;

/// Lifecycle of a file transcription, emitted to the frontend.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
#[serde(tag = "status")]
pub enum FileTranscriptionEvent {
    /// The file decoded; `total_chunks` chunks will now be transcribed.
    #[serde(rename = "started")]
    Started {
        file_name: String,
        duration_secs: f64,
        total_chunks: u32,
    },
    /// One more chunk finished; `text` is its transcript (empty if silent).
    #[serde(rename = "progress")]
    Progress {
        completed_chunks: u32,
        total_chunks: u32,
        text: String,
    },
    /// The whole file is done; `text` is the final transcript.
    #[serde(rename = "completed")]
    Completed { text: String },
    #[serde(rename = "failed")]
    Failed { error: String },
    #[serde(rename = "cancelled")]
    Cancelled,
}

/// Why a file transcription could not start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum StartFileTranscriptionError {
    /// Another file is still being transcribed.
    AlreadyRunning,
    /// A dictation is being recorded.
    RecordingInProgress,
    /// No transcription model is downloaded.
    NoModel,
}

/// Runs at most one file transcription at a time.
pub struct FileTranscriptionManager {
    app_handle: AppHandle,
    running: Arc<AtomicBool>,
    cancel_requested: Arc<AtomicBool>,
}

impl FileTranscriptionManager {
    pub fn new(app_handle: &AppHandle) -> Self {
        Self {
            app_handle: app_handle.clone(),
            running: Arc::new(AtomicBool::new(false)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Start transcribing `path` in the background. Progress and the result
    /// arrive as [`FileTranscriptionEvent`]s.
    pub fn start(&self, path: PathBuf) -> Result<(), StartFileTranscriptionError> {
        let app = &self.app_handle;
        let recording = app
            .try_state::<Arc<AudioRecordingManager>>()
            .is_some_and(|recorder| recorder.is_recording());
        if recording {
            return Err(StartFileTranscriptionError::RecordingInProgress);
        }

        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        if !tm.is_model_loaded() {
            let selected_model = get_settings(app).selected_model;
            if let Err(e) = app
                .state::<Arc<ModelManager>>()
                .get_model_path(&selected_model)
            {
                info!("Not starting file transcription: {}", e);
                return Err(StartFileTranscriptionError::NoModel);
            }
        }

        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(StartFileTranscriptionError::AlreadyRunning);
        }
        self.cancel_requested.store(false, Ordering::Release);

        // Warm the model up while the file decodes.
        tm.initiate_model_load();

        let app = app.clone();
        let running = Arc::clone(&self.running);
        let cancel_requested = Arc::clone(&self.cancel_requested);
        std::thread::spawn(move || {
            let started = Instant::now();
            info!("Transcribing file {}", path.display());
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                run_file_job(&app, &tm, &path, &cancel_requested)
            }))
            .unwrap_or_else(|_| Err(anyhow!("File transcription crashed unexpectedly")));
            tm.maybe_unload_immediately("file transcription");

            let event = match outcome {
                Ok(ChunkedTranscription::Completed(text)) => {
                    info!("File transcription completed in {:?}", started.elapsed());
                    FileTranscriptionEvent::Completed { text }
                }
                Ok(ChunkedTranscription::Cancelled) => {
                    info!("File transcription cancelled");
                    FileTranscriptionEvent::Cancelled
                }
                Err(e) => {
                    error!("File transcription failed: {e:#}");
                    FileTranscriptionEvent::Failed {
                        error: format!("{e:#}"),
                    }
                }
            };
            // Clear the flag before announcing the end so the frontend can
            // start the next file as soon as it hears about this one.
            running.store(false, Ordering::Release);
            if let Err(e) = event.emit(&app) {
                error!("Failed to emit file transcription result: {}", e);
            }
        });
        Ok(())
    }

    /// Stop the running file transcription after its current chunk.
    pub fn cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }
}

fn run_file_job(
    app: &AppHandle,
    tm: &TranscriptionManager,
    path: &Path,
    cancel_requested: &AtomicBool,
) -> Result<ChunkedTranscription> {
    let decode_started = Instant::now();
    let samples = decode_audio_file(path)?;
    let chunks = plan_chunks(&samples);
    let duration_secs = samples.len() as f64 / f64::from(WHISPER_SAMPLE_RATE);
    let total_chunks = chunks.len() as u32;
    info!(
        "Decoded {:.1}s of audio in {:?}; {} chunk(s) to transcribe",
        duration_secs,
        decode_started.elapsed(),
        total_chunks
    );

    emit(
        app,
        FileTranscriptionEvent::Started {
            file_name: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            duration_secs,
            total_chunks,
        },
    );

    let outcome = transcribe_chunks(
        &samples,
        &chunks,
        || cancel_requested.load(Ordering::Acquire),
        |chunk| tm.transcribe_segment(chunk.to_vec()),
        |index, text| {
            emit(
                app,
                FileTranscriptionEvent::Progress {
                    completed_chunks: index as u32 + 1,
                    total_chunks,
                    text: text.to_string(),
                },
            )
        },
    )?;

    // Same output handling as dictation (e.g. Chinese variant conversion), but
    // never LLM post-processing.
    Ok(match outcome {
        ChunkedTranscription::Completed(text) => {
            let processed = tauri::async_runtime::block_on(
                crate::actions::process_transcription_output(app, &text, false),
            );
            ChunkedTranscription::Completed(processed.final_text)
        }
        cancelled => cancelled,
    })
}

fn emit(app: &AppHandle, event: FileTranscriptionEvent) {
    if let Err(e) = event.emit(app) {
        error!("Failed to emit file transcription event: {}", e);
    }
}
