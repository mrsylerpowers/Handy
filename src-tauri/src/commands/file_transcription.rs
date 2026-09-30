use crate::managers::file_transcription::{FileTranscriptionManager, StartFileTranscriptionError};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

/// Start transcribing an audio or video file. Returns once the job is
/// accepted; progress and the result arrive as `FileTranscriptionEvent`s.
#[tauri::command]
#[specta::specta]
pub fn start_file_transcription(
    manager: State<'_, Arc<FileTranscriptionManager>>,
    path: String,
) -> Result<(), StartFileTranscriptionError> {
    manager.start(PathBuf::from(path))
}

/// Stop the running file transcription after its current chunk.
#[tauri::command]
#[specta::specta]
pub fn cancel_file_transcription(manager: State<'_, Arc<FileTranscriptionManager>>) {
    manager.cancel();
}

/// Write a transcript to a path the user picked in the save dialog.
#[tauri::command]
#[specta::specta]
pub fn save_transcript_file(path: String, text: String) -> Result<(), String> {
    std::fs::write(&path, text).map_err(|e| format!("Failed to save {}: {}", path, e))
}
