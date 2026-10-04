//! Built-in API server: lets other devices, such as a phone dictation app set
//! to a "custom" OpenAI-compatible provider, transcribe audio with Handy's
//! model. [`http`] speaks the protocol; this module runs it from the settings
//! and connects it to the transcription pipeline.

pub mod http;

use crate::audio_toolkit::chunking::{plan_chunks, transcribe_chunks, ChunkedTranscription};
use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;
use crate::audio_toolkit::decode_audio_bytes;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::get_settings;
use http::{Backend, Segment, TranscribeError, Transcript};
use log::{error, info, warn};
use serde::Serialize;
use specta::Type;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::async_runtime::JoinHandle;
use tauri::{AppHandle, Manager};
use tauri_specta::Event;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Largest upload accepted (about an hour of 48 kHz stereo WAV).
const MAX_UPLOAD_BYTES: usize = 500 * 1024 * 1024;

/// Whether the server is listening, and the base URLs clients can use.
/// Emitted whenever it changes.
#[derive(Clone, Debug, Serialize, Type, tauri_specta::Event)]
pub struct ApiServerStatus {
    pub running: bool,
    pub port: u16,
    /// Base URLs for a client app: by this computer's network address, then
    /// by its name.
    pub base_urls: Vec<String>,
    /// Why the server is not running although it is enabled.
    pub error: Option<ApiServerError>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApiServerError {
    /// Another program is listening on the port.
    PortInUse,
    /// Windows reserves the port (e.g. for Hyper-V or WSL) or access to it
    /// is denied.
    PortUnavailable,
    Failed {
        message: String,
    },
}

struct RunningServer {
    shutdown: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

/// Starts and stops the server to match the settings.
pub struct ApiServerManager {
    app: AppHandle,
    server: tokio::sync::Mutex<Option<RunningServer>>,
    status: Mutex<ApiServerStatus>,
}

impl ApiServerManager {
    pub fn new(app: &AppHandle) -> Self {
        Self {
            app: app.clone(),
            server: tokio::sync::Mutex::new(None),
            status: Mutex::new(ApiServerStatus {
                running: false,
                port: get_settings(app).api_server_port,
                base_urls: Vec::new(),
                error: None,
            }),
        }
    }

    pub fn status(&self) -> ApiServerStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Start, stop or restart the server to match the current settings.
    pub async fn apply_settings(&self) -> ApiServerStatus {
        let mut server = self.server.lock().await;
        if let Some(running) = server.take() {
            stop(running).await;
        }

        let settings = get_settings(&self.app);
        let port = settings.api_server_port;
        let mut status = ApiServerStatus {
            running: false,
            port,
            base_urls: Vec::new(),
            error: None,
        };
        if settings.api_server_enabled {
            match self.start(port).await {
                Ok(running) => {
                    *server = Some(running);
                    status.running = true;
                    status.base_urls = base_urls(port);
                }
                Err(e) => status.error = Some(e),
            }
        }

        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = status.clone();
        if let Err(e) = status.emit(&self.app) {
            warn!("Failed to emit API server status: {e}");
        }
        status
    }

    async fn start(&self, port: u16) -> Result<RunningServer, ApiServerError> {
        let listener = bind(port).await.map_err(|e| {
            error!("API server can't listen on port {port}: {e}");
            match e.kind() {
                std::io::ErrorKind::AddrInUse => ApiServerError::PortInUse,
                std::io::ErrorKind::PermissionDenied => ApiServerError::PortUnavailable,
                _ => ApiServerError::Failed {
                    message: e.to_string(),
                },
            }
        })?;

        let backend = Arc::new(HandyBackend {
            app: self.app.clone(),
        });
        let router = http::router(backend, MAX_UPLOAD_BYTES);
        let (shutdown, shutdown_requested) = oneshot::channel::<()>();
        let task = tauri::async_runtime::spawn(async move {
            let shutdown_requested = async {
                let _ = shutdown_requested.await;
            };
            if let Err(e) = http::serve(listener, router, shutdown_requested).await {
                error!("API server stopped: {e}");
            }
        });
        info!("API server listening on port {port} (all network interfaces)");
        Ok(RunningServer { shutdown, task })
    }
}

/// Close the listener and give requests in flight a moment to finish; a
/// transcription still running after that completes in the background.
async fn stop(running: RunningServer) {
    let _ = running.shutdown.send(());
    if tokio::time::timeout(Duration::from_secs(2), running.task)
        .await
        .is_err()
    {
        info!("API server stopped listening; a request is still finishing");
    }
}

/// Listen on every IPv4 interface, so clients can reach the server by this
/// computer's address or by any DNS name that points to it. A server that was
/// just stopped can hold the port for a moment, so retry briefly.
async fn bind(port: u16) -> std::io::Result<TcpListener> {
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let mut attempts = 0;
    loop {
        match TcpListener::bind(address).await {
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse && attempts < 10 => {
                attempts += 1;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            result => return result,
        }
    }
}

fn base_urls(port: u16) -> Vec<String> {
    let mut hosts = Vec::new();
    if let Some(ip) = lan_ip() {
        hosts.push(ip.to_string());
    }
    let name = tauri_plugin_os::hostname().to_lowercase();
    if !name.is_empty() {
        hosts.push(name);
    }
    hosts
        .into_iter()
        .map(|host| format!("http://{host}:{port}/v1"))
        .collect()
}

/// The address other devices on the network reach this computer at: the one
/// it uses for outbound traffic. Connecting a UDP socket sends nothing.
fn lan_ip() -> Option<Ipv4Addr> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(8, 8, 8, 8), 53)).ok()?;
    match socket.local_addr().ok()?.ip() {
        std::net::IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => Some(ip),
        _ => None,
    }
}

/// A random API key that is easy to type on a phone: four groups of four
/// lowercase letters and digits (80 bits), without look-alikes like 0/o, 1/l.
pub fn generate_api_key() -> String {
    // 32 symbols, so each random byte maps to one without bias.
    const ALPHABET: &[u8; 32] = b"abcdefghijkmnpqrstuvwxyz23456789";
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the OS random number generator failed");
    bytes
        .chunks(4)
        .map(|group| {
            group
                .iter()
                .map(|&byte| ALPHABET[usize::from(byte) % ALPHABET.len()] as char)
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// Transcribes uploads the way "Transcribe File" does: decode, split into
/// chunks at pauses, run the selected model, then the same output processing
/// as dictation (but never LLM post-processing; client apps do their own).
struct HandyBackend {
    app: AppHandle,
}

impl Backend for HandyBackend {
    fn api_key(&self) -> Option<String> {
        let key = get_settings(&self.app).api_server_key;
        (!key.is_empty()).then(|| key.to_string())
    }

    fn model_id(&self) -> String {
        get_settings(&self.app).selected_model
    }

    fn transcribe(
        &self,
        audio: Vec<u8>,
        file_name: Option<&str>,
    ) -> Result<Transcript, TranscribeError> {
        let extension = file_name
            .and_then(|name| Path::new(name).extension())
            .and_then(|extension| extension.to_str());
        let samples = decode_audio_bytes(audio, extension)
            .map_err(|e| TranscribeError::BadAudio(format!("Couldn't read the audio: {e:#}")))?;

        let tm = Arc::clone(&self.app.state::<Arc<TranscriptionManager>>());
        if tm.is_streaming() {
            // A live dictation holds the engine; a batch run now would fail.
            return Err(TranscribeError::Unavailable(
                "Handy is busy with a live dictation. Try again in a moment.".to_string(),
            ));
        }
        if !tm.is_model_loaded() {
            let selected_model = get_settings(&self.app).selected_model;
            if let Err(e) = self
                .app
                .state::<Arc<ModelManager>>()
                .get_model_path(&selected_model)
            {
                return Err(TranscribeError::Unavailable(format!(
                    "Handy has no transcription model ready: {e}"
                )));
            }
        }
        tm.initiate_model_load();

        let chunks = plan_chunks(&samples);
        let mut segments = Vec::new();
        let outcome = transcribe_chunks(
            &samples,
            &chunks,
            || false,
            |chunk| tm.transcribe_segment(chunk.to_vec()),
            |index, text| {
                if !text.is_empty() {
                    segments.push(Segment {
                        start_secs: seconds(chunks[index].start),
                        end_secs: seconds(chunks[index].end),
                        text: text.to_string(),
                    });
                }
            },
        );
        tm.maybe_unload_immediately("API transcription");

        let text = match outcome {
            Ok(ChunkedTranscription::Completed(text)) => text,
            Ok(ChunkedTranscription::Cancelled) => {
                return Err(TranscribeError::Failed(
                    "Transcription was cancelled".to_string(),
                ))
            }
            Err(e) => return Err(TranscribeError::Failed(format!("{e:#}"))),
        };
        let processed = tauri::async_runtime::block_on(
            crate::actions::process_transcription_output(&self.app, &text, false),
        );

        let settings = get_settings(&self.app);
        let language = crate::actions::resolve_effective_language(&self.app, &settings);
        Ok(Transcript {
            text: processed.final_text,
            duration_secs: seconds(samples.len()),
            segments,
            language: (language != "auto").then_some(language),
        })
    }
}

fn seconds(samples: usize) -> f64 {
    samples as f64 / f64::from(WHISPER_SAMPLE_RATE)
}
