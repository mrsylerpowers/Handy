//! OpenAI-compatible speech-to-text HTTP API: the subset dictation apps use
//! when pointed at a "custom" provider (`POST /v1/audio/transcriptions` and
//! `GET /v1/models`). Independent of Tauri, so it is tested against a fake
//! [`Backend`].

use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Multipart, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use log::{info, warn};
use serde_json::json;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

type SharedBackend = Arc<dyn Backend>;

/// What the server needs from the app.
pub trait Backend: Send + Sync + 'static {
    /// The key clients must present, or `None` to accept every request.
    fn api_key(&self) -> Option<String>;
    /// Id of the model that transcribes requests, listed by `/v1/models`.
    fn model_id(&self) -> String;
    /// Decode and transcribe an uploaded audio file. Blocks until done.
    fn transcribe(
        &self,
        audio: Vec<u8>,
        file_name: Option<&str>,
    ) -> Result<Transcript, TranscribeError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Length of the decoded audio, in seconds.
    pub duration_secs: f64,
    /// The transcribed stretches of the audio, in order.
    pub segments: Vec<Segment>,
    /// ISO 639-1 code of the transcript's language, when known.
    pub language: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TranscribeError {
    /// The upload is not audio this build can decode.
    BadAudio(String),
    /// Nothing can transcribe right now (e.g. no model is downloaded).
    Unavailable(String),
    /// Transcription itself failed.
    Failed(String),
}

/// The API's routes. Request bodies over `max_upload_bytes` are refused.
///
/// Every route is also served without the `/v1` prefix, so a base URL entered
/// without it still works.
pub fn router(backend: SharedBackend, max_upload_bytes: usize) -> Router {
    let api = Router::new()
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/audio/transcriptions", post(transcribe))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(
            backend.clone(),
            require_api_key,
        ));
    Router::new()
        .route("/", get(status))
        .merge(api)
        .fallback(unknown_endpoint)
        .layer(DefaultBodyLimit::max(max_upload_bytes))
        .layer(middleware::from_fn(log_request))
        .with_state(backend)
}

/// Serve `router` on `listener` until `shutdown` resolves. The listener is
/// closed as soon as it does; requests already in flight run to completion.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}

async fn status() -> &'static str {
    "Handy's API server is running. Use this address with /v1 appended as your app's base URL.\n"
}

async fn list_models(State(backend): State<SharedBackend>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": backend.model_id(),
            "object": "model",
            "created": 0,
            "owned_by": "handy",
        }],
    }))
    .into_response()
}

async fn unknown_endpoint(method: Method, uri: Uri) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!(
            "Handy has no endpoint for {method} {}. It serves POST /v1/audio/transcriptions \
             and GET /v1/models.",
            uri.path()
        ),
    )
}

/// The upload's fields this server reads; the rest (`model`, `language`,
/// `prompt`, `temperature`, ...) are accepted and ignored, since Handy's own
/// model, language and custom words apply.
struct TranscriptionForm {
    file: Option<(Vec<u8>, Option<String>)>,
    response_format: Option<String>,
    field_names: Vec<String>,
}

async fn transcribe(
    State(backend): State<SharedBackend>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let form = match multipart {
        Ok(multipart) => match read_form(multipart).await {
            Ok(form) => form,
            Err(e) => return error(e.status(), e.body_text()),
        },
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };

    let format = match form.response_format.as_deref() {
        None => ResponseFormat::Json,
        Some(name) => match ResponseFormat::parse(name) {
            Some(format) => format,
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Unsupported response_format '{name}'. Use json, text, verbose_json, \
                         srt or vtt."
                    ),
                )
            }
        },
    };

    let Some((audio, file_name)) = form.file else {
        let received = if form.field_names.is_empty() {
            "none".to_string()
        } else {
            form.field_names.join(", ")
        };
        return error(
            StatusCode::BAD_REQUEST,
            format!("The upload has no 'file' field with the audio (fields received: {received})."),
        );
    };

    let started = Instant::now();
    let size = audio.len();
    let outcome =
        tokio::task::spawn_blocking(move || backend.transcribe(audio, file_name.as_deref())).await;

    match outcome {
        Ok(Ok(transcript)) => {
            info!(
                "API transcribed {:.1}s of audio ({} bytes) in {:.2?}",
                transcript.duration_secs,
                size,
                started.elapsed()
            );
            format.render(&transcript)
        }
        Ok(Err(failure)) => {
            let (status, message) = match failure {
                TranscribeError::BadAudio(message) => (StatusCode::BAD_REQUEST, message),
                TranscribeError::Unavailable(message) => (StatusCode::SERVICE_UNAVAILABLE, message),
                TranscribeError::Failed(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
            };
            warn!("API transcription of {size} bytes failed: {message}");
            error(status, message)
        }
        Err(join_error) => {
            warn!("API transcription task died: {join_error}");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Transcription crashed unexpectedly.",
            )
        }
    }
}

async fn read_form(mut multipart: Multipart) -> Result<TranscriptionForm, MultipartError> {
    let mut form = TranscriptionForm {
        file: None,
        response_format: None,
        field_names: Vec::new(),
    };
    loop {
        let Some(field) = multipart.next_field().await? else {
            return Ok(form);
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let file_name = field.file_name().map(str::to_string);
                // Takes over the buffer instead of copying the upload.
                let bytes = Vec::from(field.bytes().await?);
                form.file = Some((bytes, file_name));
            }
            "response_format" => {
                let value = field.text().await?;
                form.response_format = Some(value.trim().to_string());
            }
            _ => {}
        }
        form.field_names.push(name);
    }
}

#[derive(Clone, Copy)]
enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "json" => Some(Self::Json),
            "text" => Some(Self::Text),
            "verbose_json" => Some(Self::VerboseJson),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            _ => None,
        }
    }

    fn render(self, transcript: &Transcript) -> Response {
        match self {
            Self::Json => Json(json!({ "text": transcript.text })).into_response(),
            Self::Text => plain_text("text/plain; charset=utf-8", transcript.text.clone()),
            Self::VerboseJson => Json(verbose_json(transcript)).into_response(),
            Self::Srt => plain_text("text/plain; charset=utf-8", srt(&transcript.segments)),
            Self::Vtt => plain_text("text/vtt; charset=utf-8", vtt(&transcript.segments)),
        }
    }
}

fn plain_text(content_type: &'static str, body: String) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

/// OpenAI's `verbose_json` shape. Handy has no per-token data, so those fields
/// carry neutral values; they are present because typed clients require them.
fn verbose_json(transcript: &Transcript) -> serde_json::Value {
    let segments: Vec<serde_json::Value> = transcript
        .segments
        .iter()
        .enumerate()
        .map(|(id, segment)| {
            json!({
                "id": id,
                "seek": 0,
                "start": segment.start_secs,
                "end": segment.end_secs,
                "text": segment.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            })
        })
        .collect();
    json!({
        "task": "transcribe",
        "language": transcript.language.as_deref().unwrap_or("unknown"),
        "duration": transcript.duration_secs,
        "text": transcript.text,
        "segments": segments,
    })
}

fn srt(segments: &[Segment]) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            format!(
                "{}\n{} --> {}\n{}\n",
                index + 1,
                timestamp(segment.start_secs, ','),
                timestamp(segment.end_secs, ','),
                segment.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn vtt(segments: &[Segment]) -> String {
    let mut out = String::from("WEBVTT\n");
    for segment in segments {
        out.push_str(&format!(
            "\n{} --> {}\n{}\n",
            timestamp(segment.start_secs, '.'),
            timestamp(segment.end_secs, '.'),
            segment.text
        ));
    }
    out
}

/// `HH:MM:SS<separator>mmm`, the cue timestamp of SRT (`,`) and WebVTT (`.`).
fn timestamp(secs: f64, separator: char) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

async fn require_api_key(
    State(backend): State<SharedBackend>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(expected) = backend.api_key() {
        if presented_key(request.headers()).as_deref() != Some(expected.as_str()) {
            return error_with_code(
                StatusCode::UNAUTHORIZED,
                "Missing or incorrect API key. Use the key shown on Handy's API Server page.",
                Some("invalid_api_key"),
            );
        }
    }
    next.run(request).await
}

/// The key a client sent: OpenAI-style `Authorization: Bearer <key>` (or the
/// bare key), else an `x-api-key` / `api-key` header as some clients send.
fn presented_key(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers.get(header::AUTHORIZATION) {
        let value = value.to_str().ok()?.trim();
        let key = match value.split_once(' ') {
            Some((scheme, rest)) if scheme.eq_ignore_ascii_case("bearer") => rest.trim(),
            _ => value,
        };
        return Some(key.to_string());
    }
    ["x-api-key", "api-key"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
}

/// Logs each request's outcome (never its content) so a client that can't
/// connect or calls an unexpected endpoint shows up in Handy's log.
async fn log_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map_or_else(|| "unknown".to_string(), |info| info.0.ip().to_string());
    let started = Instant::now();
    let response = next.run(request).await;
    info!(
        "API {method} {path} from {client}: {} in {:.2?}",
        response.status(),
        started.elapsed()
    );
    response
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    error_with_code(status, message, None)
}

/// An error in OpenAI's shape, which clients know how to show.
fn error_with_code(status: StatusCode, message: impl Into<String>, code: Option<&str>) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    let body = json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "param": null,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::sync::Mutex;
    use tower::ServiceExt;

    const KEY: &str = "test-key";
    const AUDIO: &[u8] = b"\x00\x01not-really-m4a\xff";
    const BOUNDARY: &str = "handy-test-boundary";

    /// Records every upload and answers each with the same result.
    struct FakeBackend {
        key: Option<String>,
        result: Result<Transcript, TranscribeError>,
        uploads: Mutex<Vec<(Vec<u8>, Option<String>)>>,
    }

    impl FakeBackend {
        fn returning(result: Result<Transcript, TranscribeError>) -> Arc<Self> {
            Arc::new(Self {
                key: Some(KEY.to_string()),
                result,
                uploads: Mutex::new(Vec::new()),
            })
        }

        fn without_key() -> Arc<Self> {
            Arc::new(Self {
                key: None,
                result: Ok(transcript()),
                uploads: Mutex::new(Vec::new()),
            })
        }

        fn uploads(&self) -> Vec<(Vec<u8>, Option<String>)> {
            self.uploads.lock().unwrap().clone()
        }
    }

    impl Backend for FakeBackend {
        fn api_key(&self) -> Option<String> {
            self.key.clone()
        }

        fn model_id(&self) -> String {
            "parakeet-test".to_string()
        }

        fn transcribe(
            &self,
            audio: Vec<u8>,
            file_name: Option<&str>,
        ) -> Result<Transcript, TranscribeError> {
            self.uploads
                .lock()
                .unwrap()
                .push((audio, file_name.map(str::to_string)));
            self.result.clone()
        }
    }

    fn transcript() -> Transcript {
        Transcript {
            text: "Hello there. General Kenobi.".to_string(),
            duration_secs: 3725.5,
            segments: vec![
                Segment {
                    start_secs: 0.0,
                    end_secs: 1.25,
                    text: "Hello there.".to_string(),
                },
                Segment {
                    start_secs: 1.25,
                    end_secs: 3725.5,
                    text: "General Kenobi.".to_string(),
                },
            ],
            language: Some("en".to_string()),
        }
    }

    /// A multipart/form-data body of `(field name, file name, content)` parts.
    fn form(parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, file_name, content) in parts {
            body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
            let disposition = match file_name {
                Some(file_name) => format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\n\
                     Content-Type: audio/m4a\r\n\r\n"
                ),
                None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"),
            };
            body.extend_from_slice(disposition.as_bytes());
            body.extend_from_slice(content);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        body
    }

    fn post_form(path: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
        Request::post(path)
            .header("authorization", format!("Bearer {KEY}"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(form(parts)))
            .unwrap()
    }

    /// A transcription request the way OpenAI clients send one.
    fn upload(path: &str, response_format: Option<&str>) -> Request<Body> {
        let mut parts: Vec<(&str, Option<&str>, &[u8])> = vec![
            ("file", Some("recording.m4a"), AUDIO),
            ("model", None, b"whisper-1"),
        ];
        if let Some(format) = response_format {
            parts.push(("response_format", None, format.as_bytes()));
        }
        post_form(path, &parts)
    }

    fn get(path: &str) -> Request<Body> {
        Request::get(path)
            .header("authorization", format!("Bearer {KEY}"))
            .body(Body::empty())
            .unwrap()
    }

    struct Reply {
        status: StatusCode,
        content_type: String,
        body: String,
    }

    impl Reply {
        fn json(&self) -> Value {
            serde_json::from_str(&self.body)
                .unwrap_or_else(|e| panic!("not JSON ({e}): {:?}", self.body))
        }
    }

    async fn send_with_limit(
        backend: Arc<FakeBackend>,
        request: Request<Body>,
        max_upload_bytes: usize,
    ) -> Reply {
        let response = router(backend, max_upload_bytes)
            .oneshot(request)
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .map(|value| value.to_str().unwrap().to_string())
            .unwrap_or_default();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        Reply {
            status,
            content_type,
            body: String::from_utf8(body.to_vec()).unwrap(),
        }
    }

    async fn send(backend: Arc<FakeBackend>, request: Request<Body>) -> Reply {
        send_with_limit(backend, request, 1024 * 1024).await
    }

    #[tokio::test]
    async fn transcribes_an_upload_into_openai_json() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(backend.clone(), upload("/v1/audio/transcriptions", None)).await;

        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.content_type.starts_with("application/json"));
        assert_eq!(
            reply.json(),
            json!({ "text": "Hello there. General Kenobi." })
        );
        assert_eq!(
            backend.uploads(),
            vec![(AUDIO.to_vec(), Some("recording.m4a".to_string()))]
        );
    }

    #[tokio::test]
    async fn base_urls_without_v1_work_too() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let transcribed = send(backend.clone(), upload("/audio/transcriptions", None)).await;
        let listed = send(backend.clone(), get("/models")).await;

        assert_eq!(transcribed.status, StatusCode::OK, "{}", transcribed.body);
        assert_eq!(transcribed.json()["text"], "Hello there. General Kenobi.");
        assert_eq!(listed.status, StatusCode::OK, "{}", listed.body);
        assert_eq!(listed.json()["data"][0]["id"], "parakeet-test");
    }

    #[tokio::test]
    async fn requests_without_the_right_key_are_refused() {
        let backend = FakeBackend::returning(Ok(transcript()));
        for presented in [None, Some("Bearer wrong-key")] {
            for mut request in [upload("/v1/audio/transcriptions", None), get("/v1/models")] {
                let path = request.uri().path().to_string();
                request.headers_mut().remove("authorization");
                if let Some(value) = presented {
                    request
                        .headers_mut()
                        .insert("authorization", value.parse().unwrap());
                }

                let reply = send(backend.clone(), request).await;

                assert_eq!(
                    reply.status,
                    StatusCode::UNAUTHORIZED,
                    "{path} with {presented:?}"
                );
                assert_eq!(reply.json()["error"]["code"], "invalid_api_key");
            }
        }
        assert!(backend.uploads().is_empty(), "nothing may be transcribed");
    }

    #[tokio::test]
    async fn the_key_is_accepted_in_common_header_styles() {
        for (header, value) in [
            ("authorization", "Bearer test-key"),
            ("authorization", "bearer test-key"),
            ("authorization", "test-key"),
            ("x-api-key", "test-key"),
            ("api-key", "test-key"),
        ] {
            let backend = FakeBackend::returning(Ok(transcript()));
            let mut request = upload("/v1/audio/transcriptions", None);
            request.headers_mut().remove("authorization");
            request.headers_mut().insert(header, value.parse().unwrap());

            let reply = send(backend, request).await;

            assert_eq!(reply.status, StatusCode::OK, "{header}: {value}");
        }
    }

    #[tokio::test]
    async fn no_key_is_needed_when_none_is_configured() {
        let backend = FakeBackend::without_key();
        let mut request = upload("/v1/audio/transcriptions", None);
        request.headers_mut().remove("authorization");

        let reply = send(backend.clone(), request).await;

        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert_eq!(backend.uploads().len(), 1);
    }

    #[tokio::test]
    async fn text_format_returns_the_bare_transcript() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(backend, upload("/v1/audio/transcriptions", Some("text"))).await;

        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.content_type.starts_with("text/plain"));
        assert_eq!(reply.body, "Hello there. General Kenobi.");
    }

    #[tokio::test]
    async fn verbose_json_reports_duration_language_and_segments() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(
            backend,
            upload("/v1/audio/transcriptions", Some("verbose_json")),
        )
        .await;

        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let body = reply.json();
        assert_eq!(body["task"], "transcribe");
        assert_eq!(body["language"], "en");
        assert_eq!(body["duration"], 3725.5);
        assert_eq!(body["text"], "Hello there. General Kenobi.");
        let segments = body["segments"].as_array().unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1]["id"], 1);
        assert_eq!(segments[1]["start"], 1.25);
        assert_eq!(segments[1]["end"], 3725.5);
        assert_eq!(segments[1]["text"], "General Kenobi.");
        // Typed clients (e.g. Swift Codable models) require every field
        // OpenAI documents for a segment.
        for field in [
            "seek",
            "tokens",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        ] {
            assert!(segments[0].get(field).is_some(), "segment lacks {field}");
        }
    }

    #[tokio::test]
    async fn subtitle_formats_render_one_cue_per_segment() {
        for (format, expected) in [
            (
                "srt",
                "1\n00:00:00,000 --> 00:00:01,250\nHello there.\n\n\
                 2\n00:00:01,250 --> 01:02:05,500\nGeneral Kenobi.\n",
            ),
            (
                "vtt",
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello there.\n\n\
                 00:00:01.250 --> 01:02:05.500\nGeneral Kenobi.\n",
            ),
        ] {
            let backend = FakeBackend::returning(Ok(transcript()));

            let reply = send(backend, upload("/v1/audio/transcriptions", Some(format))).await;

            assert_eq!(reply.status, StatusCode::OK, "{format}: {}", reply.body);
            assert_eq!(reply.body, expected, "{format}");
        }
    }

    #[tokio::test]
    async fn unknown_formats_are_refused_without_transcribing() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(
            backend.clone(),
            upload("/v1/audio/transcriptions", Some("docx")),
        )
        .await;

        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
        assert!(backend.uploads().is_empty());
    }

    #[tokio::test]
    async fn a_request_without_a_file_is_refused() {
        let backend = FakeBackend::returning(Ok(transcript()));
        let request = post_form(
            "/v1/audio/transcriptions",
            &[
                ("audio", Some("recording.m4a"), AUDIO),
                ("model", None, b"whisper-1"),
            ],
        );

        let reply = send(backend.clone(), request).await;

        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        // Names the fields that did arrive, to make a misbehaving client
        // diagnosable from its error message alone.
        let message = reply.json()["error"]["message"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(message.contains("audio"), "{message}");
        assert!(backend.uploads().is_empty());
    }

    #[tokio::test]
    async fn uploads_over_the_size_limit_are_refused() {
        let backend = FakeBackend::returning(Ok(transcript()));
        let big = vec![7u8; 4096];
        let request = post_form(
            "/v1/audio/transcriptions",
            &[("file", Some("long.wav"), &big)],
        );

        let reply = send_with_limit(backend.clone(), request, 1024).await;

        assert_eq!(reply.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(backend.uploads().is_empty());
    }

    #[tokio::test]
    async fn transcription_errors_map_to_http_statuses() {
        for (error, status, kind) in [
            (
                TranscribeError::BadAudio("not audio".to_string()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                TranscribeError::Unavailable("no model".to_string()),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
            ),
            (
                TranscribeError::Failed("engine exploded".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
            ),
        ] {
            let message = match &error {
                TranscribeError::BadAudio(m)
                | TranscribeError::Unavailable(m)
                | TranscribeError::Failed(m) => m.clone(),
            };
            let backend = FakeBackend::returning(Err(error));

            let reply = send(backend, upload("/v1/audio/transcriptions", None)).await;

            assert_eq!(reply.status, status, "{message}");
            let body = reply.json();
            assert_eq!(body["error"]["type"], kind, "{message}");
            assert_eq!(body["error"]["message"], message.as_str());
        }
    }

    #[tokio::test]
    async fn lists_the_model_that_transcribes() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(backend, get("/v1/models")).await;

        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let body = reply.json();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"].as_array().unwrap().len(), 1);
        assert_eq!(body["data"][0]["id"], "parakeet-test");
        assert_eq!(body["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn the_root_answers_without_a_key() {
        // Lets a phone's browser confirm the server is reachable.
        let backend = FakeBackend::returning(Ok(transcript()));
        let request = Request::get("/").body(Body::empty()).unwrap();

        let reply = send(backend, request).await;

        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("Handy"), "{}", reply.body);
    }

    #[tokio::test]
    async fn unknown_endpoints_get_an_openai_style_error() {
        let backend = FakeBackend::returning(Ok(transcript()));

        let reply = send(backend, get("/v1/chat/completions")).await;

        assert_eq!(reply.status, StatusCode::NOT_FOUND);
        let message = reply.json()["error"]["message"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(message.contains("/v1/chat/completions"), "{message}");
    }
}
