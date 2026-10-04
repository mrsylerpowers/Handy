use crate::api_server::{generate_api_key, ApiServerManager, ApiServerStatus};
use crate::settings::{get_settings, write_settings};
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

#[tauri::command]
#[specta::specta]
pub fn get_api_server_status(manager: State<'_, Arc<ApiServerManager>>) -> ApiServerStatus {
    manager.status()
}

/// Turn the API server on or off. Turning it on for the first time also
/// generates its API key. Returns once the server has started or stopped.
#[tauri::command]
#[specta::specta]
pub async fn change_api_server_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<ApiServerStatus, String> {
    let mut settings = get_settings(&app);
    settings.api_server_enabled = enabled;
    if enabled && settings.api_server_key.is_empty() {
        settings.api_server_key = generate_api_key().into();
    }
    write_settings(&app, settings);
    Ok(apply(&app).await)
}

/// Move the API server to another port, restarting it if it is running.
#[tauri::command]
#[specta::specta]
pub async fn change_api_server_port_setting(
    app: AppHandle,
    port: u16,
) -> Result<ApiServerStatus, String> {
    if port == 0 {
        return Err("Port must be between 1 and 65535".to_string());
    }
    let mut settings = get_settings(&app);
    settings.api_server_port = port;
    write_settings(&app, settings);
    Ok(apply(&app).await)
}

/// Set the key API clients must present; empty means none is required. Takes
/// effect with the next request.
#[tauri::command]
#[specta::specta]
pub fn change_api_server_key_setting(app: AppHandle, key: String) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.api_server_key = key.trim().to_string().into();
    write_settings(&app, settings);
    Ok(())
}

/// Replace the API key with a new random one, which is returned.
#[tauri::command]
#[specta::specta]
pub fn regenerate_api_server_key(app: AppHandle) -> Result<String, String> {
    let key = generate_api_key();
    let mut settings = get_settings(&app);
    settings.api_server_key = key.clone().into();
    write_settings(&app, settings);
    Ok(key)
}

async fn apply(app: &AppHandle) -> ApiServerStatus {
    let manager = Arc::clone(&app.state::<Arc<ApiServerManager>>());
    manager.apply_settings().await
}
