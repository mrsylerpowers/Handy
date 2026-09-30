import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { Check, Copy, Download, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { commands, type StartFileTranscriptionError } from "@/bindings";
import { useModelStore } from "@/stores/modelStore";
import {
  isFileTranscriptionBusy,
  useFileTranscriptionStore,
} from "@/stores/fileTranscriptionStore";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { copyToClipboard } from "../history/clipboard";

/** Extensions the backend's symphonia build decodes (see file_decode.rs). */
const SUPPORTED_EXTENSIONS = [
  "mp3",
  "wav",
  "m4a",
  "m4b",
  "aac",
  "mp4",
  "mov",
  "flac",
  "ogg",
  "oga",
  "aif",
  "aiff",
];
const SUPPORTED_FORMATS_LABEL =
  "MP3 · WAV · M4A · AAC · FLAC · OGG · AIFF · MP4 · MOV";

const START_ERROR_KEYS: Record<StartFileTranscriptionError, string> = {
  already_running: "settings.transcribeFile.errors.alreadyRunning",
  recording_in_progress: "settings.transcribeFile.errors.recordingInProgress",
  no_model: "settings.transcribeFile.errors.noModel",
};

const formatDuration = (totalSecs: number) => {
  const secs = Math.round(totalSecs);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = String(secs % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
};

export const TranscribeFileSettings: React.FC = () => {
  const { t } = useTranslation();
  const { currentModel, models } = useModelStore();
  const modelName = models.find((m) => m.id === currentModel)?.name;
  const {
    phase,
    fileName,
    durationSecs,
    completedChunks,
    totalChunks,
    text,
    error,
    cancelling,
    start,
    cancel,
  } = useFileTranscriptionStore();
  const busy = isFileTranscriptionBusy(phase);
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);

  const startFile = useCallback(
    async (path: string) => {
      try {
        const refused = await start(path);
        if (refused) toast.error(t(START_ERROR_KEYS[refused]));
      } catch (e) {
        toast.error(t("settings.transcribeFile.errors.startFailed"), {
          description: String(e),
        });
      }
    },
    [start, t],
  );

  // Tauri delivers dropped files (with real paths) to the webview.
  useEffect(() => {
    const unlisten = getCurrentWebview().onDragDropEvent((event) => {
      const payload = event.payload;
      if (payload.type === "enter" || payload.type === "over") {
        setDragging(true);
      } else if (payload.type === "leave") {
        setDragging(false);
      } else {
        setDragging(false);
        const [path] = payload.paths;
        const running = isFileTranscriptionBusy(
          useFileTranscriptionStore.getState().phase,
        );
        if (path && !running) void startFile(path);
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [startFile]);

  const chooseFile = async () => {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: t("settings.transcribeFile.audioFilterName"),
          extensions: SUPPORTED_EXTENSIONS,
        },
      ],
    });
    if (typeof selected === "string") void startFile(selected);
  };

  const copyTranscript = async () => {
    if (await copyToClipboard(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } else {
      toast.error(t("settings.history.copyError"));
    }
  };

  const saveTranscript = async () => {
    const baseName = (fileName ?? "transcript").replace(/\.[^.]+$/, "");
    const path = await save({
      defaultPath: `${baseName}.txt`,
      filters: [
        {
          name: t("settings.transcribeFile.textFilterName"),
          extensions: ["txt"],
        },
      ],
    });
    if (!path) return;
    const result = await commands.saveTranscriptFile(path, text);
    if (result.status === "ok") {
      toast.success(t("settings.transcribeFile.saved"));
    } else {
      toast.error(t("settings.transcribeFile.saveFailed"), {
        description: result.error,
      });
    }
  };

  const percent =
    totalChunks > 0 ? Math.round((completedChunks / totalChunks) * 100) : 0;
  const showTranscript =
    phase === "completed" ||
    ((phase === "transcribing" || phase === "cancelled") && text.length > 0);

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup
        title={t("settings.transcribeFile.title")}
        description={
          modelName
            ? t("settings.transcribeFile.description", { model: modelName })
            : t("settings.transcribeFile.descriptionNoModel")
        }
      >
        <div className="p-4 space-y-4">
          {busy ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <p
                    className="text-sm font-medium truncate"
                    title={fileName ?? undefined}
                  >
                    {fileName}
                    {durationSecs !== null && (
                      <span className="text-mid-gray font-normal">
                        {` · ${formatDuration(durationSecs)}`}
                      </span>
                    )}
                  </p>
                  <p className="text-xs text-mid-gray">
                    {phase === "reading"
                      ? t("settings.transcribeFile.reading")
                      : t("settings.transcribeFile.progress", {
                          completed: completedChunks,
                          total: totalChunks,
                        })}
                  </p>
                </div>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void cancel()}
                  disabled={cancelling}
                >
                  {cancelling
                    ? t("settings.transcribeFile.cancelling")
                    : t("settings.transcribeFile.cancel")}
                </Button>
              </div>
              <div className="h-2 w-full rounded-full bg-mid-gray/20 overflow-hidden">
                <div
                  className={`h-full rounded-full bg-logo-primary transition-[width] duration-300 ${
                    phase === "reading" ? "w-full animate-pulse opacity-40" : ""
                  }`}
                  style={
                    phase === "reading" ? undefined : { width: `${percent}%` }
                  }
                />
              </div>
            </div>
          ) : (
            <div
              className={`flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
                dragging
                  ? "border-logo-primary bg-logo-primary/10"
                  : "border-mid-gray/30"
              }`}
            >
              <FileAudio className="w-10 h-10 text-mid-gray" />
              <p className="text-sm font-medium">
                {t("settings.transcribeFile.dropHere")}
              </p>
              <Button variant="primary-soft" onClick={() => void chooseFile()}>
                {t("settings.transcribeFile.chooseFile")}
              </Button>
              <p className="text-xs text-mid-gray">{SUPPORTED_FORMATS_LABEL}</p>
            </div>
          )}

          {phase === "failed" && (
            <Alert variant="error">
              <p className="font-medium">
                {t("settings.transcribeFile.failedTitle", {
                  file: fileName ?? "",
                })}
              </p>
              {error && <p className="text-xs mt-1 break-words">{error}</p>}
            </Alert>
          )}
          {phase === "cancelled" && (
            <p className="text-xs text-mid-gray">
              {t("settings.transcribeFile.cancelled")}
            </p>
          )}
        </div>
      </SettingsGroup>

      {showTranscript && (
        <SettingsGroup title={t("settings.transcribeFile.transcript")}>
          <div className="p-4 space-y-3">
            {text ? (
              <textarea
                readOnly
                value={text}
                aria-label={t("settings.transcribeFile.transcript")}
                className="w-full h-64 resize-y rounded-md border border-mid-gray/30 bg-mid-gray/5 px-3 py-2 text-sm leading-relaxed select-text cursor-text focus:outline-none focus:border-logo-primary"
              />
            ) : (
              <p className="text-sm text-mid-gray">
                {t("settings.transcribeFile.noSpeech")}
              </p>
            )}
            {text && !busy && (
              <div className="flex justify-end gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  className="flex items-center gap-1.5"
                  onClick={() => void copyTranscript()}
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  {copied
                    ? t("settings.transcribeFile.copied")
                    : t("common.copy")}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="flex items-center gap-1.5"
                  onClick={() => void saveTranscript()}
                >
                  <Download className="w-3.5 h-3.5" />
                  {t("settings.transcribeFile.save")}
                </Button>
              </div>
            )}
          </div>
        </SettingsGroup>
      )}
    </div>
  );
};
