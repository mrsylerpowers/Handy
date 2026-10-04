import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Copy, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import {
  commands,
  events,
  type ApiServerError,
  type ApiServerStatus,
} from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { useModelStore } from "@/stores/modelStore";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";
import { SettingContainer } from "../../ui/SettingContainer";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { copyToClipboard } from "../history/clipboard";

const DEFAULT_PORT = 42639;

const blurOnEnter = (event: React.KeyboardEvent<HTMLInputElement>) => {
  if (event.key === "Enter") event.currentTarget.blur();
};

/** Copies `value`, confirming with a check mark for a moment. */
const CopyButton: React.FC<{ value: string }> = ({ value }) => {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (await copyToClipboard(value)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } else {
      toast.error(t("settings.history.copyError"));
    }
  };

  return (
    <Button
      variant="secondary"
      size="sm"
      className="flex items-center gap-1.5 shrink-0"
      onClick={() => void copy()}
      disabled={!value}
    >
      {copied ? (
        <Check className="w-3.5 h-3.5" />
      ) : (
        <Copy className="w-3.5 h-3.5" />
      )}
      {copied ? t("settings.apiServer.copied") : t("common.copy")}
    </Button>
  );
};

export const ApiServerSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating, refreshSettings } =
    useSettings();
  const { currentModel, models } = useModelStore();
  const modelName = models.find((m) => m.id === currentModel)?.name;

  const enabled = getSetting("api_server_enabled") ?? false;
  const port = getSetting("api_server_port") ?? DEFAULT_PORT;
  const apiKey = getSetting("api_server_key") ?? "";

  const [status, setStatus] = useState<ApiServerStatus | null>(null);
  const [portDraft, setPortDraft] = useState(String(port));
  const [keyDraft, setKeyDraft] = useState(apiKey);

  useEffect(() => setPortDraft(String(port)), [port]);
  useEffect(() => setKeyDraft(apiKey), [apiKey]);

  useEffect(() => {
    void commands.getApiServerStatus().then(setStatus);
    const unlisten = events.apiServerStatus.listen((event) =>
      setStatus(event.payload),
    );
    return () => {
      void unlisten.then((fn) => fn());
    };
  }, []);

  const toggle = async (on: boolean) => {
    await updateSetting("api_server_enabled", on);
    // Turning the server on for the first time generates its key.
    await refreshSettings();
  };

  const commitPort = () => {
    const value = Number(portDraft.trim());
    if (!Number.isInteger(value) || value < 1 || value > 65535) {
      toast.error(t("settings.apiServer.errors.invalidPort"));
      setPortDraft(String(port));
      return;
    }
    if (value !== port) void updateSetting("api_server_port", value);
  };

  const commitKey = () => {
    const value = keyDraft.trim();
    if (value !== apiKey) void updateSetting("api_server_key", value);
  };

  const regenerateKey = async () => {
    const result = await commands.regenerateApiServerKey();
    if (result.status === "ok") {
      await refreshSettings();
    } else {
      toast.error(result.error);
    }
  };

  const describeError = (error: ApiServerError, errorPort: number) => {
    switch (error.kind) {
      case "port_in_use":
        return t("settings.apiServer.errors.portInUse", { port: errorPort });
      case "port_unavailable":
        return t("settings.apiServer.errors.portUnavailable", {
          port: errorPort,
        });
      case "failed":
        return t("settings.apiServer.errors.failed", {
          message: error.message,
        });
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup
        title={t("settings.apiServer.title")}
        description={
          modelName
            ? t("settings.apiServer.description", { model: modelName })
            : t("settings.apiServer.descriptionNoModel")
        }
      >
        <ToggleSwitch
          checked={enabled}
          onChange={(on) => void toggle(on)}
          isUpdating={isUpdating("api_server_enabled")}
          label={t("settings.apiServer.enable.label")}
          description={t("settings.apiServer.enable.description")}
          descriptionMode="inline"
          grouped
        />
        <SettingContainer
          title={t("settings.apiServer.port.title")}
          description={t("settings.apiServer.port.description")}
          descriptionMode="inline"
          grouped
        >
          <Input
            type="number"
            min={1}
            max={65535}
            value={portDraft}
            onChange={(event) => setPortDraft(event.target.value)}
            onBlur={commitPort}
            onKeyDown={blurOnEnter}
            disabled={isUpdating("api_server_port")}
            aria-label={t("settings.apiServer.port.title")}
            className="w-28"
          />
        </SettingContainer>
        <SettingContainer
          title={t("settings.apiServer.apiKey.title")}
          description={t("settings.apiServer.apiKey.description")}
          descriptionMode="inline"
          layout="stacked"
          grouped
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Input
                value={keyDraft}
                onChange={(event) => setKeyDraft(event.target.value)}
                onBlur={commitKey}
                onKeyDown={blurOnEnter}
                spellCheck={false}
                autoComplete="off"
                aria-label={t("settings.apiServer.apiKey.title")}
                className="flex-1 min-w-0 font-mono"
              />
              <CopyButton value={apiKey} />
              <Button
                variant="secondary"
                size="sm"
                className="flex items-center gap-1.5 shrink-0"
                onClick={() => void regenerateKey()}
              >
                <RefreshCw className="w-3.5 h-3.5" />
                {t("settings.apiServer.apiKey.regenerate")}
              </Button>
            </div>
            {enabled && !apiKey && (
              <Alert variant="warning">
                {t("settings.apiServer.apiKey.emptyWarning")}
              </Alert>
            )}
          </div>
        </SettingContainer>
      </SettingsGroup>

      {enabled && (
        <SettingsGroup title={t("settings.apiServer.connect.title")}>
          <div className="p-4 space-y-4">
            {status?.error ? (
              <Alert variant="error">
                {describeError(status.error, status.port)}
              </Alert>
            ) : status?.running ? (
              <>
                <p className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  {t("settings.apiServer.connect.running", {
                    port: status.port,
                  })}
                </p>
                <div className="space-y-2">
                  <p className="text-xs font-medium text-mid-gray uppercase tracking-wide">
                    {t("settings.apiServer.connect.baseUrl")}
                  </p>
                  {status.base_urls.map((url) => (
                    <div key={url} className="flex items-center gap-2">
                      <code
                        className="flex-1 min-w-0 truncate rounded-md bg-mid-gray/10 px-2 py-1 text-sm select-text"
                        title={url}
                      >
                        {url}
                      </code>
                      <CopyButton value={url} />
                    </div>
                  ))}
                </div>
                <ol className="list-decimal ps-5 space-y-1 text-sm">
                  <li>{t("settings.apiServer.connect.steps.provider")}</li>
                  <li>{t("settings.apiServer.connect.steps.baseUrl")}</li>
                  <li>{t("settings.apiServer.connect.steps.apiKey")}</li>
                </ol>
                <p className="text-xs text-mid-gray">
                  {t("settings.apiServer.connect.network")}
                </p>
              </>
            ) : (
              <p className="text-sm text-mid-gray">
                {t("settings.apiServer.connect.starting")}
              </p>
            )}
          </div>
        </SettingsGroup>
      )}
    </div>
  );
};
