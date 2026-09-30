import { create } from "zustand";
import {
  commands,
  events,
  type FileTranscriptionEvent,
  type StartFileTranscriptionError,
} from "@/bindings";

export type FileTranscriptionPhase =
  | "idle"
  | "reading"
  | "transcribing"
  | "completed"
  | "failed"
  | "cancelled";

interface FileTranscriptionStore {
  phase: FileTranscriptionPhase;
  fileName: string | null;
  durationSecs: number | null;
  completedChunks: number;
  totalChunks: number;
  /** Transcript so far while running; the final transcript once completed. */
  text: string;
  /** Backend error detail when `phase` is "failed". */
  error: string | null;
  cancelling: boolean;
  initialized: boolean;

  initialize: () => void;
  /** Resolves to the reason the backend refused to start, or null. */
  start: (path: string) => Promise<StartFileTranscriptionError | null>;
  cancel: () => Promise<void>;
}

const fileNameOf = (path: string) => path.split(/[\\/]/).pop() ?? path;

export const isFileTranscriptionBusy = (phase: FileTranscriptionPhase) =>
  phase === "reading" || phase === "transcribing";

export const useFileTranscriptionStore = create<FileTranscriptionStore>()(
  (set, get) => ({
    phase: "idle",
    fileName: null,
    durationSecs: null,
    completedChunks: 0,
    totalChunks: 0,
    text: "",
    error: null,
    cancelling: false,
    initialized: false,

    initialize: () => {
      if (get().initialized) return;
      set({ initialized: true });

      events.fileTranscriptionEvent.listen(({ payload }) =>
        set(applyEvent(payload)),
      );
    },

    start: async (path) => {
      const previous = get();
      // Nothing started; put the page back the way it was.
      const restore = () =>
        set({
          phase: previous.phase,
          fileName: previous.fileName,
          durationSecs: previous.durationSecs,
          completedChunks: previous.completedChunks,
          totalChunks: previous.totalChunks,
          text: previous.text,
          error: previous.error,
        });

      set({
        phase: "reading",
        fileName: fileNameOf(path),
        durationSecs: null,
        completedChunks: 0,
        totalChunks: 0,
        text: "",
        error: null,
        cancelling: false,
      });

      try {
        const result = await commands.startFileTranscription(path);
        if (result.status === "error") {
          restore();
          return result.error;
        }
        return null;
      } catch (error) {
        restore();
        throw error;
      }
    },

    cancel: async () => {
      set({ cancelling: true });
      await commands.cancelFileTranscription();
    },
  }),
);

export function applyEvent(
  event: FileTranscriptionEvent,
): (state: FileTranscriptionStore) => Partial<FileTranscriptionStore> {
  switch (event.status) {
    case "started":
      return () => ({
        phase: "transcribing",
        fileName: event.file_name,
        durationSecs: event.duration_secs,
        totalChunks: event.total_chunks,
        completedChunks: 0,
        text: "",
      });
    case "progress":
      return (state) => ({
        completedChunks: event.completed_chunks,
        totalChunks: event.total_chunks,
        text: [state.text, event.text].filter(Boolean).join(" "),
      });
    case "completed":
      return () => ({
        phase: "completed",
        text: event.text,
        cancelling: false,
      });
    case "failed":
      return () => ({ phase: "failed", error: event.error, cancelling: false });
    case "cancelled":
      return () => ({ phase: "cancelled", cancelling: false });
  }
}
