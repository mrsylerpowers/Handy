import assert from "node:assert/strict";
import type { FileTranscriptionEvent } from "@/bindings";
import {
  applyEvent,
  useFileTranscriptionStore,
} from "./fileTranscriptionStore";

type State = ReturnType<typeof useFileTranscriptionStore.getState>;

const initial: State = useFileTranscriptionStore.getState();

/** Feed events through the reducer the way zustand's `set` would. */
const run = (state: State, ...events: FileTranscriptionEvent[]): State =>
  events.reduce(
    (current, event) => ({ ...current, ...applyEvent(event)(current) }),
    state,
  );

const started: FileTranscriptionEvent = {
  status: "started",
  file_name: "meeting.mp3",
  duration_secs: 95.5,
  total_chunks: 3,
};

// Chunk texts accumulate in order; silent chunks add no stray spaces.
{
  const state = run(
    initial,
    started,
    {
      status: "progress",
      completed_chunks: 1,
      total_chunks: 3,
      text: "Hello there.",
    },
    { status: "progress", completed_chunks: 2, total_chunks: 3, text: "" },
    {
      status: "progress",
      completed_chunks: 3,
      total_chunks: 3,
      text: "General Kenobi.",
    },
  );
  assert.equal(state.phase, "transcribing");
  assert.equal(state.text, "Hello there. General Kenobi.");
  assert.equal(state.completedChunks, 3);
  assert.equal(state.totalChunks, 3);
}

// A new file starts from a clean slate, not the previous transcript.
{
  const previous = run(
    initial,
    started,
    {
      status: "progress",
      completed_chunks: 1,
      total_chunks: 3,
      text: "old words",
    },
    { status: "completed", text: "old words" },
  );
  const state = run(previous, { ...started, file_name: "next.m4a" });
  assert.equal(state.phase, "transcribing");
  assert.equal(state.fileName, "next.m4a");
  assert.equal(state.text, "");
  assert.equal(state.completedChunks, 0);
}

// The final transcript (e.g. after Chinese variant conversion) wins.
{
  const state = run(
    initial,
    started,
    { status: "progress", completed_chunks: 1, total_chunks: 1, text: "简体" },
    { status: "completed", text: "簡體" },
  );
  assert.equal(state.phase, "completed");
  assert.equal(state.text, "簡體");
}

// Cancelling keeps what was transcribed so far.
{
  const state = run(
    { ...initial, cancelling: true },
    started,
    {
      status: "progress",
      completed_chunks: 1,
      total_chunks: 3,
      text: "partial words",
    },
    { status: "cancelled" },
  );
  assert.equal(state.phase, "cancelled");
  assert.equal(state.text, "partial words");
  assert.equal(state.cancelling, false);
}

// Failures carry the backend's reason.
{
  const state = run(initial, {
    status: "failed",
    error: "Unrecognized format",
  });
  assert.equal(state.phase, "failed");
  assert.equal(state.error, "Unrecognized format");
}

console.log("fileTranscriptionStore: all assertions passed");
