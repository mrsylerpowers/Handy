//! Splits long 16 kHz mono audio (e.g. a decoded file) into chunks the
//! transcription engines handle well, and drives a transcriber over them.

use anyhow::{Context, Result};
use std::ops::Range;

use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

const SAMPLE_RATE: usize = WHISPER_SAMPLE_RATE as usize;
/// Longest audio handed to a model in one call. Whisper's native window is
/// 30 s, and the ONNX engines (Parakeet, Moonshine, SenseVoice, Canary) are
/// sized for utterance-length input, so nothing longer is sent.
const MAX_CHUNK: usize = 30 * SAMPLE_RATE;
/// Earliest point in a chunk where a cut may land; the quietest spot between
/// here and `MAX_CHUNK` is chosen so cuts fall in pauses, not mid-word.
const MIN_CUT: usize = 20 * SAMPLE_RATE;
/// Cuts stay this far from the end of the audio so the last chunk is never a
/// word fragment that models turn into garbage.
const MIN_TAIL: usize = 5 * SAMPLE_RATE;
/// Loudness is measured over 100 ms windows...
const WINDOW: usize = SAMPLE_RATE / 10;
/// ...stepped every 10 ms when searching for a cut.
const HOP: usize = SAMPLE_RATE / 100;
/// A chunk whose loudest window stays below this RMS (about -50 dBFS) is
/// silence; whisper hallucinates text over silence, so it is skipped.
const SILENCE_RMS: f32 = 0.003;

/// Planned chunks, in order, that contain sound worth transcribing. Chunks are
/// at most 30 s long and contiguous, except that silent stretches are left out.
pub fn plan_chunks(samples: &[f32]) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let end = if samples.len() - start <= MAX_CHUNK {
            samples.len()
        } else {
            let lo = start + MIN_CUT;
            let hi = (start + MAX_CHUNK).min(samples.len() - MIN_TAIL);
            quietest_point(samples, lo, hi)
        };
        if !is_silent(&samples[start..end]) {
            chunks.push(start..end);
        }
        start = end;
    }
    chunks
}

/// Center of the quietest `WINDOW` that fits inside `lo..hi` (the earliest one
/// on ties).
fn quietest_point(samples: &[f32], lo: usize, hi: usize) -> usize {
    let mut best_start = lo;
    let mut best_energy = f64::INFINITY;
    let mut pos = lo;
    while pos + WINDOW <= hi {
        let energy: f64 = samples[pos..pos + WINDOW]
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum();
        if energy < best_energy {
            best_energy = energy;
            best_start = pos;
        }
        pos += HOP;
    }
    best_start + WINDOW / 2
}

/// How a chunked transcription run ended.
#[derive(Debug, PartialEq)]
pub enum ChunkedTranscription {
    Completed(String),
    Cancelled,
}

/// Transcribes `chunks` of `samples` in order, checking `is_cancelled` before
/// each one and reporting `(chunk index, chunk text)` to `on_chunk_done` after
/// it. The first failing chunk aborts the run. Non-blank chunk texts are joined
/// with single spaces.
pub fn transcribe_chunks(
    samples: &[f32],
    chunks: &[Range<usize>],
    is_cancelled: impl Fn() -> bool,
    mut transcribe: impl FnMut(&[f32]) -> Result<String>,
    mut on_chunk_done: impl FnMut(usize, &str),
) -> Result<ChunkedTranscription> {
    let mut texts: Vec<String> = Vec::new();
    for (index, range) in chunks.iter().enumerate() {
        if is_cancelled() {
            return Ok(ChunkedTranscription::Cancelled);
        }
        let text = transcribe(&samples[range.clone()])
            .with_context(|| format!("chunk {} of {} failed", index + 1, chunks.len()))?;
        let text = text.trim();
        on_chunk_done(index, text);
        if !text.is_empty() {
            texts.push(text.to_string());
        }
    }
    Ok(ChunkedTranscription::Completed(texts.join(" ")))
}

fn is_silent(chunk: &[f32]) -> bool {
    chunk.chunks(WINDOW).all(|window| {
        let mean_square = window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32;
        mean_square.sqrt() < SILENCE_RMS
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;

    fn tone(secs: f64, amplitude: f32) -> Vec<f32> {
        let n = (secs * SR as f64) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SR as f32).sin())
            .collect()
    }

    fn silence(secs: f64) -> Vec<f32> {
        vec![0.0; (secs * SR as f64) as usize]
    }

    fn concat(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    #[test]
    fn empty_audio_yields_no_chunks() {
        assert!(plan_chunks(&[]).is_empty());
    }

    #[test]
    fn short_audio_is_a_single_chunk() {
        let audio = tone(10.0, 0.3);
        assert_eq!(plan_chunks(&audio), vec![0..160_000]);
    }

    #[test]
    fn long_audio_is_cut_inside_the_pause() {
        // 24 s of sound, a 1 s pause, then 20 s more: the cut must land in the
        // pause (24 s..25 s), not at a fixed 30 s mark mid-word.
        let audio = concat(&[tone(24.0, 0.3), silence(1.0), tone(20.0, 0.3)]);
        let chunks = plan_chunks(&audio);

        assert_eq!(chunks.len(), 2, "chunks: {chunks:?}");
        assert_eq!(chunks[0].start, 0);
        assert!(
            (384_000..=400_000).contains(&chunks[0].end),
            "cut at {} is outside the pause",
            chunks[0].end
        );
        assert_eq!(chunks[1], chunks[0].end..720_000);
    }

    #[test]
    fn continuous_sound_never_exceeds_thirty_seconds_per_chunk() {
        let audio = tone(95.0, 0.3);
        let chunks = plan_chunks(&audio);

        assert!(chunks.len() >= 4, "chunks: {chunks:?}");
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, 1_520_000);
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "gap or overlap: {chunks:?}");
        }
        for chunk in &chunks {
            assert!(chunk.len() <= 480_000, "chunk too long: {chunk:?}");
        }
    }

    #[test]
    fn final_chunk_is_never_a_tiny_fragment() {
        // 30.2 s with its quietest spot at 29.8 s: cutting there would leave a
        // 0.35 s tail that models turn into garbage.
        let audio = concat(&[tone(29.8, 0.3), silence(0.1), tone(0.3, 0.3)]);
        let chunks = plan_chunks(&audio);

        assert_eq!(chunks.len(), 2, "chunks: {chunks:?}");
        assert_eq!(chunks[1].end, 483_200);
        assert!(chunks[1].len() >= 80_000, "tail too short: {chunks:?}");
    }

    /// Samples 0.0, 1.0, 2.0, ... so a chunk is identifiable by its values.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn joins_chunk_texts_in_order_skipping_blank_ones() {
        let samples = ramp(6);
        let chunks = [0..2, 2..4, 4..6];
        let result = transcribe_chunks(
            &samples,
            &chunks,
            || false,
            |chunk| {
                Ok(match chunk[0] as usize {
                    0 => "Hello there.".to_string(),
                    2 => "  ".to_string(),
                    _ => " General Kenobi. ".to_string(),
                })
            },
            |_, _| {},
        )
        .unwrap();

        assert_eq!(
            result,
            ChunkedTranscription::Completed("Hello there. General Kenobi.".to_string())
        );
    }

    #[test]
    fn hands_each_planned_range_to_the_transcriber() {
        // The gap (3..5) is a skipped silent stretch and must not be sent.
        let samples = ramp(10);
        let mut received: Vec<Vec<f32>> = Vec::new();
        transcribe_chunks(
            &samples,
            &[0..3, 5..10],
            || false,
            |chunk| {
                received.push(chunk.to_vec());
                Ok(String::new())
            },
            |_, _| {},
        )
        .unwrap();

        assert_eq!(
            received,
            vec![vec![0.0, 1.0, 2.0], vec![5.0, 6.0, 7.0, 8.0, 9.0]]
        );
    }

    #[test]
    fn reports_every_finished_chunk() {
        let samples = ramp(3);
        let mut reported: Vec<(usize, String)> = Vec::new();
        transcribe_chunks(
            &samples,
            &[0..1, 1..2, 2..3],
            || false,
            |chunk| Ok(["a", " ", "b "][chunk[0] as usize].to_string()),
            |index, text| reported.push((index, text.to_string())),
        )
        .unwrap();

        assert_eq!(
            reported,
            vec![
                (0, "a".to_string()),
                (1, String::new()),
                (2, "b".to_string())
            ]
        );
    }

    #[test]
    fn stops_before_the_next_chunk_once_cancelled() {
        let samples = ramp(3);
        let cancelled = std::cell::Cell::new(false);
        let mut calls = 0;
        let result = transcribe_chunks(
            &samples,
            &[0..1, 1..2, 2..3],
            || cancelled.get(),
            |_| {
                calls += 1;
                Ok("text".to_string())
            },
            |_, _| cancelled.set(true),
        )
        .unwrap();

        assert_eq!(result, ChunkedTranscription::Cancelled);
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_failed_chunk_aborts_the_run() {
        let samples = ramp(3);
        let mut calls = 0;
        let result = transcribe_chunks(
            &samples,
            &[0..1, 1..2, 2..3],
            || false,
            |chunk| {
                calls += 1;
                if chunk[0] == 1.0 {
                    anyhow::bail!("engine exploded")
                }
                Ok("ok".to_string())
            },
            |_, _| {},
        );

        let err = result.expect_err("a failing chunk must fail the run");
        assert!(format!("{err:#}").contains("engine exploded"), "{err:#}");
        assert_eq!(calls, 2, "no chunk may run after the failure");
    }

    #[test]
    fn brief_sound_in_long_silence_is_kept() {
        // One short word in 30 s of silence: quiet on average, but not silent.
        let audio = concat(&[silence(15.0), tone(0.3, 0.02), silence(14.7)]);
        assert_eq!(plan_chunks(&audio), vec![0..480_000]);
    }

    #[test]
    fn digital_silence_is_not_transcribed() {
        // Sound, 50 s of silence, sound. The all-silent middle chunk must be
        // dropped so whisper cannot hallucinate text over it.
        let audio = concat(&[tone(10.0, 0.3), silence(50.0), tone(10.0, 0.3)]);
        let chunks = plan_chunks(&audio);

        assert_eq!(chunks.len(), 2, "chunks: {chunks:?}");
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[1].end, 1_120_000);
        for chunk in &chunks {
            let inside_silence = chunk.start >= 160_000 && chunk.end <= 960_000;
            assert!(!inside_silence, "silent chunk kept: {chunk:?}");
        }
    }

    #[test]
    fn quiet_speech_level_audio_is_kept() {
        // -40 dBFS peak is a quiet recording, not silence.
        let audio = tone(10.0, 0.01);
        assert_eq!(plan_chunks(&audio), vec![0..160_000]);
    }
}
