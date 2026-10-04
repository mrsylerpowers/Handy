//! Decodes an audio or video file into the 16 kHz mono samples the
//! transcription engines expect.

use anyhow::{anyhow, bail, Context, Result};
use log::{debug, warn};
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::resampler::FrameResampler;
use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

/// Decode the first playable audio track of `path` to 16 kHz mono.
///
/// Supports whatever the bundled symphonia decoders cover: WAV, AIFF, CAF,
/// MP3, AAC/M4A (and the AAC audio of MP4/MOV video), Apple Lossless, FLAC and
/// Ogg Vorbis. Audio is resampled as it decodes, so memory stays proportional
/// to the 16 kHz output rather than the source rate.
pub fn decode_audio_file(path: &Path) -> Result<Vec<f32>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let extension = path.extension().and_then(|e| e.to_str());
    decode(Box::new(file), extension, &path.display().to_string())
}

/// Like [`decode_audio_file`], for a file held in memory (e.g. an upload).
/// `extension`, taken from the file's name if it has one, is only a hint: the
/// container is recognized from the data itself.
pub fn decode_audio_bytes(bytes: Vec<u8>, extension: Option<&str>) -> Result<Vec<f32>> {
    decode(Box::new(Cursor::new(bytes)), extension, "uploaded audio")
}

/// `label` names the source in log messages.
fn decode(source: Box<dyn MediaSource>, extension: Option<&str>, label: &str) -> Result<Vec<f32>> {
    let stream = MediaSourceStream::new(source, Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("Unrecognized or unsupported file format ({e})"))?;
    let mut format = probed.format;

    // Video containers usually list the video track first, so take the first
    // track this build can actually decode rather than the default track.
    let codecs = symphonia::default::get_codecs();
    let (track_id, mut decoder) = format
        .tracks()
        .iter()
        .filter(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .find_map(|track| {
            codecs
                .make(&track.codec_params, &DecoderOptions::default())
                .ok()
                .map(|decoder| (track.id, decoder))
        })
        .ok_or_else(|| anyhow!("No supported audio track found (unsupported audio codec?)"))?;

    let mut resampler: Option<FrameResampler> = None;
    let mut source_rate = 0u32;
    let mut source_samples = 0usize;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut mono: Vec<f32> = Vec::new();
    let mut output: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            // A chained stream (e.g. concatenated Ogg files): keep the first.
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => bail!("Failed to read audio: {e}"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet costs a few milliseconds of audio, not the file.
            Err(SymphoniaError::DecodeError(e)) => {
                warn!("Skipping undecodable audio packet: {e}");
                continue;
            }
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(e) => bail!("Failed to decode audio: {e}"),
        };
        if decoded.frames() == 0 {
            continue;
        }

        let spec = *decoded.spec();
        let channels = spec.channels.count();
        if spec.rate == 0 || channels == 0 {
            bail!(
                "The audio stream has an invalid format ({} Hz, {channels} channels)",
                spec.rate
            );
        }
        let needed = decoded.capacity() * channels;
        let buf = match &mut sample_buf {
            Some(buf) if buf.capacity() >= needed => buf,
            _ => sample_buf.insert(SampleBuffer::new(decoded.capacity() as u64, spec)),
        };
        buf.copy_interleaved_ref(decoded);

        mono.clear();
        if channels == 1 {
            mono.extend_from_slice(buf.samples());
        } else {
            mono.extend(
                buf.samples()
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }

        let resampler = resampler.get_or_insert_with(|| {
            source_rate = spec.rate;
            debug!(
                "Decoding {}: {} Hz, {} channel(s)",
                label, spec.rate, channels
            );
            FrameResampler::new(
                spec.rate as usize,
                WHISPER_SAMPLE_RATE as usize,
                Duration::from_millis(30),
            )
        });
        if spec.rate != source_rate {
            warn!(
                "Sample rate changed mid-stream ({} -> {} Hz); keeping {} Hz",
                source_rate, spec.rate, source_rate
            );
        }
        source_samples += mono.len();
        resampler.push(&mono, |frame| output.extend_from_slice(frame));
    }

    let Some(mut resampler) = resampler else {
        bail!("No audio could be decoded from this file");
    };
    resampler.finish(|frame| output.extend_from_slice(frame));
    // The resampler emits whole frames; drop the zero padding past the real end.
    let exact_len = (source_samples as u64 * u64::from(WHISPER_SAMPLE_RATE))
        .div_ceil(u64::from(source_rate)) as usize;
    output.truncate(exact_len);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_temp(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn write_wav(
        dir: &tempfile::TempDir,
        name: &str,
        sample_rate: u32,
        channels: u16,
        interleaved: &[f32],
    ) -> PathBuf {
        let path = dir.path().join(name);
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for &s in interleaved {
            writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        writer.finalize().unwrap();
        path
    }

    fn sine(sample_rate: u32, secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    /// RMS of the middle of the signal, away from codec start/end transients.
    fn middle_rms(samples: &[f32]) -> f32 {
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        (middle.iter().map(|s| s * s).sum::<f32>() / middle.len() as f32).sqrt()
    }

    /// Every fixture is ~1 s of a 440 Hz sine at amplitude 0.5 (RMS 0.354).
    fn assert_one_second_tone(name: &str, samples: &[f32]) {
        assert!(
            (14_400..=17_600).contains(&samples.len()),
            "{name}: expected ~16000 samples at 16 kHz, got {}",
            samples.len()
        );
        let rms = middle_rms(samples);
        assert!(
            (0.30..=0.41).contains(&rms),
            "{name}: expected RMS ~0.354, got {rms}"
        );
    }

    #[test]
    fn common_formats_decode_to_16k_mono() {
        let dir = tempfile::tempdir().unwrap();
        let fixtures: [(&str, &[u8]); 4] = [
            ("tone.mp3", include_bytes!("testdata/tone.mp3")),
            ("tone.m4a", include_bytes!("testdata/tone.m4a")),
            ("tone.ogg", include_bytes!("testdata/tone.ogg")), // stereo vorbis
            ("tone.flac", include_bytes!("testdata/tone.flac")), // 22.05 kHz
        ];
        for (name, bytes) in fixtures {
            let path = write_temp(&dir, name, bytes);
            let samples = decode_audio_file(&path).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_one_second_tone(name, &samples);
        }
    }

    #[test]
    fn iphone_recording_formats_decode_to_16k_mono() {
        // iPhone apps commonly record linear PCM in a CAF container, or Apple
        // Lossless in M4A.
        let dir = tempfile::tempdir().unwrap();
        let fixtures: [(&str, &[u8]); 2] = [
            ("tone.caf", include_bytes!("testdata/tone.caf")), // 48 kHz PCM
            ("tone_alac.m4a", include_bytes!("testdata/tone_alac.m4a")), // 44.1 kHz
        ];
        for (name, bytes) in fixtures {
            let path = write_temp(&dir, name, bytes);
            let samples = decode_audio_file(&path).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_one_second_tone(name, &samples);
        }
    }

    #[test]
    fn uploaded_bytes_decode_to_16k_mono() {
        let bytes = include_bytes!("testdata/tone.m4a").to_vec();
        let samples = decode_audio_bytes(bytes, Some("m4a")).unwrap();
        assert_one_second_tone("tone.m4a bytes", &samples);
    }

    #[test]
    fn uploaded_bytes_are_recognized_by_content_not_name() {
        // Uploads can arrive unnamed ("blob") or misnamed; the container is
        // then recognized from the data itself.
        for extension in [None, Some("wav")] {
            let bytes = include_bytes!("testdata/tone.m4a").to_vec();
            let samples = decode_audio_bytes(bytes, extension)
                .unwrap_or_else(|e| panic!("extension {extension:?}: {e:#}"));
            assert_one_second_tone("tone.m4a bytes", &samples);
        }
    }

    #[test]
    fn uploaded_non_audio_bytes_are_an_error() {
        let bytes = b"{\"error\": \"this is JSON, not audio\"}".to_vec();
        assert!(decode_audio_bytes(bytes, Some("wav")).is_err());
    }

    #[test]
    fn video_file_decodes_its_audio_track() {
        // The H.264 video track comes first; the AAC audio track is second.
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "video.mp4", include_bytes!("testdata/video.mp4"));
        let samples = decode_audio_file(&path).unwrap();
        assert_one_second_tone("video.mp4", &samples);
    }

    #[test]
    fn sixteen_khz_mono_passes_through_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let input = sine(16_000, 0.5, 0.25);
        let path = write_wav(&dir, "in.wav", 16_000, 1, &input);

        let samples = decode_audio_file(&path).unwrap();

        assert_eq!(samples.len(), input.len());
        for (i, (got, want)) in samples.iter().zip(&input).enumerate() {
            assert!((got - want).abs() < 1e-3, "sample {i}: {got} vs {want}");
        }
    }

    #[test]
    fn stereo_channels_are_averaged() {
        // Left carries the tone, right is silent: the average halves it. Taking
        // only the left channel, or summing, would keep the full level.
        let dir = tempfile::tempdir().unwrap();
        let left = sine(16_000, 1.0, 0.5);
        let interleaved: Vec<f32> = left.iter().flat_map(|&l| [l, 0.0]).collect();
        let path = write_wav(&dir, "stereo.wav", 16_000, 2, &interleaved);

        let samples = decode_audio_file(&path).unwrap();

        assert_eq!(samples.len(), 16_000);
        let rms = middle_rms(&samples);
        assert!(
            (0.16..=0.19).contains(&rms),
            "expected RMS ~0.177, got {rms}"
        );
    }

    #[test]
    fn resampled_output_has_the_exact_duration() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_wav(&dir, "44k.wav", 44_100, 1, &sine(44_100, 1.0, 0.5));

        let samples = decode_audio_file(&path).unwrap();

        assert_eq!(samples.len(), 16_000);
    }

    #[test]
    fn non_audio_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "notes.mp3", b"this is a shopping list, not audio");
        assert!(decode_audio_file(&path).is_err());
    }

    #[test]
    fn unsupported_codec_is_an_error() {
        // Opus is not among the bundled decoders.
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "tone.opus", include_bytes!("testdata/tone.opus"));
        assert!(decode_audio_file(&path).is_err());
    }

    #[test]
    fn missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(decode_audio_file(&dir.path().join("nope.wav")).is_err());
    }
}
