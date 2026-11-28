//! Regression test for the DSP chain on simulated noisy speech.
//! Fixtures are built by `tests/make_fixtures.ps1` (Windows TTS + ffmpeg noise generators):
//!     powershell -File tests/make_fixtures.ps1
//!
//! For each noise type: speech + noise at ~5 dB SNR goes through the chain, then we check
//!   * noise-only stretches get much quieter,
//!   * speech keeps its level,
//!   * magnitude SNR against the clean speech improves.

use sordino::{dsp, wavio};
use std::path::PathBuf;

const SR: usize = 48_000;

fn fixture(name: &str) -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/fixtures").join(name);
    if !p.exists() {
        panic!("missing fixture {} - run: powershell -File tests/make_fixtures.ps1", p.display());
    }
    wavio::read_mono_48k(&p).unwrap()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

fn db(x: f64) -> f64 {
    20.0 * (x + 1e-12).log10()
}

/// 20 ms frames where the clean signal is clearly speech.
fn speech_mask(clean: &[f32]) -> Vec<bool> {
    let n = SR / 50;
    clean.chunks(n).map(|c| db(rms(c)) > -45.0).collect()
}

fn masked(x: &[f32], mask: &[bool], want: bool) -> Vec<f32> {
    let n = SR / 50;
    x.chunks(n).zip(mask).filter(|(_, &m)| m == want).flat_map(|(c, _)| c.iter().copied()).collect()
}

/// Gain that makes `noise` sit `snr_db` below the speech level (RMS over speech frames).
fn noise_gain(clean: &[f32], noise: &[f32], mask: &[bool], snr_db: f64) -> f32 {
    let s = rms(&masked(clean, mask, true));
    let nz = rms(&masked(noise, mask, true));
    (s / nz / 10f64.powf(snr_db / 20.0)) as f32
}

/// Delay (samples) of `out` relative to `clean`, from the correlation of their envelopes.
fn find_lag(clean: &[f32], out: &[f32]) -> usize {
    let n = clean.len().min(out.len()) - SR / 10;
    let mut best = (f64::MIN, 0usize);
    // DPDFNet delays by LATENCY_HOPS (50 ms) plus the 4 ms limiter look-ahead, so search past 100 ms.
    for lag in 0..(SR / 10) {
        let dot: f64 = (0..n).step_by(3).map(|i| (clean[i].abs() * out[i + lag].abs()) as f64).sum();
        if dot > best.0 {
            best = (dot, lag);
        }
    }
    best.1
}

/// Phase-insensitive SNR: compares STFT magnitudes of `out` against `clean` over speech frames
/// (the chain legitimately shifts phase via its high-pass and RNNoise). Higher is closer to clean.
fn magnitude_snr(clean: &[f32], out: &[f32], mask: &[bool]) -> f64 {
    use realfft::RealFftPlanner;
    const W: usize = 1024;
    let lag = find_lag(clean, out);
    let n = clean.len().min(out.len() - lag);
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(W);
    let win: Vec<f32> = (0..W).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / W as f32).cos()).collect();
    let mag = |x: &[f32]| -> Vec<f32> {
        let mut buf: Vec<f32> = x.iter().zip(&win).map(|(a, b)| a * b).collect();
        let mut spec = fft.make_output_vec();
        fft.process(&mut buf, &mut spec).unwrap();
        spec.iter().map(|c| c.norm()).collect()
    };
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    let mut pos = 0;
    while pos + W <= n {
        // Only analyse windows that sit entirely on speech frames of the clean signal.
        let f0 = pos / (SR / 50);
        let f1 = (pos + W) / (SR / 50);
        if (f0..=f1).all(|f| mask.get(f).copied().unwrap_or(false)) {
            let c = mag(&clean[pos..pos + W]);
            let o = mag(&out[pos + lag..pos + lag + W]);
            // Allow a single overall gain change per window.
            let a = c.iter().zip(&o).map(|(c, o)| (c * o) as f64).sum::<f64>() / c.iter().map(|c| (c * c) as f64).sum::<f64>().max(1e-12);
            for (c, o) in c.iter().zip(&o) {
                sig += (a * *c as f64).powi(2);
                err += (*o as f64 - a * *c as f64).powi(2);
            }
        }
        pos += W / 2;
    }
    10.0 * (sig / err.max(1e-12)).log10()
}

struct Case {
    noise: &'static str,
    min_suppression_db: f64,
    min_snr_gain_db: f64,
}

fn check_suppression(engine: dsp::Engine) {
    let clean = fixture("clean.wav");
    let mask = speech_mask(&clean);
    let cases = [
        Case { noise: "pink", min_suppression_db: 15.0, min_snr_gain_db: 1.5 },
        Case { noise: "white", min_suppression_db: 15.0, min_snr_gain_db: 1.5 },
        Case { noise: "fan", min_suppression_db: 15.0, min_snr_gain_db: 1.5 },
        Case { noise: "hum", min_suppression_db: 15.0, min_snr_gain_db: 1.5 },
        Case { noise: "clicks", min_suppression_db: 10.0, min_snr_gain_db: 1.5 },
    ];
    let mut failures = vec![];
    for c in &cases {
        let noise = fixture(&format!("noise_{}.wav", c.noise));
        let n = clean.len().min(noise.len());
        let g = noise_gain(&clean[..n], &noise[..n], &mask, 5.0);
        let noisy: Vec<f32> = (0..n).map(|i| clean[i] + noise[i] * g).collect();

        // AGC off so levels are comparable; everything else as shipped.
        let params = dsp::Params::new(true, true, true, false);
        let out = dsp::process_buffer_engine(params, &noisy, 90.0, engine);

        // Noise-only stretch after the speech (skip the speech edge and the end).
        // Only the tail: RNNoise needs to have heard speech once before it classifies a loud
        // stationary noise correctly, which is how it behaves in real use too.
        let tail_start = n - SR * 15 / 10;
        let tail = &noisy[tail_start..n - SR * 3 / 10];
        let tail_out = &out[tail_start..n - SR * 3 / 10];
        let suppression = db(rms(tail)) - db(rms(tail_out));

        // Speech level retention (mask applies to out too; the delay is small next to the 20 ms
        // masks, and `find_lag` aligns the magnitude-SNR comparison exactly).
        let speech_in = db(rms(&masked(&clean, &mask, true)));
        let speech_out = db(rms(&masked(&out[..n], &mask, true)));
        let speech_loss = speech_in - speech_out;

        let snr_in = magnitude_snr(&clean[..n], &noisy, &mask);
        let snr_out = magnitude_snr(&clean[..n], &out, &mask);
        println!(
            "{:7} suppression {:5.1} dB | speech level {:+5.1} dB | SNR {:5.1} -> {:5.1} dB",
            c.noise, suppression, -speech_loss, snr_in, snr_out
        );

        if suppression < c.min_suppression_db {
            failures.push(format!("{}: noise only dropped {:.1} dB (< {})", c.noise, suppression, c.min_suppression_db));
        }
        if speech_loss > 4.0 {
            failures.push(format!("{}: speech lost {:.1} dB (> 4)", c.noise, speech_loss));
        }
        // Clean speech itself scores ~12 dB on this metric (the chain recolours it slightly), so
        // "close to clean" means reaching about 9-10 dB.
        // DPDFNet re-synthesises the voice, so this bin-exact metric scores it lower than RNNoise
        // even though listeners (and DNSMOS / Whisper, see tests/bench) rate it far better. Here it
        // only guards against the voice being wrecked.
        let neural = engine != dsp::Engine::Rnnoise;
        let floor = if neural { 6.0 } else { 9.0 };
        if snr_out < floor {
            failures.push(format!("{}: output only reaches {:.1} dB magnitude-SNR (< {})", c.noise, snr_out, floor));
        }
        if !neural && snr_out - snr_in < c.min_snr_gain_db {
            failures.push(format!("{}: SNR gain {:.1} dB (< {})", c.noise, snr_out - snr_in, c.min_snr_gain_db));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rnnoise_suppresses_noise_and_keeps_speech() {
    check_suppression(dsp::Engine::Rnnoise);
}

#[test]
fn dpdfnet_suppresses_noise_and_keeps_speech() {
    check_suppression(dsp::Engine::Dpdfnet2);
}

#[test]
fn dpdfnet_delay_matches_latency_hops() {
    // The pre-leveler's inverse gain and the whisper-rescue dry signal are delayed by
    // `LATENCY_HOPS`; if a model change alters the real delay they silently misalign.
    let clean = fixture("clean.wav");
    let params = dsp::Params::new(true, true, true, false);
    params.gate.store(false, std::sync::atomic::Ordering::Relaxed);
    let limiter_lookahead = SR / 250; // 4 ms
    let expected = sordino::dpdfnet::LATENCY_HOPS * dsp::FRAME + limiter_lookahead;
    for engine in [dsp::Engine::Dpdfnet2, dsp::Engine::Dpdfnet8] {
        let out = dsp::process_buffer_engine(params.clone(), &clean, 90.0, engine);
        let lag = find_lag(&clean, &out);
        assert!(lag.abs_diff(expected) <= 48, "{engine:?} delays by {lag} samples, expected about {expected}");
    }
}

#[test]
fn clean_speech_passes_through_intact() {
    let clean = fixture("clean.wav");
    let mask = speech_mask(&clean);
    let out = dsp::process_buffer_engine(dsp::Params::new(true, true, true, false), &clean, 90.0, dsp::Engine::Dpdfnet2);
    let loss = db(rms(&masked(&clean, &mask, true))) - db(rms(&masked(&out[..clean.len()], &mask, true)));
    let snr = magnitude_snr(&clean, &out, &mask);
    println!("clean speech: level change {:+.1} dB, magnitude SNR {:.1} dB", -loss, snr);
    assert!(loss < 3.0, "clean speech lost {loss:.1} dB");
    assert!(snr > 6.0, "clean speech distorted: magnitude SNR {snr:.1} dB");
}

#[test]
fn limiter_never_exceeds_ceiling_and_output_is_finite() {
    // Full-scale loud burst plus NaN/inf garbage from a misbehaving device.
    let mut x: Vec<f32> = (0..SR).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
    x[1000] = f32::NAN;
    x[2000] = f32::INFINITY;
    let out = dsp::process_buffer(dsp::Params::new(false, true, true, false), &x);
    assert!(out.iter().all(|s| s.is_finite()));
    assert!(out.iter().all(|s| s.abs() <= 1.0));
    let steady = &out[SR / 2..];
    let peak = steady.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(peak <= 0.9, "limiter ceiling exceeded: {peak}");
}

#[test]
fn streaming_in_odd_chunks_matches_one_shot() {
    let clean = fixture("clean.wav");
    let one = dsp::process_buffer(dsp::Params::new(true, true, true, false), &clean[..SR * 3]);
    let mut chain = dsp::Chain::new(dsp::Params::new(true, true, true, false));
    let mut out = vec![];
    for chunk in clean[..SR * 3].chunks(317) {
        chain.process(chunk, &mut out);
    }
    let n = out.len().min(one.len());
    assert!(n > SR * 2);
    for i in 0..n {
        assert!((out[i] - one[i]).abs() < 1e-5, "mismatch at {i}");
    }
}

#[test]
fn post_speech_silence_has_no_vox_leak_with_agc() {
    // Reproduces the Valorant VOX flashing bug: speech followed by room fan + clicks with AGC ON.
    let clean = fixture("clean.wav");
    let fan = fixture("noise_fan.wav");
    let clicks = fixture("noise_clicks.wav");
    let mask = speech_mask(&clean);
    let n = clean.len().min(fan.len()).min(clicks.len());
    let gf = noise_gain(&clean[..n], &fan[..n], &mask, 8.0);
    let gc = noise_gain(&clean[..n], &clicks[..n], &mask, 10.0);

    // Append 3 s of pure noise tail after speech so the gate has closed and settled.
    let mut noisy: Vec<f32> = (0..n).map(|i| clean[i] + fan[i] * gf + clicks[i] * gc).collect();
    let tail_len = SR * 3;
    for i in 0..tail_len {
        let j = (n - tail_len + i) % n;
        noisy.push(fan[j] * gf + clicks[j] * gc);
    }

    let params = dsp::Params::new(true, true, true, true); // AGC ON, Gate ON (shipped defaults)
    let out = dsp::process_buffer_engine(params, &noisy, 90.0, dsp::Engine::Dpdfnet2);

    // Check the last 2 seconds of noise-only tail: worst 20 ms frame peak must sit well below
    // any game VOX threshold (e.g. Valorant ~ -45 dBFS).
    let tail = &out[out.len() - SR * 2..out.len() - SR / 5];
    let max_frame_peak = tail
        .chunks(SR / 50)
        .map(|c| c.iter().fold(0.0f32, |m, s| m.max(s.abs())))
        .fold(0.0f32, f32::max);
    let peak_db = db(max_frame_peak as f64);
    println!("post-speech tail max 20 ms peak (AGC ON, DPDFNet2): {peak_db:.1} dBFS");
    assert!(peak_db < -65.0, "post-speech noise leaked at {peak_db:.1} dBFS (would trigger VOX)");
}

#[test]
fn quiet_whisper_is_preserved_and_lifted_by_agc() {
    // Simulate a very quiet voice / whisper (-28 dB quieter than normal speech, ~ -48 dBFS)
    // in a quiet room (~ -58 dBFS pink floor).
    let clean = fixture("clean.wav");
    let pink = fixture("noise_pink.wav");
    let mask = speech_mask(&clean);
    let n = clean.len().min(pink.len());
    let whisper_scale = 10f32.powf(-28.0 / 20.0);
    let noise_scale = 10f32.powf(-38.0 / 20.0);
    let noisy: Vec<f32> = (0..n).map(|i| clean[i] * whisper_scale + pink[i] * noise_scale).collect();

    let in_speech_db = db(rms(&masked(&noisy, &mask, true)));
    let params = dsp::Params::new(true, true, true, true);
    let out = dsp::process_buffer_engine(params, &noisy, 90.0, dsp::Engine::Dpdfnet2);
    let out_speech_db = db(rms(&masked(&out[..n], &mask, true)));
    let tail = &out[n - SR..n - SR / 5];
    let tail_peak_db = db(tail.iter().fold(0.0f32, |m, s| m.max(s.abs())) as f64);

    println!("quiet/whisper: in {in_speech_db:.1} -> out {out_speech_db:.1} dBFS (lift {:+.1} dB) | tail peak {tail_peak_db:.1} dBFS", out_speech_db - in_speech_db);
    assert!(out_speech_db > -28.0, "whisper remained too quiet at {out_speech_db:.1} dBFS");
    assert!(out_speech_db - in_speech_db > 15.0, "AGC did not lift whisper enough ({:+.1} dB)", out_speech_db - in_speech_db);
    assert!(tail_peak_db < -65.0, "whisper AGC pumped silence tail to {tail_peak_db:.1} dBFS");
}

#[test]
fn whisper_to_shout_stays_below_limiter_ceiling() {
    let clean = fixture("clean.wav");
    let mask = speech_mask(&clean);
    // Start at the first active speech frame (skipping the leading silence in clean.wav).
    let start = mask.iter().position(|&m| m).unwrap_or(0) * (SR / 50);
    let n = (SR * 2).min(clean.len() - start);
    let speech = &clean[start..start + n];
    let mut x = Vec::with_capacity(n * 2);
    // 2 s of whisper (-28 dB) so AGC boosts strongly, immediately followed by loud shout (+8 dB).
    let w = 10f32.powf(-28.0 / 20.0);
    for &s in speech {
        x.push(s * w);
    }
    for &s in speech {
        x.push((s * 2.5).clamp(-0.98, 0.98));
    }
    let out = dsp::process_buffer_engine(dsp::Params::new(true, true, true, true), &x, 90.0, dsp::Engine::Dpdfnet2);
    assert!(out.iter().all(|s| s.is_finite()));
    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    println!("whisper->shout peak: {peak:.4} ({:.2} dBFS)", db(peak as f64));
    assert!(peak > 0.3, "whisper->shout unexpectedly silent: {peak}");
    assert!(peak <= 0.92, "limiter allowed overshoot during whisper->shout: {peak}");
}
