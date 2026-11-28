//! Voice DSP chain: high-pass -> DPDFNet / RNNoise -> gate -> auto gain -> peak limiter.
//!
//! Everything runs on 48 kHz mono `f32` audio in [-1, 1] in blocks of [`FRAME`] (10 ms) samples,
//! which is what RNNoise expects. The chain is stateful and streaming: feed it any number of
//! samples and it returns whole frames as they become available.

use crate::dpdfnet::{Dpdfnet, Size, LATENCY_HOPS};
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

pub const SAMPLE_RATE: u32 = 48_000;
/// Usable range for the high-pass cutoff (Hz); `0` means "off" where a CLI accepts it.
pub const MIN_HP_HZ: f32 = 10.0;
pub const MAX_HP_HZ: f32 = SAMPLE_RATE as f32 / 2.0 - 100.0;
pub const FRAME: usize = DenoiseState::FRAME_SIZE; // 480 samples = 10 ms

/// Which neural denoiser runs inside the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// RNNoise: tiny and fast, but it garbles speech under loud, close noise such as a fan.
    Rnnoise,
    /// DPDFNet-2: clean voice, light on the CPU.
    Dpdfnet2,
    /// DPDFNet-8: smoother, fewer artifacts, about twice the CPU of DPDFNet-2. The default.
    Dpdfnet8,
}

impl Engine {
    pub fn from_name(s: &str) -> Option<Engine> {
        match s {
            "rnnoise" | "light" => Some(Engine::Rnnoise),
            "dpdfnet2" | "balanced" | "studio" => Some(Engine::Dpdfnet2),
            "dpdfnet8" | "best" => Some(Engine::Dpdfnet8),
            _ => None,
        }
    }

    /// True when the model this engine needs is installed (RNNoise is built in).
    pub fn is_available(self) -> bool {
        match self {
            Engine::Rnnoise => true,
            Engine::Dpdfnet2 => Size::Two.is_installed(),
            Engine::Dpdfnet8 => Size::Eight.is_installed(),
        }
    }

    /// This engine if its model is installed, otherwise the best engine that is
    /// (DPDFNet-8, then DPDFNet-2, then RNNoise).
    pub fn or_best_available(self) -> Engine {
        if self.is_available() {
            return self;
        }
        [Engine::Dpdfnet8, Engine::Dpdfnet2].into_iter().find(|e| e.is_available()).unwrap_or(Engine::Rnnoise)
    }
}

enum Denoiser {
    Rnn {
        st: Box<DenoiseState<'static>>,
        bp: Biquad,
    },
    Dpd {
        model: Box<Dpdfnet>,
        /// Lightweight RNNoise instance used strictly as a sidechain neural VAD on
        /// level-normalized DPDFNet output (< 1% CPU, 0 ms latency, zero extra DLLs).
        side_vad: Box<DenoiseState<'static>>,
        bp: Biquad,
    },
}

struct VadFeatures {
    /// Neural speech probability in 0..1 (RNNoise native or sidechain).
    prob: f32,
    /// Post-denoiser frame RMS.
    rms: f32,
    /// Fraction of frame energy in the vocal / whisper formant band (~300–4500 Hz).
    voice_band_ratio: f32,
}

impl Denoiser {
    fn new(engine: Engine) -> Self {
        let bp = Biquad::bandpass(1200.0, 0.45);
        let size = match engine {
            Engine::Rnnoise => return Denoiser::Rnn { st: DenoiseState::new(), bp },
            Engine::Dpdfnet2 => Size::Two,
            Engine::Dpdfnet8 => Size::Eight,
        };
        match Dpdfnet::new(size) {
            Ok(m) => Denoiser::Dpd { model: Box::new(m), side_vad: DenoiseState::new(), bp },
            Err(e) => {
                log::error!("DPDFNet failed to load ({e}); falling back to RNNoise");
                Denoiser::Rnn { st: DenoiseState::new(), bp }
            }
        }
    }

    /// `input`/`output` are in [-1, 1]. Returns neural VAD probability + spectral cues.
    fn run(&mut self, input: &[f32; FRAME], output: &mut [f32; FRAME]) -> VadFeatures {
        match self {
            Denoiser::Rnn { st, bp } => {
                let mut scaled = [0.0f32; FRAME];
                for i in 0..FRAME {
                    scaled[i] = input[i] * 32768.0;
                }
                let prob = st.process_frame(output, &scaled);
                let mut e_tot = 0.0f32;
                let mut e_band = 0.0f32;
                for s in output.iter_mut() {
                    *s /= 32768.0;
                    e_tot += *s * *s;
                    let b = bp.run(*s);
                    e_band += b * b;
                }
                let rms = (e_tot / FRAME as f32).sqrt();
                let voice_band_ratio = e_band / (e_tot + 1e-12);
                VadFeatures { prob, rms, voice_band_ratio }
            }
            Denoiser::Dpd { model, side_vad, bp } => {
                if model.process(input, output).is_err() {
                    output.copy_from_slice(input);
                    let rms = frame_rms(output);
                    return VadFeatures { prob: 0.5, rms, voice_band_ratio: 0.5 };
                }
                let mut e_tot = 0.0f32;
                let mut e_band = 0.0f32;
                for &s in output.iter() {
                    e_tot += s * s;
                    let b = bp.run(s);
                    e_band += b * b;
                }
                let rms = (e_tot / FRAME as f32).sqrt();
                let voice_band_ratio = e_band / (e_tot + 1e-12);
                // Normalize the cleaned frame into RNNoise's sweet spot (~-22 dBFS, max +36 dB)
                // so soft voices and whispers get an accurate neural VAD score, while true
                // digital silence (< -76 dBFS) stays at 0 without amplifying float residue.
                let mut side_in = [0.0f32; FRAME];
                let mut side_out = [0.0f32; FRAME];
                let prob = if rms < 1.5e-4 {
                    let _ = side_vad.process_frame(&mut side_out, &side_in);
                    0.0
                } else {
                    let gain = (0.08 / rms).clamp(1.0, 64.0) * 32768.0;
                    for i in 0..FRAME {
                        side_in[i] = (output[i] * gain).clamp(-32767.0, 32767.0);
                    }
                    side_vad.process_frame(&mut side_out, &side_in)
                };
                VadFeatures { prob, rms, voice_band_ratio }
            }
        }
    }

    fn is_dpdfnet(&self) -> bool {
        matches!(self, Denoiser::Dpd { .. })
    }
}

#[inline]
fn frame_rms(x: &[f32]) -> f32 {
    (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

#[inline]
fn lin_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-7).log10()
}

#[inline]
fn db_to_lin(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Live-adjustable switches, shared between the UI and the audio thread.
pub struct Params {
    pub denoise: AtomicBool,
    pub highpass: AtomicBool,
    pub limiter: AtomicBool,
    pub agc: AtomicBool,
    pub gate: AtomicBool,
}

impl Params {
    pub fn new(denoise: bool, highpass: bool, limiter: bool, agc: bool) -> Arc<Self> {
        Arc::new(Self {
            denoise: AtomicBool::new(denoise),
            highpass: AtomicBool::new(highpass),
            limiter: AtomicBool::new(limiter),
            agc: AtomicBool::new(agc),
            gate: AtomicBool::new(true),
        })
    }
}

/// 2nd-order RBJ biquad filter (transposed direct form II).
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn highpass(freq: f32) -> Self {
        // Keep the poles inside the unit circle for any requested cutoff (0 or negative would be
        // marginally stable/unstable; Nyquist would zero every coefficient). Below ~10 Hz the f32
        // biquad is numerically poor anyway.
        let freq = freq.clamp(MIN_HP_HZ, MAX_HP_HZ);
        let w0 = 2.0 * PI * freq / SAMPLE_RATE as f32;
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let cw = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cw) / 2.0 / a0,
            b1: -(1.0 + cw) / a0,
            b2: (1.0 + cw) / 2.0 / a0,
            a1: -2.0 * cw / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn bandpass(freq: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * freq / SAMPLE_RATE as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cw = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: -2.0 * cw / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn peaking_eq(freq: f32, q: f32, gain_db: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / SAMPLE_RATE as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cw = w0.cos();
        let a0 = 1.0 + alpha / a;
        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: -2.0 * cw / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: -2.0 * cw / a0,
            a2: (1.0 - alpha / a) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Unity-compensated pre-denoiser leveler + quiet-room whisper cue.
/// Lifts quiet voices/whispers into DPDFNet's training dynamic range (up to +12 dB, bounded by
/// ambient noise floor and peak headroom) before STFT, then divides back out after DPDFNet's
/// [`LATENCY_HOPS`] of lag so overall gain stays exact. Also tracks the ambient floor, input
/// vocal-band ratio, crest factor and envelope modulation so unvoiced whispers that DPDFNet
/// over-suppresses can be rescued cleanly without leaking stationary room noise or keyboard clicks.
struct PreLeveler {
    gain: f32,
    /// `(start, end)` gain ramps applied to the last [`LATENCY_HOPS`] frames, oldest first. The
    /// front entry is the one belonging to the frame DPDFNet is emitting right now.
    applied: VecDeque<(f32, f32)>,
    noise_floor: f32,
    rms_env: f32,
    in_bp: Biquad,
    whisper_streak: u32,
    rescue_blend: f32,
    /// Recent per-frame input level in dB; its spread tells speech (syllabic) from steady noise.
    env_db: [f32; MOD_WIN],
    env_pos: usize,
    env_len: usize,
}

/// Frames (10 ms each) over which envelope modulation is measured.
const MOD_WIN: usize = 30;
/// Minimum standard deviation (dB) of the frame level over [`MOD_WIN`] for a signal to count as
/// speech-like. Stationary noise measures about 0.6-1.9 dB (even narrow-band hiss); whispers
/// measure 2.3+ dB.
const MOD_MIN_DB: f32 = 2.2;

impl PreLeveler {
    const TARGET_RMS: f32 = 0.055; // -25 dBFS into denoiser
    const MAX_GAIN: f32 = 4.0; // +12 dB into denoiser
    const MAX_NOISE_OUT: f32 = 0.035; // never push ambient noise above -29 dBFS into DPDFNet
    const PEAK_CEIL: f32 = 0.70; // keep peak headroom before STFT

    fn new() -> Self {
        Self {
            gain: 1.0,
            applied: std::iter::repeat_n((1.0, 1.0), LATENCY_HOPS).collect(),
            noise_floor: 0.002,
            rms_env: 0.05,
            in_bp: Biquad::bandpass(1800.0, 0.55), // whisper formant / intelligibility band (~600–4800 Hz)
            whisper_streak: 0,
            rescue_blend: 0.0,
            env_db: [0.0; MOD_WIN],
            env_pos: 0,
            env_len: 0,
        }
    }

    /// Standard deviation (dB) of the recent frame level, or `None` until the window has filled.
    fn modulation_db(&self) -> Option<f32> {
        if self.env_len < MOD_WIN {
            return None;
        }
        let mean = self.env_db.iter().sum::<f32>() / MOD_WIN as f32;
        let var = self.env_db.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / MOD_WIN as f32;
        Some(var.sqrt())
    }

    /// Computes this frame's pre-gain and returns
    /// `(pre_gain_ramp, inverse_ramp, rescue_blend)`, each ramp a `(start, end)` pair of linear
    /// gains. `pre_gain_ramp` is applied to the frame going into DPDFNet; `inverse_ramp` is the
    /// ramp that was applied [`LATENCY_HOPS`] frames ago, i.e. to the audio coming out now.
    fn update(&mut self, frame: &[f32], active: bool) -> ((f32, f32), (f32, f32), f32) {
        let inv = self.applied.pop_front().unwrap_or((1.0, 1.0));

        if !active {
            self.gain = 1.0;
            self.whisper_streak = 0;
            self.rescue_blend *= 0.7;
            self.env_len = 0;
            self.applied.push_back((1.0, 1.0));
            return ((1.0, 1.0), inv, 0.0);
        }

        let mut e_tot = 0.0f32;
        let mut e_band = 0.0f32;
        let mut peak = 0.0f32;
        for &s in frame {
            e_tot += s * s;
            let b = self.in_bp.run(s);
            e_band += b * b;
            peak = peak.max(s.abs());
        }
        let rms = (e_tot / frame.len().max(1) as f32).sqrt();
        let band_ratio = e_band / (e_tot + 1e-12);
        let crest = peak / (rms + 1e-9);

        self.env_db[self.env_pos] = lin_to_db(rms + 1e-9);
        self.env_pos = (self.env_pos + 1) % MOD_WIN;
        self.env_len = (self.env_len + 1).min(MOD_WIN);

        // Asymmetric ambient floor tracker: falls quickly on quiet frames, rises very slowly.
        if rms < self.noise_floor {
            self.noise_floor += (rms - self.noise_floor) * 0.08;
        } else {
            self.noise_floor += (rms - self.noise_floor) * 0.0004;
        }
        self.noise_floor = self.noise_floor.clamp(1e-5, 0.15);

        // Active RMS envelope (fast attack, ~1.5 s release).
        if rms > self.rms_env {
            self.rms_env += (rms - self.rms_env) * 0.30;
        } else {
            self.rms_env += (rms - self.rms_env) * 0.007;
        }

        let mut want = if self.rms_env < Self::TARGET_RMS && rms > 1e-4 {
            (Self::TARGET_RMS / self.rms_env.max(rms).max(1e-4)).clamp(1.0, Self::MAX_GAIN)
        } else {
            1.0
        };
        // Don't overload DPDFNet in loud-fan rooms.
        let noise_cap = (Self::MAX_NOISE_OUT / self.noise_floor).clamp(1.0, Self::MAX_GAIN);
        want = want.min(noise_cap);
        // Peak headroom: the boosted frame must never reach the +/-1.0 clamp in front of the STFT
        // (a hard clip is broadband distortion DPDFNet then mistakes for noise). The ramp's start
        // is capped too, not only its end, so a loud syllable after quiet speech is never clipped.
        let peak_cap = if peak > 1e-4 { (Self::PEAK_CEIL / peak).clamp(1.0, Self::MAX_GAIN) } else { Self::MAX_GAIN };
        want = want.min(peak_cap);

        let start = self.gain.min(peak_cap);
        let k = if want < start { 0.45 } else { 0.04 };
        let end = (start + (want - start) * k).min(peak_cap);
        self.gain = end;
        self.applied.push_back((start, end));

        // Quiet-room unvoiced whisper detector:
        // - Must sit at least +8 dB (2.5x) above ambient room floor (rejects fans/AC/hiss/hum)
        // - Quiet level (< -28 dBFS) with moderate crest factor (< 4.8, rejects sharp keystrokes)
        // - Concentrated in whisper formant band (band_ratio >= 0.50)
        // - Level fluctuating like syllables (rejects stationary noise that merely outgrew the
        //   slow floor tracker, e.g. an AC vent or fan switching on)
        // - Sustained for >= 5 frames (50 ms, rejects impulsive 10-30 ms clicks)
        let snr_ok = rms > self.noise_floor * 2.5 && rms > 4e-4 && rms < 0.04;
        let modulated = self.modulation_db().is_some_and(|m| m >= MOD_MIN_DB);
        let whisper_like = snr_ok && crest < 4.8 && band_ratio >= 0.50 && modulated;
        if whisper_like {
            self.whisper_streak = self.whisper_streak.saturating_add(1);
        } else {
            self.whisper_streak = self.whisper_streak.saturating_sub(2);
        }
        let target_rescue = if self.whisper_streak >= 5 { 0.45 } else { 0.0 };
        let rk = if target_rescue > self.rescue_blend { 0.35 } else { 0.25 };
        self.rescue_blend += (target_rescue - self.rescue_blend) * rk;

        ((start, end), inv, self.rescue_blend)
    }
}

/// Look-ahead peak limiter: ceiling -1 dBFS, 4 ms look-ahead, ~120 ms release.
struct Limiter {
    delay: Vec<f32>,
    pos: usize,
    gain: f32,
}

const LIMIT: f32 = 0.891; // -1 dBFS
const LOOKAHEAD: usize = 192; // 4 ms

impl Limiter {
    fn new() -> Self {
        Self { delay: vec![0.0; LOOKAHEAD], pos: 0, gain: 1.0 }
    }

    /// Forgets the look-ahead audio so a limiter that was bypassed doesn't replay stale samples
    /// (and a stale gain reduction) when it is switched back on.
    fn reset(&mut self) {
        self.delay.fill(0.0);
        self.pos = 0;
        self.gain = 1.0;
    }

    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        self.delay[self.pos] = x;
        self.pos = (self.pos + 1) % LOOKAHEAD;
        // The oldest sample is at `pos`. The peak over the whole look-ahead window sets the gain.
        let mut peak = 0.0f32;
        for &s in &self.delay {
            peak = peak.max(s.abs());
        }
        let target = if peak > LIMIT { LIMIT / peak } else { 1.0 };
        if target < self.gain {
            // Reach `target` smoothly within the 4 ms look-ahead window without overshoot.
            self.gain += (target - self.gain) * 0.08;
            if self.gain < target {
                self.gain = target;
            }
        } else {
            self.gain += (target - self.gain) * (1.0 / (0.12 * SAMPLE_RATE as f32));
        }
        (self.delay[self.pos] * self.gain).clamp(-1.0, 1.0)
    }
}

/// dB-domain AGC2 speech leveler + fast upward compressor for whispers and soft syllables.
/// Adapts strictly while the neural speech gate reports active speech, and gently relaxes extreme
/// gain during silence so the next loud word doesn't clip.
struct Agc {
    gain_db: f32,
    upward_db: f32, // fast extra upward compression (0..=UPWARD_MAX_DB) for whispers / quiet endings
    power: f32,     // smoothed active-speech power
}

impl Agc {
    const TARGET_DB: f32 = -20.0; // ~0.10 RMS: clear Discord / Valorant level
    const MAX_GAIN_DB: f32 = 28.0; // enough headroom to lift -50 dBFS whispers cleanly
    const MIN_GAIN_DB: f32 = -9.0;
    const MIN_RMS: f32 = 0.00025; // -72 dBFS: safe because gated by neural VAD
    const UPWARD_KNEE_DB: f32 = -25.0;
    const UPWARD_RATIO: f32 = 0.65; // ~2.85:1 upward compression below knee
    const UPWARD_MAX_DB: f32 = 12.0;
    const TOTAL_MAX_DB: f32 = 34.0;

    fn new() -> Self {
        Self { gain_db: 0.0, upward_db: 0.0, power: 0.0025 }
    }

    /// Current effective linear gain.
    fn eff(&self) -> f32 {
        db_to_lin((self.gain_db + self.upward_db).min(Self::TOTAL_MAX_DB))
    }

    /// 0..1 blend factor for adaptive whisper presence EQ (flat on normal speech).
    fn presence_blend(&self, enabled: bool) -> f32 {
        if !enabled {
            return 0.0;
        }
        ((self.upward_db - 2.0) / 6.0).clamp(0.0, 1.0)
    }

    /// Updates gain in dB and returns the linear gain this frame starts from.
    fn update(&mut self, frame: &[f32], speech_active: bool, enabled: bool) -> f32 {
        let prev = self.eff();
        let p = frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32;
        let frame_rms = p.sqrt();

        if enabled && speech_active && frame_rms > Self::MIN_RMS {
            self.power += (p - self.power) * 0.35;
            let smoothed_db = lin_to_db(self.power.sqrt());
            let want_db = (Self::TARGET_DB - smoothed_db).clamp(Self::MIN_GAIN_DB, Self::MAX_GAIN_DB);
            let delta = want_db - self.gain_db;
            // Cut quickly on loud talkers, boost smoothly on quiet talkers / whispers.
            let step = if delta < 0.0 {
                // Emergency back-off: after a whisper the gain is high, so a normal-level word
                // would slam the limiter for hundreds of ms at the usual -0.6 dB/frame.
                let post_rms = frame_rms * db_to_lin(self.gain_db + self.upward_db);
                let fastest = if post_rms > 0.35 { -3.0 } else { -0.6 };
                (delta * 0.25).max(fastest)
            } else {
                (delta * 0.08).min(0.22)
            };
            self.gain_db = (self.gain_db + step).clamp(Self::MIN_GAIN_DB, Self::MAX_GAIN_DB);

            // Fast upward compression: lifts short whispered callouts and soft word endings within ~25 ms.
            let post_macro_db = lin_to_db(frame_rms) + self.gain_db;
            let want_up = if post_macro_db < Self::UPWARD_KNEE_DB {
                ((Self::UPWARD_KNEE_DB - post_macro_db) * Self::UPWARD_RATIO).min(Self::UPWARD_MAX_DB)
            } else {
                0.0
            };
            let k = if want_up > self.upward_db { 0.35 } else { 0.12 };
            self.upward_db += (want_up - self.upward_db) * k;
        } else if !enabled {
            self.gain_db += (0.0 - self.gain_db) * 0.12;
            self.upward_db += (0.0 - self.upward_db) * 0.2;
        } else {
            // Silence / trailing hold: never boost noise; slowly relax extreme gain (> +6 dB)
            // back toward a safe baseline so a sudden loud syllable won't slam the limiter.
            if self.gain_db > 6.0 {
                self.gain_db = (self.gain_db - 0.03).max(6.0);
            }
            self.upward_db *= 0.80;
            if self.upward_db < 0.05 {
                self.upward_db = 0.0;
            }
        }
        prev
    }
}

/// Hybrid neural-VAD zero-squelch hysteresis gate for all engines (DPDFNet and RNNoise).
/// Closes cleanly down to exact digital zero (`0.0`) after speech ends so downstream game VOX
/// detectors (Valorant, Discord) never flash on residual noise, while still opening on soft
/// unvoiced whispers via sustained vocal-band persistence.
struct Gate {
    gain: f32,
    hold: u32,
    open: bool,
    voiced_streak: u32,
    whisper_streak: u32,
    noise_rms: f32,
}

impl Gate {
    const HOLD_FRAMES: u32 = 28; // 280 ms hold bridges pauses & trailing syllables
    const RELEASE_MUL: f32 = 0.84; // smooth ~150-350 ms release tail
    const ZERO_SNAP: f32 = 1e-4; // -80 dBFS -> snap to exact 0.0 digital silence

    fn new() -> Self {
        Self { gain: 0.0, hold: 0, open: false, voiced_streak: 0, whisper_streak: 0, noise_rms: 1e-5 }
    }

    /// Returns `(gain_start, gain_end, speech_active)`.
    /// `speech_active` is true during active speech (not the trailing hold tail) so AGC never
    /// pumps residual room noise after the user finishes speaking.
    fn update(&mut self, f: &VadFeatures, whisper_rescued: bool) -> (f32, f32, bool) {
        let start = self.gain;

        // 1. Voiced speech trigger: high confidence opens immediately (0-frame lag); moderate
        //    confidence requires 2 consecutive frames (20 ms) to reject stray 1-frame click blips.
        let strong_voiced = f.prob >= 0.65 && f.rms > 3.5e-4;
        if f.prob >= 0.42 && f.rms > 3.5e-4 {
            self.voiced_streak = self.voiced_streak.saturating_add(1);
        } else {
            self.voiced_streak = 0;
        }
        let voiced = strong_voiced || self.voiced_streak >= 2;

        // 2. Sustained quiet / unvoiced whisper trigger (requires 40 ms persistence so 10-30 ms
        //    keyboard ticks / mouse clicks / DPDFNet glitches never open the gate).
        let whisper_frame = f.rms > 5.0e-4 // > -66 dBFS
            && f.rms > self.noise_rms * 3.5
            && (f.prob >= 0.14 || f.voice_band_ratio >= 0.52 || whisper_rescued);
        if whisper_frame {
            self.whisper_streak = self.whisper_streak.saturating_add(1);
        } else {
            self.whisper_streak = 0;
        }
        let whisper_open = whisper_rescued || self.whisper_streak >= 4;

        // 3. Lower stay-open threshold (hysteresis prevents chattering on quiet syllables).
        let stay_open = self.open
            && ((f.prob >= 0.22 && f.rms > 2.5e-4)
                || (f.rms > 4.0e-4 && f.rms > self.noise_rms * 2.5 && (f.prob >= 0.10 || f.voice_band_ratio >= 0.48)));

        let detected = voiced || whisper_open || stay_open;
        if detected {
            self.open = true;
            self.hold = Self::HOLD_FRAMES;
        } else if self.hold > 0 {
            self.hold -= 1;
        } else {
            self.open = false;
            // Track post-denoiser residual floor only during true closed silence.
            let clamped = f.rms.min(0.003);
            self.noise_rms += (clamped - self.noise_rms) * 0.05;
        }

        if self.open {
            self.gain = 1.0;
        } else {
            self.gain *= Self::RELEASE_MUL;
            if self.gain < Self::ZERO_SNAP {
                self.gain = 0.0;
            }
        }

        // AGC adapts on active speech frames (plus a tiny 30 ms grace) — never on trailing silence.
        let speech_active = detected || (self.open && self.hold >= Self::HOLD_FRAMES - 3);
        (start, self.gain, speech_active)
    }
}

/// The complete mic -> clean voice processor.
pub struct Chain {
    params: Arc<Params>,
    hp: Biquad,
    presence: Biquad,
    pre: PreLeveler,
    /// HPF'd input of the last [`LATENCY_HOPS`] frames, oldest first: the dry signal time-aligned
    /// with what DPDFNet is emitting now, for whisper rescue.
    dry_hist: VecDeque<[f32; FRAME]>,
    dn: Denoiser,
    gate: Gate,
    agc: Agc,
    lim: Limiter,
    pending: Vec<f32>,
    wet: f32, // 0..1 crossfade between dry and denoised
    warmup: bool,
    pub vad: f32,
    pub peak: f32, // peak of the last processed frame (post chain)
}

impl Chain {
    pub fn new(params: Arc<Params>) -> Self {
        Self::with_highpass(params, 90.0)
    }

    pub fn with_highpass(params: Arc<Params>, hp_hz: f32) -> Self {
        Self::with_engine(params, hp_hz, Engine::Rnnoise)
    }

    pub fn with_engine(params: Arc<Params>, hp_hz: f32, engine: Engine) -> Self {
        Self {
            params,
            hp: Biquad::highpass(hp_hz),
            presence: Biquad::peaking_eq(3000.0, 0.9, 3.0),
            pre: PreLeveler::new(),
            dry_hist: std::iter::repeat_n([0.0; FRAME], LATENCY_HOPS).collect(),
            dn: Denoiser::new(engine),
            gate: Gate::new(),
            agc: Agc::new(),
            lim: Limiter::new(),
            pending: Vec::with_capacity(FRAME * 4),
            wet: 1.0,
            warmup: true,
            vad: 0.0,
            peak: 0.0,
        }
    }

    /// Feeds samples and appends every finished 10 ms frame to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        let mut done = 0;
        while self.pending.len() - done >= FRAME {
            let mut frame = [0.0f32; FRAME];
            frame.copy_from_slice(&self.pending[done..done + FRAME]);
            done += FRAME;
            self.frame(&mut frame);
            out.extend_from_slice(&frame);
        }
        self.pending.drain(..done);
    }

    fn frame(&mut self, f: &mut [f32; FRAME]) {
        let p = &self.params;
        let hp_on = p.highpass.load(Relaxed);
        let dn_on = p.denoise.load(Relaxed);
        let agc_on = p.agc.load(Relaxed);
        let lim_on = p.limiter.load(Relaxed);
        let gate_on = p.gate.load(Relaxed);

        // Reject NaN/inf from a misbehaving device so they can't poison the filters.
        for s in f.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        if hp_on {
            for s in f.iter_mut() {
                *s = self.hp.run(*s);
            }
        } else {
            self.hp.reset();
        }

        // Pre-level quiet input into DPDFNet's dynamic range, then restore unity gain after its
        // lag so whispers aren't crushed while overall level stays calibrated.
        let use_pre = dn_on && self.dn.is_dpdfnet();
        let ((pre0, pre1), (inv0, inv1), rescue_blend) = self.pre.update(f, use_pre);
        let mut dn_in = *f;
        if use_pre {
            for i in 0..FRAME {
                let t = (i + 1) as f32 / FRAME as f32;
                dn_in[i] *= pre0 + (pre1 - pre0) * t;
            }
        }

        // The denoiser always runs so its state stays warm; the crossfade handles toggling.
        let mut wet = [0.0f32; FRAME];
        let mut feat = self.dn.run(&dn_in, &mut wet);
        if use_pre {
            // Divide out exactly the gain ramp that was applied to the audio now leaving DPDFNet.
            for i in 0..FRAME {
                let t = (i + 1) as f32 / FRAME as f32;
                wet[i] /= inv0 + (inv1 - inv0) * t;
            }
            // Quiet-room unvoiced whisper rescue (attenuation limit): when sustained soft vocal-band
            // speech sits clearly above the ambient room floor (+8 dB) but DPDFNet over-attenuated
            // it as breathy noise, blend a bounded amount of the time-aligned HPF'd frame back in.
            // The detector (`rescue_blend`) sees the present while the audio it acts on is
            // LATENCY_HOPS old, so it effectively looks ahead and catches the word's onset.
            let dry = self.dry_hist.front().copied().unwrap_or([0.0; FRAME]);
            if rescue_blend > 0.02 {
                let wet_rms = frame_rms(&wet);
                let dry_rms = frame_rms(&dry);
                if dry_rms > 4e-4 && wet_rms < dry_rms * 0.5 {
                    let mix = rescue_blend * (1.0 - (wet_rms / dry_rms).clamp(0.0, 1.0));
                    for i in 0..FRAME {
                        wet[i] = wet[i] * (1.0 - mix) + dry[i] * mix;
                    }
                }
            }
            feat.rms = frame_rms(&wet);
        }
        self.dry_hist.pop_front();
        self.dry_hist.push_back(*f);

        if self.warmup {
            // The first frame carries fade-in artifacts.
            self.warmup = false;
            wet.fill(0.0);
            feat.prob = 0.0;
            feat.rms = 0.0;
        }
        self.vad = feat.prob;

        let w0 = self.wet;
        let w1 = if dn_on { 1.0 } else { 0.0 };
        self.wet = w1;
        for i in 0..FRAME {
            let t = (i + 1) as f32 / FRAME as f32;
            let w = w0 + (w1 - w0) * t;
            f[i] = f[i] * (1.0 - w) + wet[i] * w;
        }

        // The speech detector runs every frame for ALL engines, whatever the toggles: the AGC
        // needs it to tell speech from room noise even with the gate or denoising switched off.
        // The denoiser keeps running while "off", so its VAD is valid.
        let whisper_rescued = use_pre && rescue_blend > 0.15;
        let (gate_g0, gate_g1, speech_active) = self.gate.update(&feat, whisper_rescued);
        let (g0, g1) = if dn_on && gate_on { (gate_g0, gate_g1) } else { (1.0, 1.0) };

        // AGC2 runs before the zero-squelch gate multiplication so when the gate is closed (0.0),
        // VB-CABLE receives exact digital silence regardless of AGC gain.
        let a0 = self.agc.update(f, speech_active, agc_on);
        let a1 = self.agc.eff();
        let eq_blend = self.agc.presence_blend(agc_on);

        // Fast 3 ms attack ramp when opening from closed so word onsets are never dulled.
        let open_ramp_len = if g1 > g0 { (SAMPLE_RATE as usize * 3) / 1000 } else { FRAME };
        if !lim_on {
            self.lim.reset();
        }
        let mut peak = 0.0f32;
        for i in 0..FRAME {
            let tf = (i + 1) as f32 / FRAME as f32;
            let tg = ((i + 1) as f32 / open_ramp_len as f32).min(1.0);
            let g = g0 + (g1 - g0) * tg;
            let mut s = f[i] * (a0 + (a1 - a0) * tf);
            if eq_blend > 0.0 {
                let eq = self.presence.run(s);
                s += (eq - s) * eq_blend;
            } else {
                // Keep EQ state warm on silence/normal speech without jumping when blended in.
                let _ = self.presence.run(s);
            }
            s *= g;
            s = if lim_on { self.lim.run(s) } else { s.clamp(-1.0, 1.0) };
            // Guarantee exact digital zero once the gate has fully closed and limiter tail drained.
            if g == 0.0 && s.abs() < 1e-6 {
                s = 0.0;
            }
            f[i] = s;
            peak = peak.max(s.abs());
        }
        self.peak = peak;
    }
}

/// Offline helper: runs a whole buffer through a fresh chain. The output is a whole number of
/// frames and slightly delayed relative to the input (RNNoise latency plus limiter look-ahead).
pub fn process_buffer(params: Arc<Params>, input: &[f32]) -> Vec<f32> {
    process_buffer_hp(params, input, 90.0)
}

pub fn process_buffer_hp(params: Arc<Params>, input: &[f32], hp_hz: f32) -> Vec<f32> {
    process_buffer_engine(params, input, hp_hz, Engine::Rnnoise)
}

pub fn process_buffer_engine(params: Arc<Params>, input: &[f32], hp_hz: f32, engine: Engine) -> Vec<f32> {
    let mut chain = Chain::with_engine(params, hp_hz, engine);
    let mut out = Vec::with_capacity(input.len() + FRAME);
    chain.process(input, &mut out);
    // Flush the tail with silence so the last partial frame is produced.
    let pad = (FRAME - input.len() % FRAME) % FRAME;
    chain.process(&vec![0.0; pad], &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic white noise in [-1, 1] (xorshift), so tests never flake.
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Frames of white noise band-limited around the whisper band, scaled to `rms_db` dBFS.
    /// `am_hz > 0` adds syllable-like amplitude modulation (what real whispers have).
    fn band_noise(rng: &mut Rng, frames: usize, rms_db: f32, am_hz: f32, t0: &mut f32) -> Vec<[f32; FRAME]> {
        let mut bp = Biquad::bandpass(1800.0, 1.0);
        let scale = db_to_lin(rms_db) / 0.3; // bandpassed white noise has ~0.3 rms at unit input
        (0..frames)
            .map(|_| {
                let mut f = [0.0f32; FRAME];
                for s in f.iter_mut() {
                    let am = if am_hz > 0.0 { 1.0 + 0.9 * (2.0 * PI * am_hz * *t0).sin() } else { 1.0 };
                    *s = bp.run(rng.next()) * scale * am;
                    *t0 += 1.0 / SAMPLE_RATE as f32;
                }
                f
            })
            .collect()
    }

    fn quiet_room(rng: &mut Rng, frames: usize) -> Vec<[f32; FRAME]> {
        (0..frames).map(|_| std::array::from_fn(|_| rng.next() * db_to_lin(-65.0) * 1.7)).collect()
    }

    #[test]
    fn stationary_noise_onset_never_triggers_whisper_rescue() {
        let mut rng = Rng(0x1234_5678);
        let mut pre = PreLeveler::new();
        for f in quiet_room(&mut rng, 200) {
            pre.update(&f, true);
        }
        // An AC vent / fan whine switches on: well above the slowly-rising floor tracker for
        // many seconds, in the whisper band, with a flat envelope.
        let mut t = 0.0;
        let mut late_max = 0.0f32;
        for (i, f) in band_noise(&mut rng, 1500, -38.0, 0.0, &mut t).iter().enumerate() {
            let (_, _, blend) = pre.update(f, true);
            // Allow the first window to fill (the onset itself looks like a word onset).
            if i >= 2 * MOD_WIN {
                late_max = late_max.max(blend);
            }
        }
        assert!(late_max < 0.02, "steady noise kept whisper rescue on (blend {late_max})");
    }

    #[test]
    fn syllabic_whisper_still_triggers_whisper_rescue() {
        let mut rng = Rng(0x9e37_79b9);
        let mut pre = PreLeveler::new();
        for f in quiet_room(&mut rng, 200) {
            pre.update(&f, true);
        }
        let mut t = 0.0;
        let mut max_blend = 0.0f32;
        for f in band_noise(&mut rng, 300, -40.0, 4.0, &mut t) {
            let (_, _, blend) = pre.update(&f, true);
            max_blend = max_blend.max(blend);
        }
        assert!(max_blend > 0.15, "modulated whisper never rescued (blend {max_blend})");
    }

    #[test]
    fn pre_gain_never_clips_a_loud_syllable_after_quiet_speech() {
        let mut rng = Rng(7);
        let mut pre = PreLeveler::new();
        let mut t = 0.0;
        // Quiet speech lifts the pre-gain towards its +12 dB maximum ...
        for f in band_noise(&mut rng, 200, -38.0, 4.0, &mut t) {
            pre.update(&f, true);
        }
        assert!(pre.gain > 2.0, "setup failed: gain {}", pre.gain);
        // ... then a near-full-scale frame arrives.
        let loud: [f32; FRAME] = std::array::from_fn(|i| 0.95 * (i as f32 * 0.07).sin());
        let ((s, e), _, _) = pre.update(&loud, true);
        assert!(s * 0.95 <= 1.0 && e * 0.95 <= 1.0, "boosted frame would clip: ramp {s}..{e}");
    }

    #[test]
    fn inverse_gain_is_the_ramp_applied_latency_hops_ago() {
        let mut rng = Rng(99);
        let mut pre = PreLeveler::new();
        let mut t = 0.0;
        let mut applied = vec![];
        for (i, f) in band_noise(&mut rng, 40, -40.0, 3.0, &mut t).iter().enumerate() {
            let (ramp, inv, _) = pre.update(f, true);
            applied.push(ramp);
            let expect = if i >= LATENCY_HOPS { applied[i - LATENCY_HOPS] } else { (1.0, 1.0) };
            assert_eq!(inv, expect, "frame {i}");
        }
    }

    #[test]
    fn highpass_is_stable_for_any_cutoff() {
        for hz in [-10.0, 0.0, 1.0, 10.0, 90.0, 23_900.0, 24_000.0, 1e9] {
            let mut hp = Biquad::highpass(hz);
            // Jury stability triangle (with slack for f32 rounding at very low cutoffs).
            assert!(hp.a2.abs() < 1.0 && hp.a1.abs() <= 1.0 + hp.a2 + 1e-5, "unstable at {hz} Hz: a1 {} a2 {}", hp.a1, hp.a2);
            let y: f32 = (0..48_000).map(|i| hp.run(((i % 97) as f32 - 48.0) / 100.0).abs()).fold(0.0, f32::max);
            assert!(y.is_finite() && y < 4.0, "blew up at {hz} Hz: {y}");
        }
    }

    #[test]
    fn limiter_reset_forgets_stale_audio() {
        let mut lim = Limiter::new();
        for _ in 0..LOOKAHEAD * 2 {
            lim.run(0.8);
        }
        lim.reset();
        let first = lim.run(0.0);
        assert_eq!(first, 0.0, "stale look-ahead audio replayed after reset");
    }

    #[test]
    fn agc_backs_off_fast_when_loud_speech_follows_a_whisper() {
        let mut agc = Agc::new();
        agc.gain_db = 26.0;
        let loud = [0.16f32; FRAME]; // about -16 dBFS
        for _ in 0..12 {
            agc.update(&loud, true, true);
        }
        // Back to a level where the boosted speech no longer hammers the limiter (RMS < 0.35).
        let post_rms = 0.16 * db_to_lin(agc.gain_db + agc.upward_db);
        assert!(post_rms < 0.4, "AGC still at {:+.1} dB after 120 ms of loud speech (post RMS {post_rms})", agc.gain_db);
    }

    #[test]
    fn agc_does_not_pump_room_noise_when_denoise_is_off() {
        // Denoise OFF + Auto Gain ON must not lift a quiet room by tens of dB.
        let engine = Engine::Dpdfnet2;
        if !engine.is_available() {
            return; // model files not present in this environment
        }
        let mut rng = Rng(42);
        let mut chain = Chain::with_engine(Params::new(false, true, true, true), 90.0, engine);
        let n = SAMPLE_RATE as usize * 5;
        let room: Vec<f32> = (0..n).map(|_| rng.next() * db_to_lin(-55.0) * 1.7).collect();
        let mut out = vec![];
        chain.process(&room, &mut out);
        let sec = SAMPLE_RATE as usize;
        let gain_db = lin_to_db(frame_rms(&out[out.len() - sec..])) - lin_to_db(frame_rms(&room[n - sec..]));
        assert!(gain_db < 6.0, "room noise lifted by {gain_db:.1} dB");
    }
}
