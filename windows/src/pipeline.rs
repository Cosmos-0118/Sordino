//! Live audio pipeline: physical mic (WASAPI capture) -> DSP chain -> ring buffer -> virtual cable
//! playback endpoint (WASAPI render).
//!
//! Feedback guard (mirrors the macOS app): the output endpoint is resolved by its stable ID and must
//! be a virtual cable. If it can't be opened, or the mic is itself a virtual cable, nothing starts —
//! processed audio is never sent to a real speaker, and the pipeline never falls back to the default
//! playback device.

use crate::devices::is_virtual_name;
use crate::dsp::{self, Chain, Params, SAMPLE_RATE};
use crate::ring::RingBuffer;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::{Relaxed, Release}};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use wasapi::{DeviceEnumerator, Direction, Role, SampleType, StreamMode, WaveFormat};

pub struct Config {
    /// Endpoint ID of the microphone; `None` uses the default recording device.
    pub input_id: Option<String>,
    /// Endpoint ID of the virtual cable playback side (required).
    pub output_id: String,
    pub params: Arc<Params>,
    pub engine: dsp::Engine,
    /// Test hook: feed this audio (48 kHz mono) instead of capturing a microphone.
    pub source_samples: Option<Arc<Vec<f32>>>,
}

pub struct Shared {
    peak_bits: AtomicU32,
    stop: AtomicBool,
    /// Set once the first processed audio has been written to the virtual mic.
    flowing: AtomicBool,
    error: Mutex<Option<String>>,
    ring: RingBuffer,
}

impl Shared {
    fn set_error(&self, msg: String) {
        let mut e = self.error.lock().unwrap_or_else(|e| e.into_inner());
        if e.is_none() {
            *e = Some(msg);
        }
        self.stop.store(true, Relaxed);
    }
}

pub struct Pipeline {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Pipeline {
    /// Opens both endpoints and starts the audio threads. Blocks briefly while they initialise.
    pub fn start(cfg: Config) -> Result<Pipeline, String> {
        if cfg.output_id.is_empty() {
            return Err("No virtual mic found. Install VB-CABLE from vb-audio.com/Cable".into());
        }
        let shared = Arc::new(Shared {
            peak_bits: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            flowing: AtomicBool::new(false),
            error: Mutex::new(None),
            ring: RingBuffer::new(SAMPLE_RATE),
        });
        let (tx, rx) = mpsc::channel::<Result<(), String>>();
        let mut threads = vec![];

        // Render first: if the virtual mic can't be opened we never touch the microphone.
        {
            let (sh, tx, id) = (shared.clone(), tx.clone(), cfg.output_id.clone());
            threads.push(thread::Builder::new().name("sordino-render".into()).spawn(move || render_thread(sh, id, tx)).unwrap());
        }
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(abort(shared, threads, e)),
            Err(_) => return Err(abort(shared, threads, "Timed out opening the virtual mic".into())),
        }

        {
            let (sh, tx) = (shared.clone(), tx.clone());
            let (input, params, src, engine) = (cfg.input_id.clone(), cfg.params.clone(), cfg.source_samples.clone(), cfg.engine);
            threads.push(
                thread::Builder::new()
                    .name("sordino-capture".into())
                    .spawn(move || match src {
                        Some(samples) => file_source_thread(sh, samples, params, engine, tx),
                        None => capture_thread(sh, input, params, engine, tx),
                    })
                    .unwrap(),
            );
        }
        // The capture thread loads the neural model before opening the mic; right after a reboot
        // (cold disk, busy CPU) that can take several seconds.
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(abort(shared, threads, e)),
            Err(_) => return Err(abort(shared, threads, "Timed out opening the microphone".into())),
        }
        Ok(Pipeline { shared, threads })
    }

    /// Output peak since the last call (0..1), for the level meter.
    pub fn take_peak(&self) -> f32 {
        f32::from_bits(self.shared.peak_bits.swap(0, Relaxed))
    }

    /// Set once a device disappears or a stream fails; the owner should stop and rebuild.
    pub fn error(&self) -> Option<String> {
        if let Some(e) = self.shared.error.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return Some(e);
        }
        // An audio thread that died (panic) without reporting would leave a silent virtual mic.
        if !self.shared.stop.load(Relaxed) && self.threads.iter().any(|t| t.is_finished()) {
            return Some("Audio thread stopped unexpectedly".into());
        }
        None
    }

    /// True once cleaned audio is actually reaching the virtual mic (model loaded, mic streaming).
    pub fn is_flowing(&self) -> bool {
        self.shared.flowing.load(Relaxed)
    }

    /// (underruns, overflows) of the capture->render ring buffer.
    pub fn ring_stats(&self) -> (u64, u64) {
        self.shared.ring.stats()
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.shared.stop.store(true, Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn abort(shared: Arc<Shared>, threads: Vec<JoinHandle<()>>, msg: String) -> String {
    shared.stop.store(true, Relaxed);
    for t in threads {
        let _ = t.join();
    }
    msg
}

fn bump_peak(shared: &Shared, peak: f32) {
    let cur = f32::from_bits(shared.peak_bits.load(Relaxed));
    if peak > cur {
        shared.peak_bits.store(peak.to_bits(), Relaxed);
    }
}

fn f32_from_bytes(bytes: &[u8], out: &mut Vec<f32>) {
    out.clear();
    out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
}

/// Flushes x86 subnormal (denormal) floats to zero on this thread so silent/quiet passages don't
/// trigger 10x CPU spikes inside GRU / biquad IIR filters while gaming.
fn enable_ftz_daz() {
    #[cfg(target_arch = "x86_64")]
    #[allow(deprecated)]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        // Bit 15 = FTZ (Flush-To-Zero), Bit 6 = DAZ (Denormals-Are-Zero).
        _mm_setcsr(_mm_getcsr() | 0x8040);
    }
}

/// Elevates an audio worker thread via Windows MMCSS ("Pro Audio") so heavy game loads don't
/// preempt capture/render and cause buffer underruns.
struct ProAudioGuard(Option<windows::Win32::Foundation::HANDLE>);

impl ProAudioGuard {
    fn enter() -> Self {
        use windows::core::w;
        use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
        let mut idx = 0u32;
        let h = unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut idx) }.ok();
        Self(h)
    }
}

impl Drop for ProAudioGuard {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            unsafe {
                let _ = windows::Win32::System::Threading::AvRevertMmThreadCharacteristics(h);
            }
        }
    }
}

/// Downmixes multi-channel WASAPI capture without throwing away 6 dB on 1-mic / 2-ch interfaces.
/// If one channel dominates (e.g. mono XLR/USB mic on Ch 1, dead Ch 2), smoothly selects the
/// active channel at unity gain; otherwise averages balanced channels.
struct Downmixer {
    channels: usize,
    power: Vec<f32>,
    weights: Vec<f32>,
}

impl Downmixer {
    fn new(channels: usize) -> Self {
        let ch = channels.max(1);
        Self { channels: ch, power: vec![0.0; ch], weights: vec![1.0 / ch as f32; ch] }
    }

    fn mix(&mut self, bytes: &[u8], out: &mut Vec<f32>) {
        if self.channels == 1 {
            f32_from_bytes(bytes, out);
            return;
        }
        let ch = self.channels;
        let stride = ch * 4;
        let frames = bytes.len() / stride;
        out.clear();
        out.reserve(frames);

        // Estimate per-channel energy over this packet.
        let mut pkt_e = [0.0f32; 8];
        let use_ch = ch.min(8);
        for f in 0..frames {
            let base = f * stride;
            for c in 0..use_ch {
                let o = base + c * 4;
                let s = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                if s.is_finite() {
                    pkt_e[c] += s * s;
                }
            }
        }
        if frames > 0 {
            for c in 0..use_ch {
                let mean = pkt_e[c] / frames as f32;
                self.power[c] += (mean - self.power[c]) * 0.05;
            }
        }

        // If one channel is >12 dB (16x power) louder than the rest, pick it at unity (+6 dB win).
        let (mut best_c, mut best_p) = (0usize, 0.0f32);
        let mut sum_p = 0.0f32;
        for c in 0..use_ch {
            sum_p += self.power[c];
            if self.power[c] > best_p {
                best_p = self.power[c];
                best_c = c;
            }
        }
        let others = (sum_p - best_p).max(0.0);
        let solo = best_p > 1e-10 && best_p > others * 16.0;

        let mut target = [0.0f32; 8];
        if solo {
            target[best_c] = 1.0;
        } else {
            let w = 1.0 / use_ch as f32;
            for c in 0..use_ch {
                target[c] = w;
            }
        }
        for c in 0..use_ch {
            self.weights[c] += (target[c] - self.weights[c]) * 0.1;
        }

        for f in 0..frames {
            let base = f * stride;
            let mut acc = 0.0f32;
            for c in 0..use_ch {
                let o = base + c * 4;
                let s = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                // A dead channel's NaN/Inf must not leak in even at weight 0 (NaN * 0.0 is NaN).
                if s.is_finite() {
                    acc += s * self.weights[c];
                }
            }
            out.push(acc);
        }
    }
}

// ---------------------------------------------------------------- render (virtual mic)

fn render_thread(shared: Arc<Shared>, id: String, ready: mpsc::Sender<Result<(), String>>) {
    let _ = wasapi::initialize_mta();
    let _mmcss = ProAudioGuard::enter();
    let result = (|| -> Result<_, String> {
        let en = DeviceEnumerator::new().map_err(|e| format!("Audio system unavailable: {e}"))?;
        let dev = en.get_device(&id).map_err(|_| "The virtual mic is no longer available. Is VB-CABLE installed?".to_string())?;
        if dev.get_direction() != Direction::Render {
            return Err("Output must be a playback device".into());
        }
        let name = dev.get_friendlyname().unwrap_or_default();
        if !is_virtual_name(&name) {
            return Err(format!("\"{name}\" is a real playback device. Sordino only plays to a virtual mic, to avoid feedback."));
        }
        let mut client = dev.get_iaudioclient().map_err(|e| format!("Couldn't open the virtual mic: {e}"))?;
        let fmt = WaveFormat::new(32, 32, SampleType::Float, SAMPLE_RATE as usize, 2, None);
        let (period, _) = client.get_device_period().map_err(|e| e.to_string())?;
        client
            .initialize_client(&fmt, Direction::Render, StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: period.max(200_000) }, None)
            .map_err(|e| format!("Couldn't route audio to the virtual mic: {e}"))?;
        let event = client.set_get_eventhandle().map_err(|e| e.to_string())?;
        let render = client.get_audiorenderclient().map_err(|e| e.to_string())?;
        // Prefill with silence so playback starts with a full buffer.
        let frames = client.get_available_space_in_frames().map_err(|e| e.to_string())? as usize;
        render.write_to_device(&vec![0u8; frames * 8], None).map_err(|e| e.to_string())?;
        client.start_stream().map_err(|e| e.to_string())?;
        Ok((client, event, render))
    })();

    let (client, event, render) = match result {
        Ok(v) => {
            let _ = ready.send(Ok(()));
            v
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    let mut mono: Vec<f32> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    while !shared.stop.load(Relaxed) {
        if event.wait_for_event(200).is_err() {
            continue; // timeout: nothing to do, re-check the stop flag
        }
        let frames = match client.get_available_space_in_frames() {
            Ok(n) => n as usize,
            Err(e) => {
                shared.set_error(format!("Virtual mic stopped: {e}"));
                break;
            }
        };
        if frames == 0 {
            continue;
        }
        mono.resize(frames, 0.0);
        shared.ring.read(&mut mono);
        bytes.clear();
        for s in &mono {
            let b = s.to_le_bytes();
            bytes.extend_from_slice(&b);
            bytes.extend_from_slice(&b);
        }
        if let Err(e) = render.write_to_device(&bytes, None) {
            shared.set_error(format!("Virtual mic stopped: {e}"));
            break;
        }
    }
    let _ = client.stop_stream();
}

// ---------------------------------------------------------------- capture (physical mic)

fn capture_thread(shared: Arc<Shared>, input: Option<String>, params: Arc<Params>, engine: dsp::Engine, ready: mpsc::Sender<Result<(), String>>) {
    let _ = wasapi::initialize_mta();
    enable_ftz_daz();
    // Load the (slow) neural model first and only then open the mic, so the capture stream never
    // sits unread while the model loads and "ready" really means audio is about to flow. This runs
    // at normal priority: elevating it (MMCSS) would let a boot-time model load starve the audio
    // threads of other apps (browser, games), which then play scrambled audio.
    let mut chain = Chain::with_engine(params, 90.0, engine);
    let _mmcss = ProAudioGuard::enter();
    let result = (|| -> Result<_, String> {
        let en = DeviceEnumerator::new().map_err(|e| format!("Audio system unavailable: {e}"))?;
        let dev = match &input {
            Some(id) => en.get_device(id).map_err(|_| "The selected microphone is not available".to_string())?,
            None => en.get_default_device_for_role(Direction::Capture, Role::Console).map_err(|_| "No microphone found".to_string())?,
        };
        if dev.get_direction() != Direction::Capture {
            return Err("Input must be a recording device".into());
        }
        let name = dev.get_friendlyname().unwrap_or_default();
        if is_virtual_name(&name) {
            return Err(format!("\"{name}\" is a virtual mic. Pick your real microphone in Sordino."));
        }
        let mut client = dev.get_iaudioclient().map_err(|e| format!("Couldn't open the microphone: {e}"))?;
        // Preserve native channel count (up to 8) so 1-mic / 2-ch USB/XLR interfaces don't lose 6 dB
        // to Windows' blind (L+R)/2 mono downmix.
        let ch = client.get_mixformat().map(|f| (f.get_nchannels() as usize).clamp(1, 8)).unwrap_or(1);
        let fmt = WaveFormat::new(32, 32, SampleType::Float, SAMPLE_RATE as usize, ch, None);
        let (period, _) = client.get_device_period().map_err(|e| e.to_string())?;
        client
            .initialize_client(&fmt, Direction::Capture, StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: period.max(200_000) }, None)
            .map_err(|e| format!("Couldn't open the microphone (is it in use or blocked by Windows privacy settings?): {e}"))?;
        let event = client.set_get_eventhandle().map_err(|e| e.to_string())?;
        let cap = client.get_audiocaptureclient().map_err(|e| e.to_string())?;
        client.start_stream().map_err(|e| e.to_string())?;
        Ok((client, event, cap, ch))
    })();

    let (client, event, cap, ch) = match result {
        Ok(v) => {
            let _ = ready.send(Ok(()));
            v
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    let mut downmix = Downmixer::new(ch);
    let mut raw: Vec<u8> = vec![0; 48_000 * 4 * ch]; // up to 1 s per read
    let mut samples: Vec<f32> = Vec::new();
    let mut processed: Vec<f32> = Vec::new();
    while !shared.stop.load(Relaxed) {
        if event.wait_for_event(200).is_err() {
            continue;
        }
        loop {
            match cap.get_next_packet_size() {
                Ok(Some(0)) | Ok(None) => break,
                Ok(Some(_)) => {}
                Err(e) => {
                    shared.set_error(format!("Microphone stopped: {e}"));
                    break;
                }
            }
            match cap.read_from_device(&mut raw) {
                Ok((frames, _)) if frames > 0 => {
                    downmix.mix(&raw[..frames as usize * ch * 4], &mut samples);
                    processed.clear();
                    chain.process(&samples, &mut processed);
                    if !processed.is_empty() {
                        shared.ring.write(&processed);
                        bump_peak(&shared, chain.peak);
                        shared.flowing.store(true, Release);
                    }
                }
                Ok(_) => break,
                Err(e) => {
                    shared.set_error(format!("Microphone stopped: {e}"));
                    break;
                }
            }
        }
    }
    let _ = client.stop_stream();
}

/// Plays a prepared sample buffer through the DSP chain in real time (test/diagnostic source).
fn file_source_thread(shared: Arc<Shared>, samples: Arc<Vec<f32>>, params: Arc<Params>, engine: dsp::Engine, ready: mpsc::Sender<Result<(), String>>) {
    enable_ftz_daz();
    let mut chain = Chain::with_engine(params, 90.0, engine);
    let _ = ready.send(Ok(()));
    let mut out = Vec::new();
    let chunk = dsp::FRAME;
    let silence = vec![0.0f32; chunk];
    let start = std::time::Instant::now();
    let mut pos = 0usize;
    while !shared.stop.load(Relaxed) {
        // Pace to real time, including the silence that keeps the stream alive after the file
        // ends (it must not run ahead of the clock or the ring buffer overflows).
        let due = (start.elapsed().as_secs_f64() * SAMPLE_RATE as f64) as usize;
        while pos < due {
            out.clear();
            if pos < samples.len() {
                let end = (pos + chunk).min(samples.len());
                chain.process(&samples[pos..end], &mut out);
                pos = end;
            } else {
                chain.process(&silence, &mut out);
                pos += chunk;
            }
            if !out.is_empty() {
                shared.ring.write(&out);
                bump_peak(&shared, chain.peak);
                shared.flowing.store(true, Release);
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(ch0: f32, ch1: f32, frames: usize) -> Vec<u8> {
        (0..frames).flat_map(|i| [(ch0 * (i as f32 * 0.1).sin()).to_le_bytes(), ch1.to_le_bytes()]).flatten().collect()
    }

    #[test]
    fn nan_on_a_dead_channel_does_not_poison_the_mix() {
        let mut dm = Downmixer::new(2);
        let mut out = Vec::new();
        for _ in 0..60 {
            out.clear();
            dm.mix(&packet(0.3, f32::NAN, 480), &mut out);
            assert!(out.iter().all(|s| s.is_finite()), "NaN leaked through the downmix");
        }
        assert!(out.iter().any(|s| s.abs() > 0.1), "live channel lost");
    }
}
