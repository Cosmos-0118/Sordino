//! DPDFNet speech enhancement (Ceva, Apache-2.0): a causal, streaming, 48 kHz full-band model
//! derived from DeepFilterNet2, run with the pure-Rust `tract` ONNX engine.
//!
//! The ONNX graph takes one STFT frame (960-sample Vorbis window, 480 hop, 481 bins as re/im) plus a
//! recurrent state vector and returns the enhanced frame plus the next state. We do the STFT /
//! overlap-add ourselves, exactly like the reference Python `StreamEnhancer`.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;
use anyhow::{anyhow, bail};
use tract_onnx::prelude::*;

type Result<T> = anyhow::Result<T>;

pub const HOP: usize = 480;
/// End-to-end lag of [`Dpdfnet::process`] in hops. Measured by cross-correlating input and output
/// for both model sizes: 2400 samples = 5 hops (1 for the STFT overlap-add, 4 of model look-ahead).
/// Everything that must stay time-aligned with the denoised audio (the pre-leveler's inverse gain,
/// the whisper-rescue dry signal) is delayed by exactly this much; see `tests/denoise.rs`.
pub const LATENCY_HOPS: usize = 5;
const WIN: usize = 960;
const BINS: usize = WIN / 2 + 1;

/// Which size of the model to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    /// DPDFNet-2: light on CPU.
    Two,
    /// DPDFNet-8: a little cleaner, ~2x the CPU.
    Eight,
}

impl Size {
    /// File name of the ONNX model, as shipped in the release and by the installer.
    pub fn file_name(self) -> &'static str {
        match self {
            Size::Two => "dpdfnet2_48khz_hr.onnx",
            Size::Eight => "dpdfnet8_48khz_hr.onnx",
        }
    }

    /// True when this model can be loaded (embedded, or present in one of the model folders).
    pub fn is_installed(self) -> bool {
        cfg!(feature = "embed-models") || find_model(self).is_some()
    }
}

/// Folders searched for model files, in order: `SORDINO_MODELS_DIR`, `models` next to the exe,
/// `%LOCALAPPDATA%\Sordino\models` (where the installer puts them) and, for development builds,
/// the repository's `windows/models`.
fn model_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("SORDINO_MODELS_DIR") {
        dirs.push(d.into());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = exe.parent() {
            dirs.push(p.join("models"));
        }
    }
    if let Some(l) = std::env::var_os("LOCALAPPDATA") {
        dirs.push(std::path::Path::new(&l).join("Sordino").join("models"));
    }
    dirs.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models"));
    dirs
}

fn find_model(size: Size) -> Option<std::path::PathBuf> {
    model_dirs().into_iter().map(|d| d.join(size.file_name())).find(|p| p.is_file())
}

#[cfg(feature = "embed-models")]
fn model_bytes(size: Size) -> Result<std::borrow::Cow<'static, [u8]>> {
    Ok(std::borrow::Cow::Borrowed(match size {
        Size::Two => include_bytes!("../models/dpdfnet2_48khz_hr.onnx"),
        Size::Eight => include_bytes!("../models/dpdfnet8_48khz_hr.onnx"),
    }))
}

#[cfg(not(feature = "embed-models"))]
fn model_bytes(size: Size) -> Result<std::borrow::Cow<'static, [u8]>> {
    let path = find_model(size).ok_or_else(|| {
        anyhow!("model {} is not installed (run the installer again and select it)", size.file_name())
    })?;
    Ok(std::borrow::Cow::Owned(std::fs::read(&path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?))
}

type Plan = TypedRunnableModel;

pub struct Dpdfnet {
    run: TypedSimpleState,
    init_state: Vec<f32>,
    state: Option<Tensor>,
    in_tensor: Arc<Tensor>,
    window: [f32; WIN],
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    prev: [f32; HOP],
    ola: [f32; WIN],
    scratch_in: Vec<f32>,
    spec: Vec<Complex<f32>>,
    time: Vec<f32>,
    /// Idle skipping: consecutive hops quieter than `IDLE_RMS`, and the number of hops skipped so far.
    quiet_run: u32,
    pub skipped: u64,
    pub skip_idle: bool,
}

/// Hops this quiet (RMS below about -80 dBFS, true digital silence / muted mic; low enough that
/// soft whispers on quiet mics never put the model to sleep) ...
const IDLE_RMS: f32 = 1e-4;
/// ... for this many in a row (500 ms) put the model to sleep. Its recurrent state is frozen, not
/// reset: on near-silence it has settled to a fixed point anyway, so waking up continues seamlessly.
const IDLE_HOPS: u32 = 50;

/// Parses and optimizes the model. This is the expensive step (hundreds of ms of CPU), so it is
/// done once per model size and process: restarts (device changes, retries after boot) reuse it
/// instead of burning CPU each time, which starves other apps' audio threads.
fn cached_plan(size: Size) -> Result<(Arc<Plan>, Vec<f32>)> {
    static CACHE: std::sync::Mutex<Vec<(Size, Arc<Plan>, Vec<f32>)>> = std::sync::Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p, i)) = cache.iter().find(|(s, _, _)| *s == size) {
        return Ok((p.clone(), i.clone()));
    }
    let bytes = model_bytes(size)?;
    let proto = tract_onnx::onnx().proto_model_for_read(&mut &bytes[..])?;
    let meta = |key: &str| -> Result<String> {
        proto
            .metadata_props
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.value.clone())
            .ok_or_else(|| anyhow!("model metadata '{key}' missing"))
    };
    let floats = |s: String| -> Result<Vec<f32>> {
        s.split(',').map(|x| x.trim().parse::<f32>().map_err(|e| e.into())).collect()
    };
    let state_size: usize = meta("state_size")?.parse()?;
    let erb_n: usize = meta("erb_norm_state_size")?.parse()?;
    let spec_n: usize = meta("spec_norm_state_size")?.parse()?;
    let erb_init = floats(meta("erb_norm_init")?)?;
    let spec_init = floats(meta("spec_norm_init")?)?;
    if erb_init.len() != erb_n || spec_init.len() != spec_n || erb_n + spec_n > state_size {
        bail!("inconsistent DPDFNet state metadata");
    }
    let mut init_state = vec![0.0f32; state_size];
    init_state[..erb_n].copy_from_slice(&erb_init);
    init_state[erb_n..erb_n + spec_n].copy_from_slice(&spec_init);

    let mut typed = tract_onnx::onnx().model_for_proto_model(&proto)?.into_optimized()?;
    // The ONNX contract restarts every GRU from its initial state on each call; the model carries
    // its own recurrence in the explicit state tensor. Say so, so one tract session can be reused.
    for node in typed.nodes_mut() {
        if let Some(g) = node.op_as_mut::<tract_onnx::tract_core::ops::gru_seq::GruSeq>() {
            g.reset_every_turn = true;
        }
    }
    let plan: Arc<Plan> = typed.into_runnable()?;
    cache.push((size, plan.clone(), init_state.clone()));
    Ok((plan, init_state))
}

impl Dpdfnet {
    pub fn new(size: Size) -> Result<Self> {
        let (plan, init_state) = cached_plan(size)?;
        let run = plan.spawn()?;
        Self::with_run(run, init_state)
    }

    fn with_run(run: TypedSimpleState, init_state: Vec<f32>) -> Result<Self> {
        let mut window = [0.0f32; WIN];
        for (i, w) in window.iter_mut().enumerate() {
            let s = (0.5 * PI * (i as f32 + 0.5) / (WIN as f32 / 2.0)).sin();
            *w = (0.5 * PI * s * s).sin();
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(WIN);
        let inv = planner.plan_fft_inverse(WIN);
        let state = Some(tensor1(&init_state));
        let in_tensor = Arc::new(Tensor::zero::<f32>(&[1, 1, BINS, 2])?);
        Ok(Self {
            run,
            init_state,
            state,
            in_tensor,
            window,
            fwd,
            inv,
            prev: [0.0; HOP],
            ola: [0.0; WIN],
            scratch_in: vec![0.0; WIN],
            spec: vec![Complex::new(0.0, 0.0); BINS],
            time: vec![0.0; WIN],
            quiet_run: 0,
            skipped: 0,
            skip_idle: true,
        })
    }

    /// Back to a clean stream (used when the denoiser is re-enabled after a long gap).
    pub fn reset(&mut self) {
        self.state = Some(tensor1(&self.init_state));
        self.prev = [0.0; HOP];
        self.ola = [0.0; WIN];
        self.quiet_run = 0;
    }

    /// Enhances one 10 ms hop. The output lags the input by [`LATENCY_HOPS`] hops (50 ms): one for
    /// the STFT overlap-add plus the model's own look-ahead.
    pub fn process(&mut self, input: &[f32; HOP], output: &mut [f32; HOP]) -> Result<()> {
        if self.skip_idle {
            let rms = (input.iter().map(|s| s * s).sum::<f32>() / HOP as f32).sqrt();
            self.quiet_run = if rms < IDLE_RMS { self.quiet_run.saturating_add(1) } else { 0 };
            if self.quiet_run > IDLE_HOPS {
                // Asleep: play out the tail of the last frame, then silence. Keep `prev` current so the
                // first frame after waking has the right left half.
                output.copy_from_slice(&self.ola[..HOP]);
                self.ola.fill(0.0);
                self.prev = *input;
                self.skipped += 1;
                return Ok(());
            }
        }
        self.scratch_in[..HOP].copy_from_slice(&self.prev);
        self.scratch_in[HOP..].copy_from_slice(input);
        self.prev = *input;
        for i in 0..WIN {
            self.scratch_in[i] *= self.window[i];
        }
        self.fwd.process(&mut self.scratch_in, &mut self.spec).map_err(|e| anyhow!("fft: {e}"))?;
        {
            let t = Arc::make_mut(&mut self.in_tensor);
            let mut view = t.to_plain_array_view_mut::<f32>()?;
            let dst = view.as_slice_mut().ok_or_else(|| anyhow!("non-contiguous input tensor"))?;
            for (i, c) in self.spec.iter().enumerate() {
                dst[i * 2] = c.re;
                dst[i * 2 + 1] = c.im;
            }
        }
        let state = self.state.take().unwrap_or_else(|| tensor1(&self.init_state));
        let mut outs = self.run.run(tvec!(self.in_tensor.clone().into_tvalue(), state.into_tvalue()))?;
        let next_state = outs.pop().ok_or_else(|| anyhow!("no state output"))?;
        let spec_e = outs.pop().ok_or_else(|| anyhow!("no spec output"))?;
        self.state = Some(next_state.into_tensor());
        let spec_e = spec_e.into_tensor();
        let view = spec_e.to_plain_array_view::<f32>()?;
        if view.len() != BINS * 2 {
            bail!("unexpected DPDFNet output size {}", view.len());
        }
        let mut it = view.iter();
        for c in self.spec.iter_mut() {
            let (re, im) = (*it.next().unwrap(), *it.next().unwrap());
            *c = Complex::new(re, im);
        }
        // The inverse FFT needs real DC/Nyquist bins.
        self.spec[0].im = 0.0;
        self.spec[BINS - 1].im = 0.0;
        self.inv.process(&mut self.spec, &mut self.time).map_err(|e| anyhow!("ifft: {e}"))?;
        let norm = 1.0 / WIN as f32;
        for i in 0..WIN {
            self.ola[i] += self.time[i] * norm * self.window[i];
        }
        output.copy_from_slice(&self.ola[..HOP]);
        self.ola.copy_within(HOP.., 0);
        self.ola[WIN - HOP..].fill(0.0);
        if output.iter().any(|s| !s.is_finite()) {
            output.fill(0.0);
            self.reset();
        }
        Ok(())
    }
}

fn tensor1(v: &[f32]) -> Tensor {
    tract_ndarray::Array1::from_vec(v.to_vec()).into_tensor()
}
