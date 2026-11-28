#![cfg_attr(not(test), windows_subsystem = "windows")]

use sordino::pipeline::{Config, Pipeline};
use sordino::{app, devices, dsp, settings, wavio};
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Duration;

const USAGE: &str = "Sordino for Windows

Usage:
  sordino.exe                         run in the system tray
  sordino.exe --list-devices          list microphones and virtual mics
  sordino.exe --restore-default       make a real microphone the default again
  sordino.exe --selftest [options]    run the pipeline and print levels
        --mic TEXT      use the microphone whose name contains TEXT
        --out TEXT      use the virtual mic whose name contains TEXT
        --file WAV      feed a 48 kHz WAV instead of the microphone
        --seconds N     duration (default 8)
        --no-denoise --no-gate --no-agc --engine rnnoise|dpdfnet2|dpdfnet8
  sordino.exe --process-file IN.wav OUT.wav [--no-denoise] [--no-gate] [--agc] [--hp HZ] [--engine rnnoise|dpdfnet2|dpdfnet8]
";

fn parse_engine(args: &[String]) -> anyhow::Result<dsp::Engine> {
    match arg_value(args, "--engine") {
        None => Ok(dsp::Engine::Dpdfnet8),
        Some(n) => dsp::Engine::from_name(n).ok_or_else(|| anyhow::anyhow!("unknown engine '{n}' (rnnoise, dpdfnet2, dpdfnet8)")),
    }
}

/// GUI-subsystem exes have no console; attach to the parent shell's so CLI output is visible.
fn attach_console() {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Console::{AttachConsole, GetStdHandle, SetStdHandle, ATTACH_PARENT_PROCESS, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
    unsafe {
        // Already redirected to a file or pipe by the caller: keep it.
        if matches!(GetStdHandle(STD_OUTPUT_HANDLE), Ok(h) if !h.is_invalid() && !h.0.is_null()) {
            return;
        }
        if AttachConsole(ATTACH_PARENT_PROCESS).is_ok() {
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open("CONOUT$") {
                let h = HANDLE(f.as_raw_handle());
                let _ = SetStdHandle(STD_OUTPUT_HANDLE, h);
                let _ = SetStdHandle(STD_ERROR_HANDLE, h);
                std::mem::forget(f);
            }
        }
    }
}

/// Returns false when another Sordino is already running.
fn single_instance() -> bool {
    use windows::core::w;
    use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};
    use windows::Win32::System::Threading::CreateMutexW;
    unsafe {
        let h = CreateMutexW(None, true, w!("Local\\Sordino.SingleInstance"));
        let dup = h.is_err() || GetLastError() == ERROR_ALREADY_EXISTS;
        std::mem::forget(h);
        !dup
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let has = |f: &str| args.iter().any(|a| a == f);

    if args.len() > 1 {
        attach_console();
    }
    if has("--help") || has("-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if has("--process-file") {
        let i = args.iter().position(|a| a == "--process-file").unwrap();
        let (Some(inp), Some(out)) = (args.get(i + 1), args.get(i + 2)) else {
            eprintln!("{USAGE}");
            std::process::exit(2);
        };
        let input = wavio::read_mono_48k(Path::new(inp))?;
        // `--hp 0` turns the high-pass off; anything else must be a usable cutoff frequency.
        let hp: f32 = match arg_value(&args, "--hp") {
            None => 90.0,
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--hp expects a frequency in Hz, got \"{v}\""))?,
        };
        if hp != 0.0 && !(dsp::MIN_HP_HZ..=dsp::MAX_HP_HZ).contains(&hp) {
            anyhow::bail!("--hp must be 0 (off) or between {} and {} Hz, got {hp}", dsp::MIN_HP_HZ, dsp::MAX_HP_HZ);
        }
        let params = dsp::Params::new(!has("--no-denoise"), hp > 0.0, true, has("--agc"));
        params.gate.store(!has("--no-gate"), Relaxed);
        let engine = parse_engine(&args)?;
        wavio::write_mono_48k(Path::new(out), &dsp::process_buffer_engine(params, &input, hp, engine))?;
        return Ok(());
    }
    let _ = wasapi::initialize_mta();
    if has("--list-devices") {
        println!("Microphones:");
        let default = devices::default_capture_id();
        for d in devices::list(devices::DirectionKind::Capture) {
            let star = if default.as_deref() == Some(d.id.as_str()) { "  [system default]" } else { "" };
            println!("  {}{}{}\n      {}", d.name, if d.is_virtual { "  [virtual]" } else { "" }, star, d.id);
        }
        println!("Playback:");
        for d in devices::list(devices::DirectionKind::Render) {
            println!("  {}{}\n      {}", d.name, if d.is_virtual { "  [virtual - usable as Sordino output]" } else { "" }, d.id);
        }
        return Ok(());
    }
    if has("--restore-default") {
        return restore_default();
    }
    if has("--selftest") {
        return selftest(&args);
    }
    if !single_instance() {
        return Ok(());
    }
    app::run()
}

/// Rescue command: puts a real microphone back as the Windows default recording device.
fn restore_default() -> anyhow::Result<()> {
    let st = settings::Settings::load();
    let mics = devices::microphones();
    // Check every role: Discord/Teams follow Communications, which can be stuck on the dead
    // virtual mic even when Console already points at a real one.
    let stuck = devices::capture_roles_where(|d| d.is_virtual);
    if stuck.is_empty() {
        println!("Default recording device is already a real microphone.");
        return Ok(());
    }
    let target = mics.iter().find(|d| d.id == st.previous_default_input_id).or_else(|| mics.first());
    let Some(t) = target else { anyhow::bail!("No microphone found") };
    sordino::defaultmic::set_default_capture_roles(&t.id, &stuck)?;
    println!("Default recording device set to: {}", t.name);
    Ok(())
}

fn selftest(args: &[String]) -> anyhow::Result<()> {
    let st = settings::Settings::load();
    let mics = devices::microphones();
    let outs = devices::virtual_outputs();
    let pick = |list: &[devices::AudioDevice], needle: Option<&str>, saved: &str| {
        list.iter()
            .find(|d| needle.map(|n| d.name.to_lowercase().contains(&n.to_lowercase())).unwrap_or(d.id == saved))
            .or_else(|| list.first())
            .cloned()
    };
    let mic = pick(&mics, arg_value(args, "--mic"), &st.input_id);
    let out = pick(&outs, arg_value(args, "--out"), &st.output_id);
    println!("mic    : {}", mic.as_ref().map(|d| d.name.as_str()).unwrap_or("(none)"));
    println!("virtual: {}", out.as_ref().map(|d| d.name.as_str()).unwrap_or("(none - install VB-CABLE)"));
    let Some(out) = out else { anyhow::bail!("No virtual mic found. Install VB-CABLE from https://vb-audio.com/Cable/") };

    let has = |f: &str| args.iter().any(|a| a == f);
    let params = dsp::Params::new(!has("--no-denoise"), true, true, !has("--no-agc"));
    params.gate.store(!has("--no-gate"), Relaxed);
    let source = match arg_value(args, "--file") {
        Some(p) => Some(Arc::new(wavio::read_mono_48k(Path::new(p))?)),
        None => None,
    };
    let secs: u64 = arg_value(args, "--seconds").and_then(|v| v.parse().ok()).unwrap_or(8);
    let pipe = Pipeline::start(Config {
        input_id: mic.map(|d| d.id),
        output_id: out.id,
        params,
        engine: parse_engine(args)?,
        source_samples: source,
    })
    .map_err(|e| anyhow::anyhow!(e))?;
    for t in 1..=secs {
        std::thread::sleep(Duration::from_secs(1));
        let (under, over) = pipe.ring_stats();
        println!("t={t} peak={:.4} under={under} over={over} error={}", pipe.take_peak(), pipe.error().unwrap_or_else(|| "-".into()));
    }
    pipe.stop();
    Ok(())
}
