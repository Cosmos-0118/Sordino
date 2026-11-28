//! The tray application: menu, settings, device supervision and default-mic management.

use crate::devices::{self, AudioDevice};
use crate::dsp::{self, Params};
use crate::pipeline::{Config, Pipeline};
use crate::{defaultmic, icon, settings};
use settings::Settings;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{TrayIcon, TrayIconBuilder};
use wasapi::DeviceEventCallbacks;

const CABLE_URL: &str = "https://vb-audio.com/Cable/";

#[derive(Debug)]
enum UserEvent {
    Menu(String),
    Tick,
    DevicesChanged,
    DefaultChanged,
}

struct App {
    st: Settings,
    params: Arc<Params>,
    pipe: Option<Pipeline>,
    mics: Vec<AudioDevice>,
    outs: Vec<AudioDevice>,
    tray: TrayIcon,
    status: Option<String>,
    level: f32,
    shown: (bool, bool, u8),
    login: bool,
    restart_at: Option<Instant>,
    fails: u32,
    own_default_change: Instant,
    menu_dirty: bool,
    started: Instant,
    /// Virtual mic should become the default once the pipeline is actually delivering audio.
    default_pending: bool,
    /// A leftover virtual-mic default from a killed/shut-down run still has to be undone.
    stale_pending: bool,
    last_stale_check: Instant,
}

/// After launch, devices (USB mics, the VB-CABLE driver) may not be enumerated yet; during this
/// window a missing saved device is waited for instead of being replaced and overwritten.
const BOOT_GRACE: Duration = Duration::from_secs(60);

pub fn run() -> anyhow::Result<()> {
    let _ = wasapi::initialize_mta();
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    {
        let p = Mutex::new(proxy.clone());
        MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
            let _ = p.lock().unwrap().send_event(UserEvent::Menu(e.id.0));
        }));
    }
    spawn_ticker(proxy.clone());
    let _notify = register_device_notifications(proxy.clone());

    let st = Settings::load();
    let params = Params::new(st.denoise, st.highpass, st.limiter, st.agc);
    params.gate.store(st.gate, Relaxed);
    let tray = TrayIconBuilder::new()
        .with_icon(icon::render(false, false, 0.0))
        .with_tooltip("Sordino")
        .build()?;
    let mut app = App {
        st,
        params,
        pipe: None,
        mics: vec![],
        outs: vec![],
        tray,
        status: None,
        level: 0.0,
        shown: (false, false, 255),
        login: settings::launch_at_login(),
        restart_at: None,
        fails: 0,
        own_default_change: Instant::now() - Duration::from_secs(60),
        menu_dirty: true,
        started: Instant::now(),
        default_pending: false,
        stale_pending: true,
        last_stale_check: Instant::now() - Duration::from_secs(60),
    };

    event_loop.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        match event {
            Event::NewEvents(StartCause::Init) => {
                app.refresh_devices();
                app.stale_pending = !app.cleanup_stale_default();
                if app.st.enabled {
                    app.start();
                }
                app.sync_ui();
            }
            Event::UserEvent(UserEvent::Menu(id)) => {
                if app.handle_menu(&id) {
                    app.shutdown();
                    *flow = ControlFlow::Exit;
                    return;
                }
                app.sync_ui();
            }
            Event::UserEvent(UserEvent::Tick) => {
                app.tick();
                app.sync_ui();
            }
            Event::UserEvent(UserEvent::DevicesChanged) => {
                app.refresh_devices();
                // Boot / hot-plug produces bursts of events: debounce into one (re)start attempt
                // instead of opening and closing streams for each, which makes games lose audio.
                if app.st.enabled && app.pipe.is_none() {
                    app.restart_at = Some(Instant::now() + Duration::from_millis(1500));
                }
                app.sync_ui();
            }
            Event::UserEvent(UserEvent::DefaultChanged) => {
                app.default_changed();
            }
            _ => {}
        }
    })
}

fn spawn_ticker(proxy: EventLoopProxy<UserEvent>) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(100));
        if proxy.send_event(UserEvent::Tick).is_err() {
            break;
        }
    });
}

/// Keeps the registration alive for the whole run; callbacks run on a Windows audio thread.
fn register_device_notifications(proxy: EventLoopProxy<UserEvent>) -> Option<wasapi::DeviceEventRegistration> {
    let en = wasapi::DeviceEnumerator::new().ok()?;
    let mut cb = DeviceEventCallbacks::new();
    let p = Arc::new(Mutex::new(proxy));
    let send = move |ev: fn() -> UserEvent| {
        let p = p.clone();
        move || {
            let _ = p.lock().unwrap().send_event(ev());
        }
    };
    let (a, b, c, d) = (send(|| UserEvent::DevicesChanged), send(|| UserEvent::DevicesChanged), send(|| UserEvent::DevicesChanged), send(|| UserEvent::DefaultChanged));
    cb.set_device_added_callback(move |_| a());
    cb.set_device_removed_callback(move |_| b());
    cb.set_device_state_callback(move |_, _| c());
    cb.set_default_device_callback(move |_, _, _| d());
    en.register_notification_callback(cb).ok()
}

impl App {
    fn save(&self) {
        self.st.save();
    }

    // ---------------------------------------------------------------- devices

    fn refresh_devices(&mut self) {
        let mics = devices::microphones();
        let outs = devices::virtual_outputs();
        if mics != self.mics || outs != self.outs {
            self.mics = mics;
            self.outs = outs;
            self.menu_dirty = true;
        }
        let mut changed = false;
        let grace = self.started.elapsed() < BOOT_GRACE;
        // Never replace (and persist) a saved device because the list is empty or still filling
        // up right after boot: that silently loses the user's choice.
        let settle = |list_empty: bool, saved: &str| list_empty || (grace && !saved.is_empty());
        if !self.mics.iter().any(|d| d.id == self.st.input_id) && !settle(self.mics.is_empty(), &self.st.input_id) {
            // Prefer what the user was using before Sordino moved the default, then the current
            // default if it is physical, then the first microphone.
            let prev = &self.st.previous_default_input_id;
            let cur = devices::default_capture_id().unwrap_or_default();
            let pick = [prev.as_str(), cur.as_str()]
                .iter()
                .find_map(|id| self.mics.iter().find(|d| d.id == *id))
                .or_else(|| self.mics.first())
                .map(|d| d.id.clone())
                .unwrap_or_default();
            if pick != self.st.input_id {
                self.st.input_id = pick;
                changed = true;
            }
        }
        if !self.outs.iter().any(|d| d.id == self.st.output_id) && !settle(self.outs.is_empty(), &self.st.output_id) {
            let pick = self
                .outs
                .iter()
                .find(|d| d.name.contains("CABLE Input"))
                .or_else(|| self.outs.first())
                .map(|d| d.id.clone())
                .unwrap_or_default();
            if pick != self.st.output_id {
                self.st.output_id = pick;
                changed = true;
            }
        }
        if changed {
            self.save();
            self.menu_dirty = true;
        }
    }

    fn selected_output(&self) -> Option<&AudioDevice> {
        self.outs.iter().find(|d| d.id == self.st.output_id)
    }

    // ---------------------------------------------------------------- pipeline

    fn start(&mut self) {
        self.stop_pipeline();
        self.restart_at = None;
        let cfg = Config {
            input_id: if self.st.input_id.is_empty() { None } else { Some(self.st.input_id.clone()) },
            output_id: self.st.output_id.clone(),
            params: self.params.clone(),
            engine: dsp::Engine::from_name(&self.st.engine).unwrap_or(dsp::Engine::Dpdfnet8).or_best_available(),
            source_samples: None,
        };
        match Pipeline::start(cfg) {
            Ok(p) => {
                self.pipe = Some(p);
                self.status = None;
                self.fails = 0;
                // Switch the Windows default only once audio really flows (see `tick`).
                self.default_pending = true;
            }
            Err(e) => {
                self.status = Some(e);
                // Retry with backoff: device busy, Bluetooth mic reconnecting, or (right after boot)
                // the audio service / VB-CABLE driver not ready yet.
                self.schedule_restart();
                // Don't leave games on a dead virtual mic while we keep failing.
                if self.fails >= 2 {
                    self.restore_default();
                }
            }
        }
        self.menu_dirty = true;
    }

    fn schedule_restart(&mut self) {
        self.fails += 1;
        let secs = (1u64 << self.fails.min(4)).min(15);
        self.restart_at = Some(Instant::now() + Duration::from_secs(secs));
    }

    fn stop_pipeline(&mut self) {
        if let Some(p) = self.pipe.take() {
            p.stop();
        }
        self.level = 0.0;
    }

    fn stop(&mut self) {
        self.restart_at = None;
        self.status = None;
        self.stop_pipeline();
        self.restore_default();
        self.menu_dirty = true;
    }

    fn shutdown(&mut self) {
        self.stop_pipeline();
        self.restore_default();
    }

    fn restart_if_running(&mut self) {
        if self.st.enabled {
            self.start();
        }
    }

    fn tick(&mut self) {
        let mut apply = false;
        if let Some(p) = &self.pipe {
            let peak = p.take_peak();
            self.level = peak.max(self.level * 0.85);
            if let Some(err) = p.error() {
                // Device vanished / stream failed: stop, then rebuild after a short pause.
                self.stop_pipeline();
                self.status = Some(err);
                self.schedule_restart();
                self.menu_dirty = true;
            } else if self.default_pending && p.is_flowing() {
                apply = true;
            }
        } else {
            self.level = 0.0;
        }
        if apply {
            self.default_pending = false;
            self.apply_default();
        }
        if self.stale_pending && self.last_stale_check.elapsed() >= Duration::from_secs(2) {
            self.last_stale_check = Instant::now();
            self.refresh_devices();
            self.stale_pending = !self.cleanup_stale_default() && self.started.elapsed() < BOOT_GRACE;
        }
        if let Some(at) = self.restart_at {
            if Instant::now() >= at && self.st.enabled {
                self.refresh_devices();
                self.start();
            }
        }
    }

    // ---------------------------------------------------------------- default mic

    fn cable_capture(&self) -> Option<AudioDevice> {
        devices::virtual_capture_for(self.selected_output()?)
    }

    /// Makes the virtual mic the Windows default recording device so apps pick it automatically.
    fn apply_default(&mut self) {
        if !self.st.use_as_default || !self.pipe.as_ref().is_some_and(|p| p.is_flowing()) {
            return;
        }
        let Some(cable) = self.cable_capture() else { return };
        // Take over every role that isn't already on the virtual mic (Windows can move just one).
        let not_cable = devices::capture_roles_where(|d| d.id != cable.id);
        if not_cable.is_empty() {
            return;
        }
        // Remember the real mic to give back later, from whichever role still points at one.
        let prev = not_cable
            .iter()
            .filter_map(|&r| devices::default_capture_id_for_role(r))
            .find(|id| self.mics.iter().any(|d| d.id == *id));
        if let Some(prev) = prev {
            self.st.previous_default_input_id = prev;
            self.save();
        }
        self.own_default_change = Instant::now();
        let _ = defaultmic::set_default_capture_roles(&cable.id, &not_cable);
    }

    /// Puts the user's real default mic back (only if the virtual mic is still the default).
    fn restore_default(&mut self) {
        let Some(cable) = self.cable_capture() else { return };
        let stuck = devices::capture_roles_where(|d| d.id == cable.id);
        if stuck.is_empty() {
            return;
        }
        self.set_real_default(&stuck);
    }

    /// Sets the user's previous mic (or the first real one) as the Windows default for `roles`.
    fn set_real_default(&mut self, roles: &[i32]) -> bool {
        let prev = &self.st.previous_default_input_id;
        let Some(d) = self.mics.iter().find(|d| d.id == *prev).or_else(|| self.mics.first()) else { return false };
        self.own_default_change = Instant::now();
        defaultmic::set_default_capture_roles(&d.id, roles).is_ok()
    }

    /// If a previous run was killed or Windows shut down while the virtual mic was the default,
    /// no pipeline feeds it after the next boot: every app (games!) would get a dead microphone
    /// until we are up. Undo that. Returns true when nothing is left to do.
    fn cleanup_stale_default(&mut self) -> bool {
        if self.pipe.is_some() {
            return true;
        }
        // Any role left on a virtual mic counts, not just Console (Discord follows Communications).
        let stuck = devices::capture_roles_where(|d| d.is_virtual);
        if stuck.is_empty() {
            return true;
        }
        self.set_real_default(&stuck)
    }

    /// Windows switches the default recording device by itself when a headset connects.
    fn default_changed(&mut self) {
        if self.pipe.is_some() && self.st.use_as_default && self.own_default_change.elapsed() > Duration::from_millis(1500) {
            self.apply_default();
        }
    }

    // ---------------------------------------------------------------- menu

    /// Returns true when the app should quit.
    fn handle_menu(&mut self, id: &str) -> bool {
        match id {
            "quit" => return true,
            "toggle" => {
                self.st.enabled = !self.st.enabled;
                self.save();
                if self.st.enabled {
                    self.refresh_devices();
                    self.start();
                } else {
                    self.stop();
                }
            }
            "opt:denoise" => {
                self.st.denoise = !self.st.denoise;
                self.params.denoise.store(self.st.denoise, Relaxed);
                self.save();
            }
            "opt:gate" => {
                self.st.gate = !self.st.gate;
                self.params.gate.store(self.st.gate, Relaxed);
                self.save();
            }
            id if id.starts_with("engine:") => {
                // Re-selecting the active engine must not tear the pipeline down.
                if self.st.engine != id["engine:".len()..] {
                    self.st.engine = id["engine:".len()..].to_string();
                    self.save();
                    self.restart_if_running();
                }
            }
            "opt:highpass" => {
                self.st.highpass = !self.st.highpass;
                self.params.highpass.store(self.st.highpass, Relaxed);
                self.save();
            }
            "opt:limiter" => {
                self.st.limiter = !self.st.limiter;
                self.params.limiter.store(self.st.limiter, Relaxed);
                self.save();
            }
            "opt:agc" => {
                self.st.agc = !self.st.agc;
                self.params.agc.store(self.st.agc, Relaxed);
                self.save();
            }
            "opt:default" => {
                self.st.use_as_default = !self.st.use_as_default;
                self.save();
                if self.st.use_as_default {
                    // During warm-up the pipeline isn't flowing yet; `tick` applies it once it is.
                    self.default_pending = self.pipe.is_some();
                    self.apply_default();
                } else {
                    self.default_pending = false;
                    self.restore_default();
                }
            }
            "opt:login" => {
                let want = !self.login;
                match settings::set_launch_at_login(want) {
                    Ok(()) => self.login = want,
                    Err(e) => self.status = Some(format!("Launch at login failed: {e}")),
                }
            }
            "install-cable" => {
                let _ = std::process::Command::new("explorer").arg(CABLE_URL).spawn();
            }
            _ => {
                if let Some(dev) = id.strip_prefix("mic:") {
                    if dev != self.st.input_id {
                        self.st.input_id = dev.to_string();
                        self.save();
                        self.restart_if_running();
                    }
                } else if let Some(dev) = id.strip_prefix("out:") {
                    if dev != self.st.output_id {
                        self.restore_default(); // while the old virtual mic is still known
                        self.st.output_id = dev.to_string();
                        self.save();
                        self.restart_if_running();
                    }
                }
            }
        }
        self.menu_dirty = true;
        false
    }

    fn build_menu(&self) -> Menu {
        let menu = Menu::new();
        let running = self.pipe.is_some();
        let _ = menu.append(&CheckMenuItem::with_id("toggle", "Sordino On", true, self.st.enabled, None));
        if let Some(s) = &self.status {
            let _ = menu.append(&MenuItem::with_id("status", truncate(s, 90), false, None));
        } else if running {
            let _ = menu.append(&MenuItem::with_id("status", "Cleaning your voice", false, None));
        }
        let _ = menu.append(&PredefinedMenuItem::separator());

        let mic = Submenu::new("Microphone", true);
        if self.mics.is_empty() {
            let _ = mic.append(&MenuItem::with_id("none-mic", "No microphone found", false, None));
        }
        for d in &self.mics {
            let _ = mic.append(&CheckMenuItem::with_id(format!("mic:{}", d.id), &d.name, true, d.id == self.st.input_id, None));
        }
        let _ = menu.append(&mic);

        let out = Submenu::new("Virtual Mic", true);
        if self.outs.is_empty() {
            let _ = out.append(&MenuItem::with_id("none-out", "VB-CABLE isn't installed", false, None));
            let _ = out.append(&MenuItem::with_id("install-cable", "Get VB-CABLE…", true, None));
        }
        for d in &self.outs {
            let _ = out.append(&CheckMenuItem::with_id(format!("out:{}", d.id), &d.name, true, d.id == self.st.output_id, None));
        }
        let _ = menu.append(&out);
        let _ = menu.append(&PredefinedMenuItem::separator());

        let opt = |id: &str, text: &str, on: bool| CheckMenuItem::with_id(id, text, true, on, None);
        let _ = menu.append(&opt("opt:denoise", "Noise Suppression", self.st.denoise));
        let eng = Submenu::new("Voice Engine", true);
        let cur = dsp::Engine::from_name(&self.st.engine).unwrap_or(dsp::Engine::Dpdfnet8).or_best_available();
        // Only models that are installed are listed; the installer adds more (re-run it to change them).
        for (id, label, e) in [
            ("dpdfnet8", "Best quality (DPDFNet-8)", dsp::Engine::Dpdfnet8),
            ("dpdfnet2", "Lighter on CPU (DPDFNet-2)", dsp::Engine::Dpdfnet2),
            ("rnnoise", "Minimal (RNNoise, weakest)", dsp::Engine::Rnnoise),
        ] {
            if e.is_available() {
                let _ = eng.append(&CheckMenuItem::with_id(format!("engine:{id}"), label, true, cur == e, None));
            }
        }
        let _ = menu.append(&eng);
        let _ = menu.append(&opt("opt:highpass", "Reduce Air Thumps", self.st.highpass));
        let _ = menu.append(&opt("opt:gate", "Silence Gate", self.st.gate));
        let _ = menu.append(&opt("opt:limiter", "Peak Limiter", self.st.limiter));
        let _ = menu.append(&opt("opt:agc", "Auto Gain", self.st.agc));
        let _ = menu.append(&opt("opt:default", "Use as Default Mic", self.st.use_as_default));
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&opt("opt:login", "Open at Login", self.login));
        let _ = menu.append(&MenuItem::with_id("quit", "Quit Sordino", true, None));
        menu
    }

    fn sync_ui(&mut self) {
        if self.menu_dirty {
            self.menu_dirty = false;
            self.tray.set_menu(Some(Box::new(self.build_menu())));
        }
        let on = self.pipe.is_some();
        let err = self.status.is_some() && self.st.enabled;
        let lvl = (self.level.sqrt() * 6.0).round().min(6.0) as u8;
        let want = (on, err, if on { lvl } else { 0 });
        if want != self.shown {
            self.shown = want;
            let _ = self.tray.set_icon(Some(icon::render(on, err, want.2 as f32 / 6.0)));
            let tip = match (&self.status, on) {
                (Some(s), _) if self.st.enabled => format!("Sordino: {}", truncate(s, 100)),
                (_, true) => "Sordino: on".to_string(),
                _ => "Sordino: off".to_string(),
            };
            let _ = self.tray.set_tooltip(Some(tip));
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}
