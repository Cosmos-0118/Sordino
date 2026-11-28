import AVFoundation
import AppKit
import AudioToolbox
import CoreAudio
import ServiceManagement
import os

/// Drives only the level meter, so 20 Hz updates never re-render the rest of the menu.
final class LevelModel: ObservableObject {
    @Published var level: Float = 0
}

enum Mode: String, CaseIterable, Identifiable {
    case voiceIsolation = "Voice Isolation"
    case voiceProcessing = "Voice Processing"
    var id: String { rawValue }
}

private let log = Logger(subsystem: "com.sordino.mac", category: "pipeline")

/// Mic -> [AVCaptureSession (Voice Isolation) | Apple voice processing] -> ring -> high-pass + limiter
/// -> AUHAL bound to the virtual mic (BlackHole).
///
/// Robustness model (desired state + watchdog):
/// - The main thread only states what it WANTS: a `Config` (which mic, which virtual mic, which mode) or
///   nil for off. It never starts or stops audio itself.
/// - The serial queue `q` owns all audio objects. `reconcile()` runs after every relevant event and once
///   a second. It rebuilds only when the wanted Config changed or the running pipeline is actually
///   broken (a device vanished, capture stopped delivering, the output stopped rendering).
/// - Unrelated churn is ignored: other apps' voice-processing aggregates appearing, sample-rate changes
///   on BlackHole (the AUHAL converts), configuration notifications that didn't stop anything. This is
///   what keeps a WhatsApp call starting during a Discord call from setting off a rebuild storm that
///   disrupts both apps.
/// - Repeated failures back off (1, 2, 4, 8, 15 s) and Sordino stays on, retrying, instead of switching
///   itself off.
final class Pipeline: ObservableObject {
    @Published var isRunning = false
    @Published var status: String?
    @Published var devices: [AudioDevice] = []
    @Published var micMode = "Standard"
    let meter = LevelModel()

    /// The user's choices. If a chosen device is unplugged, a fallback is used until it comes back.
    @Published var inputUID: String { didSet { if inputUID != oldValue { save(); pushConfig() } } }
    @Published var outputUID: String { didSet { if outputUID != oldValue { save(); pushConfig() } } }
    @Published var mode: Mode { didSet { if mode != oldValue { save(); pushConfig() } } }
    @Published var highPass: Bool { didSet { save(); dsp.highPass = highPass } }
    @Published var limiter: Bool { didSet { save(); dsp.limiter = limiter } }
    @Published var agc: Bool {
        didSet {
            save()
            let v = agc
            q.async { self.agcOn = v; self.engine?.inputNode.isVoiceProcessingAGCEnabled = v }
        }
    }
    @Published var useAsDefault: Bool {
        didSet {
            save()
            let on = useAsDefault
            q.async {
                self.useDefaultOn = on
                guard self.built?.mode == .voiceIsolation else { return }
                on ? self.applyDefaultInput() : self.restoreDefaultInput()
            }
        }
    }
    @Published var launchAtLogin: Bool {
        didSet {
            do {
                if launchAtLogin { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            } catch { status = "Launch at login failed: \(error.localizedDescription)" }
        }
    }

    // MARK: audio state (owned by `q`)

    private let q = DispatchQueue(label: "sordino.audio", qos: .userInteractive)
    private let dsp = VoiceDSP()
    private var engine: AVAudioEngine?          // Voice Processing capture
    private var mic: MicCapture?                // Voice Isolation capture
    private var output: VirtualMicOutput?
    private var ring: RingBuffer?
    private var observers: [NSObjectProtocol] = []
    private var timer: DispatchSourceTimer?

    private var wanted: Config?                 // what the UI asked for; nil = off
    private var attempted: Config?              // config of the last build attempt
    private var built: Config?                  // config that is running right now
    private var attemptAt: Double = 0
    private var builtAt: Double = 0
    private var failures = 0
    private var lastError: String?
    private var shownStatus: String?
    private var lastDefaultApply: Double = 0
    private var agcOn = true
    private var useDefaultOn = true

    private var rawLevel: Float = 0             // written by audio threads, read by the UI timer
    private var tick = 0
    private var pendingPush: DispatchWorkItem?
    private let d = UserDefaults.standard

    var inputs: [AudioDevice] { devices.filter { $0.hasInput && !$0.isVirtualLoopback && !$0.isInternalAggregate } }
    var outputs: [AudioDevice] { devices.filter { $0.hasOutput && !$0.isInternalAggregate } }
    var blackHole: AudioDevice? { devices.first { $0.name.lowercased().contains("blackhole") && $0.hasOutput } }

    /// The mic actually used: the user's choice, else the built-in mic, else anything.
    var activeInput: AudioDevice? {
        inputs.first { $0.uid == inputUID } ?? inputs.first { $0.name.contains("MacBook") } ?? inputs.first
    }
    /// The virtual mic actually used: the user's choice, else BlackHole. Only ever a virtual mic: a real
    /// output would play the user's voice out of the speakers and straight back into the mic.
    var activeOutput: AudioDevice? { outputs.first { $0.uid == outputUID && $0.isVirtualLoopback } ?? blackHole }

    init() {
        inputUID = d.string(forKey: "inputUID") ?? ""
        outputUID = d.string(forKey: "outputUID") ?? ""
        mode = Mode(rawValue: d.string(forKey: "mode") ?? "") ?? .voiceIsolation
        highPass = d.object(forKey: "highPass") as? Bool ?? true
        limiter = d.object(forKey: "limiter") as? Bool ?? true
        agc = d.object(forKey: "agc") as? Bool ?? true
        useAsDefault = d.object(forKey: "useAsDefault") as? Bool ?? true
        launchAtLogin = SMAppService.mainApp.status == .enabled
        dsp.highPass = highPass
        dsp.limiter = limiter
        agcOn = agc
        useDefaultOn = useAsDefault

        refreshDevices()
        AudioDevices.observeChanges { [weak self] in self?.refreshDevices() }
        AudioDevices.observeDefaultInput { [weak self] in self?.defaultInputChanged() }

        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            self.q.sync {
                self.wanted = nil
                self.teardown()
                self.restoreDefaultInput()
            }
        }
        // After sleep, devices and capture sessions are often dead: check right away instead of
        // waiting for the stall detector.
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self?.refreshDevices(); self?.pushConfig() }
        }
        // .common so the meter keeps updating while the menu is open (menu tracking runs a non-default mode).
        let uiTimer = Timer(timeInterval: 1.0 / 20, repeats: true) { [weak self] _ in self?.uiTick() }
        RunLoop.main.add(uiTimer, forMode: .common)

        let t = DispatchSource.makeTimerSource(queue: q)
        t.schedule(deadline: .now() + 1, repeating: 1, leeway: .milliseconds(100))
        t.setEventHandler { [weak self] in self?.reconcile() }
        t.resume()
        timer = t

        if CommandLine.arguments.contains("--selftest") { selfTest() }
        else if d.object(forKey: "enabled") as? Bool ?? true { start() }
    }

    private func uiTick() {
        tick += 1
        let l = isRunning ? rawLevel : 0
        if abs(l - meter.level) > 0.002 || (l == 0 && meter.level != 0) { meter.level = l }
        if tick % 20 == 0 {
            let m = Self.micModeName()
            if m != micMode { micMode = m }
        }
    }

    private func save() {
        d.set(inputUID, forKey: "inputUID"); d.set(outputUID, forKey: "outputUID")
        d.set(highPass, forKey: "highPass"); d.set(limiter, forKey: "limiter"); d.set(agc, forKey: "agc")
        d.set(useAsDefault, forKey: "useAsDefault"); d.set(mode.rawValue, forKey: "mode")
    }

    /// Device list changed. Other apps' private aggregates come and go constantly during calls; those only
    /// refresh the menu. A rebuild happens only if the mic or virtual mic we use changed.
    func refreshDevices() {
        let all = AudioDevices.all()
        guard all != devices else { return }
        devices = all
        pushConfig()
    }

    // MARK: control (main thread)

    func toggle() { isRunning ? stop() : start() }

    func start(persist: Bool = true) {
        if persist { d.set(true, forKey: "enabled") }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            isRunning = true            // optimistic: UI responds instantly
            status = nil
            sendConfig()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                DispatchQueue.main.async { ok ? self.start(persist: persist) : (self.status = "Microphone access denied") }
            }
        default:
            isRunning = false
            status = "Microphone access denied. Enable Sordino in System Settings > Privacy & Security > Microphone."
        }
    }

    func stop(persist: Bool = true) {
        if persist { d.set(false, forKey: "enabled") }
        isRunning = false
        status = nil
        sendConfig()
    }

    /// Debounced: bursts of device notifications collapse into one reconcile.
    private func pushConfig() {
        pendingPush?.cancel()
        let w = DispatchWorkItem { [weak self] in self?.sendConfig() }
        pendingPush = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25, execute: w)
    }

    private func sendConfig() {
        pendingPush?.cancel(); pendingPush = nil
        let want: Config? = isRunning
            ? Config(mode: mode, inputUID: activeInput?.uid, inputID: activeInput?.id, outputUID: activeOutput?.uid, outputID: activeOutput?.id)
            : nil
        q.async {
            self.wanted = want
            self.reconcile()
        }
    }

    /// macOS moves the default input on its own when a headset connects. (main thread)
    private func defaultInputChanged() {
        guard isRunning else { return }
        q.async {
            guard let built = self.built else { return }
            switch built.mode {
            case .voiceIsolation:
                // Keep the virtual mic as the default so apps don't silently bypass Sordino.
                // Rate-limited so Sordino never ping-pongs with another app.
                let now = Clock.now()
                guard self.useDefaultOn, now - self.lastDefaultApply > 2 else { return }
                self.lastDefaultApply = now
                self.applyDefaultInput()
            case .voiceProcessing:
                self.reconcile()    // VPIO follows the default input; health check rebinds if needed
            }
        }
    }

    // MARK: reconcile (queue `q`)

    private struct Config: Equatable {
        let mode: Mode
        let inputUID: String?, inputID: AudioDeviceID?
        let outputUID: String?, outputID: AudioDeviceID?
    }

    private func reconcile() {
        guard let want = wanted else {
            if attempted != nil || built != nil {
                log.info("stopping")
                teardown()
                restoreDefaultInput()
                attempted = nil; failures = 0; lastError = nil; shownStatus = nil
            }
            return
        }
        let now = Clock.now()
        if want == attempted {
            guard let problem = healthProblem(now) else {
                if failures > 0, now - builtAt > 10 { failures = 0 }
                setStatus(nil)
                return
            }
            let wait = min(15, pow(2, Double(failures)))
            guard now - attemptAt >= wait else {
                setStatus("Reconnecting: \(problem)")
                return
            }
            failures += 1
            log.notice("rebuilding (\(problem, privacy: .public)), attempt \(self.failures)")
        } else {
            failures = 0
            log.info("building for new config")
        }
        attempted = want
        attemptAt = now
        lastError = build(want)
        setStatus(lastError.map { "Reconnecting: \($0)" })
    }

    /// nil when the running pipeline is healthy; otherwise a short user-facing reason.
    private func healthProblem(_ now: Double) -> String? {
        guard let cfg = built, let out = output, let ring else { return lastError ?? "starting" }
        let settled = now - builtAt > 2.5
        if let id = cfg.outputID, !AudioDevices.isAlive(id) { return "the virtual mic disappeared" }
        if let id = cfg.inputID, !AudioDevices.isAlive(id) { return "the microphone was disconnected" }
        if !out.isBound { return "audio was routed away from the virtual mic" }
        if settled, now - Clock.seconds(out.lastRenderTicks) > 1.5 { return "the virtual mic stopped" }
        if settled, now - Clock.seconds(ring.lastWriteTicks) > 1.5 { return "the microphone stopped sending audio" }
        if cfg.mode == .voiceProcessing {
            if engine?.isRunning != true { return "voice processing stopped" }
            if let id = cfg.inputID, AudioDevices.defaultInput() != id { return "the system input changed" }
        }
        return nil
    }

    private func setStatus(_ s: String?) {
        guard s != shownStatus else { return }
        shownStatus = s
        DispatchQueue.main.async { if self.isRunning, self.status != s { self.status = s } }
    }

    // MARK: build / teardown (queue `q`)

    private func build(_ cfg: Config) -> String? {
        teardown()
        guard let outID = cfg.outputID else { return "No virtual mic found. Install BlackHole: brew install --cask blackhole-2ch" }
        guard let inUID = cfg.inputUID, let inID = cfg.inputID else { return "No microphone found" }
        let onPeak: (Float) -> Void = { [weak self] peak in
            guard let self else { return }
            self.rawLevel = max(peak, self.rawLevel * 0.85)
        }
        do {
            let ring: RingBuffer
            switch cfg.mode {
            case .voiceIsolation:
                // Explicit mic via AVCaptureSession: independent of the system default input, so the
                // virtual mic can safely be the default. macOS Mic Mode (Voice Isolation) applies.
                ring = RingBuffer(sampleRate: MicCapture.sampleRate)
                guard let m = MicCapture(deviceUID: inUID, ring: ring, onPeak: onPeak) else {
                    throw Self.err("Couldn't open the microphone")
                }
                if useDefaultOn { applyDefaultInput() }
                m.start()
                mic = m
            case .voiceProcessing:
                // VPIO binds to the system default input at creation, so point that at the chosen mic
                // first; binding to BlackHole would loop the virtual mic into itself.
                if AudioDevices.defaultInput() != inID {
                    AudioDevices.setDefaultInput(inID)
                    Thread.sleep(forTimeInterval: 0.2)
                }
                if let cur = AudioDevices.defaultInput(), AudioDevices.all().first(where: { $0.id == cur })?.isVirtualLoopback ?? false {
                    throw Self.err("Voice Processing needs a real mic as the system input. Set Sound › Input to your microphone.")
                }
                let e = AVAudioEngine()
                let input = e.inputNode
                try input.setVoiceProcessingEnabled(true)
                input.isVoiceProcessingAGCEnabled = agcOn
                if #available(macOS 14.0, *) {
                    input.voiceProcessingOtherAudioDuckingConfiguration =
                        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
                }
                let fmt = input.outputFormat(forBus: 0)
                guard fmt.sampleRate > 0, fmt.channelCount > 0 else { throw Self.err("Input device unavailable") }
                ring = RingBuffer(sampleRate: fmt.sampleRate)
                let r = ring
                input.installTap(onBus: 0, bufferSize: 512, format: fmt) { buf, _ in
                    guard let ch = buf.floatChannelData else { return }
                    let n = Int(buf.frameLength)
                    r.write(ch[0], n)
                    var peak: Float = 0
                    for i in 0..<n { peak = max(peak, abs(ch[0][i])) }
                    onPeak(peak)
                }
                engine = e
                try e.start()
                // The engine stops itself on configuration changes; check promptly instead of waiting
                // for the stall detector. reconcile() ignores notifications that didn't break anything.
                observers.append(NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: e, queue: nil) { [weak self] _ in
                    self?.q.asyncAfter(deadline: .now() + 0.3) { self?.reconcile() }
                })
            }
            self.ring = ring

            dsp.configure(sampleRate: ring.sampleRate)
            let out = try VirtualMicOutput(deviceID: outID, sampleRate: ring.sampleRate, ring: ring, dsp: dsp)
            try out.start()
            guard out.isBound else {
                out.stop()
                throw Self.err("Couldn't route audio to the virtual mic")
            }
            output = out
            built = cfg
            builtAt = Clock.now()
            log.info("running: mode=\(cfg.mode.rawValue, privacy: .public)")
            return nil
        } catch {
            // The default input may already point at the (now silent) virtual mic: give it back while
            // retrying. A later successful build makes the virtual mic the default again.
            teardown()
            restoreDefaultInput()
            log.error("build failed: \(error.localizedDescription, privacy: .public)")
            return error.localizedDescription
        }
    }

    private func teardown() {
        observers.forEach(NotificationCenter.default.removeObserver); observers = []
        if let e = engine {
            e.inputNode.removeTap(onBus: 0)
            e.stop()
        }
        engine = nil
        mic?.stop(); mic = nil
        output?.stop(); output = nil
        ring = nil
        built = nil
        rawLevel = 0
    }

    /// Makes the virtual mic the system default input so apps pick it automatically.
    private func applyDefaultInput() {
        let all = AudioDevices.all()
        guard let bh = all.first(where: { $0.name.lowercased().contains("blackhole") && $0.hasInput }) else { return }
        let current = AudioDevices.defaultInput()
        guard current != bh.id else { return }
        if let cur = all.first(where: { $0.id == current }), !cur.isVirtualLoopback {
            d.set(cur.uid, forKey: "previousDefaultInputUID")
        }
        AudioDevices.setDefaultInput(bh.id)
    }

    /// Puts the user's real default mic back (only if the virtual mic is still the default).
    private func restoreDefaultInput() {
        let all = AudioDevices.all()
        guard let bh = all.first(where: { $0.name.lowercased().contains("blackhole") && $0.hasInput }),
              AudioDevices.defaultInput() == bh.id else { return }
        // The previous mic may have been unplugged meanwhile: fall back to any real microphone
        // rather than leaving every app on a silent virtual one.
        let prevUID = d.string(forKey: "previousDefaultInputUID")
        guard let prev = all.first(where: { $0.uid == prevUID && $0.hasInput && !$0.isVirtualLoopback })
            ?? all.first(where: { $0.hasInput && !$0.isVirtualLoopback && !$0.isInternalAggregate }) else { return }
        AudioDevices.setDefaultInput(prev.id)
    }

    private static func err(_ msg: String) -> NSError {
        NSError(domain: "Sordino", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    // MARK: helpers

    private static func micModeName() -> String {
        switch AVCaptureDevice.activeMicrophoneMode {
        case .voiceIsolation: return "Voice Isolation"
        case .wideSpectrum: return "Wide Spectrum"
        default: return "Standard"
        }
    }

    /// Opens the macOS Mic Mode picker (Voice Isolation / Wide Spectrum / Standard).
    func showMicModes() { AVCaptureDevice.showSystemUserInterface(.microphoneModes) }

    /// `Sordino --selftest [seconds]`: prints what the pipeline is doing once a second, then exits.
    private func selfTest() {
        start(persist: false)
        let args = CommandLine.arguments
        let limit = args.firstIndex(of: "--selftest").flatMap { i in i + 1 < args.count ? Int(args[i + 1]) : nil } ?? 8
        var n = 0
        Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] t in
            guard let self else { return }
            n += 1
            let all = AudioDevices.all()
            let name = { (id: AudioDeviceID?) in all.first { $0.id == id }?.name ?? "-" }
            let (out, bound, ring, problem) = self.q.sync {
                (self.output?.deviceID, self.output?.isBound ?? false, self.ring, self.healthProblem(Clock.now()))
            }
            print("t=\(n) running=\(self.isRunning) peak=\(String(format: "%.6f", self.rawLevel)) out=\(name(out)) bound=\(bound) defaultInput=\(name(AudioDevices.defaultInput())) health=\(problem ?? "ok") status=\(self.status ?? "-") under=\(ring?.underruns ?? -1) over=\(ring?.overflows ?? -1) wN=\(ring?.lastWriteN ?? -1) rN=\(ring?.lastReadN ?? -1)")
            fflush(stdout)
            if n >= limit {
                self.stop(persist: false)
                self.q.sync {}   // `stop` queues the teardown (which restores the default mic) on `q`: wait for it
                t.invalidate()
                exit(0)
            }
        }
    }
}
