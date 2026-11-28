import AppKit
import Combine
import SwiftUI

/// Menu-bar-only app: no SwiftUI scenes (an empty `Settings` scene shows a blank "Settings" window).
@main
enum SordinoApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
        withExtendedLifetime(delegate) {}
    }
}

/// Menu-bar item backed by a real NSMenu, laid out like the system Wi-Fi / Sound / Bluetooth menus:
/// a bold header with a switch, section headers, selectable rows with circular icons, and
/// "Settings…" / "Quit" at the bottom. AppKit owns sizing, material, highlight and positioning.
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private let pipe = Pipeline()
    private var statusItem: NSStatusItem!
    private let menu = NSMenu()
    private var bag = Set<AnyCancellable>()
    private var refreshers: [() -> Void] = []
    private var signature = ""
    private var pendingSync = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        menu.delegate = self
        menu.autoenablesItems = false
        menu.minimumWidth = MenuMetrics.width
        statusItem.menu = menu
        updateIcon(pipe.isRunning)
        rebuild()

        pipe.$isRunning.receive(on: DispatchQueue.main).sink { [weak self] in self?.updateIcon($0) }.store(in: &bag)
        // DispatchQueue.main (not RunLoop.main) so updates also land while the menu is tracking.
        pipe.objectWillChange.receive(on: DispatchQueue.main).sink { [weak self] _ in self?.scheduleSync() }.store(in: &bag)

        if CommandLine.arguments.contains("--open-menu") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.statusItem.button?.performClick(nil) }
        }
    }

    private func updateIcon(_ running: Bool) {
        let img = NSImage(systemSymbolName: running ? "waveform.badge.mic" : "mic.slash", accessibilityDescription: "Sordino")
        img?.isTemplate = true
        statusItem.button?.image = img
    }

    // MARK: NSMenuDelegate

    func menuNeedsUpdate(_ menu: NSMenu) { rebuild() }

    // MARK: Sync

    private func scheduleSync() {
        guard !pendingSync else { return }
        pendingSync = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.pendingSync = false
            if self.currentSignature() != self.signature { self.rebuild() } else { self.refreshers.forEach { $0() } }
        }
    }

    /// Anything that changes which rows exist. Value-only changes just refresh controls in place,
    /// so a switch never gets torn down mid-animation.
    private func currentSignature() -> String {
        [pipe.inputs.map { $0.uid + $0.name }.joined(), pipe.outputs.map { $0.uid + $0.name }.joined(),
         pipe.activeInput?.uid ?? "", pipe.activeOutput?.uid ?? "", pipe.mode.rawValue, pipe.status ?? "", pipe.micMode,
         pipe.isRunning ? "on" : "off",
         pipe.blackHole == nil ? "0" : "1"].joined(separator: "|")
    }

    // MARK: Build

    private func rebuild() {
        signature = currentSignature()
        refreshers.removeAll()
        menu.removeAllItems()

        // Header
        let header = HeaderView(title: "Sordino") { [weak self] in self?.pipe.toggle() }
        add(view: header)
        refreshers.append { [weak self, weak header] in
            guard let self, let header else { return }
            header.isOn = self.pipe.isRunning
            header.subtitle = !self.pipe.isRunning ? "Off" : (self.pipe.status == nil ? "On — cleaning your voice" : "On — reconnecting…")
        }
        let meter = LevelMeterView(model: pipe.meter)
        add(view: meter)

        if pipe.blackHole == nil {
            add(view: NoticeView(symbol: "exclamationmark.triangle.fill", tint: .systemOrange,
                                 title: "BlackHole isn’t installed",
                                 message: "Run “brew install --cask blackhole-2ch”, then restart your Mac."))
        }
        if let s = pipe.status {
            if pipe.isRunning {
                add(view: NoticeView(symbol: "arrow.triangle.2.circlepath", tint: .systemOrange, title: "Reconnecting…", message: s))
            } else {
                add(view: NoticeView(symbol: "xmark.octagon.fill", tint: .systemRed, title: "Something went wrong", message: s))
            }
        }

        // Microphone
        menu.addItem(.separator())
        addSection("Microphone")
        for dev in pipe.inputs {
            addChoice(dev.name, symbol: Self.inputSymbol(dev.name), selected: dev.uid == pipe.activeInput?.uid) { [weak self] in
                self?.pipe.inputUID = dev.uid
            }
        }
        if pipe.inputs.isEmpty { addDisabled("No microphones found") }

        // Virtual mic (loopback devices first, everything else tucked into a submenu)
        menu.addItem(.separator())
        addSection("Virtual Mic")
        let activeOut = pipe.activeOutput?.uid
        let loopbacks = pipe.outputs.filter { $0.isVirtualLoopback || $0.uid == activeOut }
        let others = pipe.outputs.filter { !($0.isVirtualLoopback || $0.uid == activeOut) }
        for dev in loopbacks {
            addChoice(dev.name, symbol: "point.3.connected.trianglepath.dotted", selected: dev.uid == activeOut) { [weak self] in
                self?.pipe.outputUID = dev.uid
            }
        }
        if loopbacks.isEmpty { addDisabled("No virtual mic found") }
        if !others.isEmpty {
            let more = NSMenuItem(title: "Other Devices", action: nil, keyEquivalent: "")
            more.image = CircleIcon.make(symbol: "ellipsis", selected: false)
            let sub = NSMenu()
            for dev in others {
                let item = ActionItem(dev.name) { [weak self] in self?.pipe.outputUID = dev.uid }
                sub.addItem(item)
            }
            more.submenu = sub
            menu.addItem(more)
        }

        // Processing
        menu.addItem(.separator())
        addSection("Processing")
        addChoice("Voice Isolation", symbol: "person.wave.2.fill", selected: pipe.mode == .voiceIsolation,
                  subtitle: "Works as the default mic") { [weak self] in self?.pipe.mode = .voiceIsolation }
        addChoice("Voice Processing", symbol: "waveform", selected: pipe.mode == .voiceProcessing,
                  subtitle: "Select BlackHole in each app") { [weak self] in self?.pipe.mode = .voiceProcessing }
        let micMode = ActionItem("Mic Mode…") { [weak self] in self?.pipe.showMicModes() }
        micMode.image = CircleIcon.make(symbol: "mic.badge.plus", selected: false)
        if #available(macOS 14.4, *) {
            micMode.subtitle = pipe.micMode == "Voice Isolation" ? "Voice Isolation" : "\(pipe.micMode) — Voice Isolation recommended"
        }
        menu.addItem(micMode)

        // Options
        menu.addItem(.separator())
        addSection("Options")
        if pipe.mode == .voiceIsolation {
            addToggle("Use as Default Mic", help: "Apps pick up Sordino automatically", \.useAsDefault)
        }
        addToggle("Reduce Air Thumps", help: "High-pass filter at 90 Hz", \.highPass)
        addToggle("Peak Limiter", help: "Stops loud peaks from clipping", \.limiter)
        if pipe.mode == .voiceProcessing {
            addToggle("Auto Gain", help: "Apple’s automatic level control", \.agc)
        }
        addToggle("Open at Login", help: nil, \.launchAtLogin)

        // Footer
        menu.addItem(.separator())
        menu.addItem(ActionItem("Sound Settings…") {
            if let url = URL(string: "x-apple.systempreferences:com.apple.Sound-Settings.extension") { NSWorkspace.shared.open(url) }
        })
        let quit = NSMenuItem(title: "Quit Sordino", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)

        refreshers.forEach { $0() }
    }

    private func add(view: NSView) {
        let item = NSMenuItem()
        item.view = view
        menu.addItem(item)
    }

    private func addSection(_ title: String) {
        if #available(macOS 14.0, *) {
            menu.addItem(.sectionHeader(title: title))
        } else {
            addDisabled(title)
        }
    }

    private func addDisabled(_ title: String) {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        menu.addItem(item)
    }

    private func addChoice(_ title: String, symbol: String, selected: Bool, subtitle: String? = nil, _ action: @escaping () -> Void) {
        let item = ActionItem(title, action)
        item.image = CircleIcon.make(symbol: symbol, selected: selected)
        item.state = .off
        if let subtitle, #available(macOS 14.4, *) { item.subtitle = subtitle }
        item.setAccessibilitySelected(selected)
        menu.addItem(item)
    }

    private func addToggle(_ title: String, help: String?, _ key: ReferenceWritableKeyPath<Pipeline, Bool>) {
        let row = ToggleRowView(title: title) { [weak self] in self?.pipe[keyPath: key] = $0 }
        row.toolTip = help
        add(view: row)
        refreshers.append { [weak self, weak row] in
            guard let self, let row else { return }
            row.isOn = self.pipe[keyPath: key]
        }
    }

    private static func inputSymbol(_ name: String) -> String {
        let n = name.lowercased()
        if n.contains("airpods") { return "airpods.gen3" }
        if n.contains("macbook") { return "laptopcomputer" }
        if n.contains("iphone") { return "iphone" }
        if n.contains("display") || n.contains("studio") { return "display" }
        if n.contains("headset") || n.contains("headphone") { return "headphones" }
        return "mic.fill"
    }
}

// MARK: - Native item helpers

/// NSMenuItem that runs a closure.
private final class ActionItem: NSMenuItem {
    private let handler: () -> Void
    init(_ title: String, _ handler: @escaping () -> Void) {
        self.handler = handler
        super.init(title: title, action: #selector(fire), keyEquivalent: "")
        target = self
    }
    required init(coder: NSCoder) { fatalError() }
    @objc private func fire() { handler() }
}

/// The round icon used by Control Center / Sound / Bluetooth menu rows:
/// accent-filled with a white glyph when selected, a subtle grey disc otherwise.
private enum CircleIcon {
    static let size: CGFloat = 26

    static func make(symbol: String, selected: Bool) -> NSImage {
        let base = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
            ?? NSImage(systemSymbolName: "speaker.wave.2.fill", accessibilityDescription: nil)!
        let img = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
            let disc = NSBezierPath(ovalIn: rect)
            (selected ? NSColor.controlAccentColor : NSColor.labelColor.withAlphaComponent(0.1)).setFill()
            disc.fill()
            let glyphColor: NSColor = selected ? .white : .labelColor
            let config = NSImage.SymbolConfiguration(pointSize: 12, weight: .medium)
                .applying(.init(paletteColors: [glyphColor]))
            if let g = base.withSymbolConfiguration(config) {
                let s = g.size
                g.draw(in: NSRect(x: rect.midX - s.width / 2, y: rect.midY - s.height / 2, width: s.width, height: s.height))
            }
            return true
        }
        img.isTemplate = false
        return img
    }
}

// MARK: - Custom row views

enum MenuMetrics {
    static let width: CGFloat = 300
    /// Leading/trailing inset that lines custom rows up with native item text.
    static let inset: CGFloat = 14
}

/// Bold title + status line on the left, a switch on the right (like the Wi-Fi / Bluetooth header).
private final class HeaderView: NSView {
    private let titleLabel = NSTextField(labelWithString: "")
    private let subtitleLabel = NSTextField(labelWithString: "")
    private let toggle: PillSwitch
    private let onToggle: () -> Void

    var isOn: Bool {
        get { toggle.isOn }
        set { toggle.setOn(newValue) }
    }
    var subtitle: String {
        get { subtitleLabel.stringValue }
        set { subtitleLabel.stringValue = newValue }
    }

    init(title: String, onToggle: @escaping () -> Void) {
        self.onToggle = onToggle
        toggle = PillSwitch(width: 32, height: 18)
        super.init(frame: NSRect(x: 0, y: 0, width: MenuMetrics.width, height: 44))
        autoresizingMask = [.width]
        titleLabel.stringValue = title
        titleLabel.font = .systemFont(ofSize: 13, weight: .bold)
        subtitleLabel.font = .systemFont(ofSize: 11)
        subtitleLabel.textColor = .secondaryLabelColor
        subtitleLabel.lineBreakMode = .byTruncatingTail
        toggle.target = self
        toggle.action = #selector(changed)
        toggle.setAccessibilityLabel(title)
        [titleLabel, subtitleLabel, toggle].forEach(addSubview)
    }
    required init?(coder: NSCoder) { fatalError() }

    @objc private func changed() { onToggle() }

    override func layout() {
        super.layout()
        let i = MenuMetrics.inset
        toggle.sizeToFit()
        let ts = toggle.frame.size
        toggle.frame.origin = NSPoint(x: bounds.width - i - ts.width, y: (bounds.height - ts.height) / 2)
        let textW = toggle.frame.minX - 8 - i
        titleLabel.frame = NSRect(x: i, y: bounds.height / 2 - 1, width: textW, height: 17)
        subtitleLabel.frame = NSRect(x: i, y: bounds.height / 2 - 16, width: textW, height: 14)
    }
}

/// Mic glyph + segmented input level, like System Settings › Sound › Input level.
private final class LevelMeterView: NSView {
    private let icon = NSImageView()
    private var level: Float = 0
    private var sub: AnyCancellable?
    private static let segments = 24

    init(model: LevelModel) {
        super.init(frame: NSRect(x: 0, y: 0, width: MenuMetrics.width, height: 26))
        autoresizingMask = [.width]
        icon.image = NSImage(systemSymbolName: "mic.fill", accessibilityDescription: "Input level")
        icon.symbolConfiguration = .init(pointSize: 12, weight: .medium)
        icon.contentTintColor = .secondaryLabelColor
        addSubview(icon)
        sub = model.$level.receive(on: DispatchQueue.main).sink { [weak self] in
            self?.level = $0
            self?.needsDisplay = true
        }
        setAccessibilityElement(true)
        setAccessibilityRole(.levelIndicator)
        setAccessibilityLabel("Input level")
    }
    required init?(coder: NSCoder) { fatalError() }

    private var fraction: CGFloat {
        guard level > 0.0001 else { return 0 }
        return CGFloat(max(0, min(1, (20 * log10(level) + 60) / 60)))
    }

    override func layout() {
        super.layout()
        icon.frame = NSRect(x: MenuMetrics.inset, y: (bounds.height - 16) / 2, width: 16, height: 16)
    }

    override func draw(_ dirtyRect: NSRect) {
        let i = MenuMetrics.inset
        let x0 = icon.frame.maxX + 8
        let width = bounds.width - i - x0
        let n = Self.segments
        let gap: CGFloat = 3
        let segW = (width - gap * CGFloat(n - 1)) / CGFloat(n)
        let h: CGFloat = 8
        let y = (bounds.height - h) / 2
        let lit = Int((fraction * CGFloat(n)).rounded(.up))
        for k in 0..<n {
            let r = NSRect(x: x0 + CGFloat(k) * (segW + gap), y: y, width: segW, height: h)
            let color: NSColor
            if k < lit {
                color = k >= n - 2 ? .systemRed : (k >= n - 6 ? .systemYellow : .systemGreen)
            } else {
                color = NSColor.labelColor.withAlphaComponent(0.12)
            }
            color.setFill()
            NSBezierPath(roundedRect: r, xRadius: segW / 2, yRadius: segW / 2).fill()
        }
    }
}

/// Title on the left, mini switch on the right. Clicking anywhere on the row flips it and keeps the menu open.
private final class ToggleRowView: NSView {
    private let label = NSTextField(labelWithString: "")
    private let toggle: PillSwitch
    private let onChange: (Bool) -> Void

    var isOn: Bool {
        get { toggle.isOn }
        set { toggle.setOn(newValue) }
    }

    init(title: String, onChange: @escaping (Bool) -> Void) {
        self.onChange = onChange
        toggle = PillSwitch(width: 28, height: 16)
        super.init(frame: NSRect(x: 0, y: 0, width: MenuMetrics.width, height: 28))
        autoresizingMask = [.width]
        label.stringValue = title
        label.font = .menuFont(ofSize: 0)
        label.lineBreakMode = .byTruncatingTail
        toggle.target = self
        toggle.action = #selector(changed)
        toggle.setAccessibilityLabel(title)
        [label, toggle].forEach(addSubview)
    }
    required init?(coder: NSCoder) { fatalError() }

    @objc private func changed() { onChange(isOn) }

    override func mouseUp(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        guard bounds.contains(p), !toggle.frame.contains(p) else { return }
        toggle.flip()
    }

    override func layout() {
        super.layout()
        let i = MenuMetrics.inset
        toggle.sizeToFit()
        let ts = toggle.frame.size
        toggle.frame.origin = NSPoint(x: bounds.width - i - ts.width, y: (bounds.height - ts.height) / 2)
        let lh = label.intrinsicContentSize.height
        label.frame = NSRect(x: i, y: (bounds.height - lh) / 2, width: toggle.frame.minX - 8 - i, height: lh)
    }
}

/// Coloured glyph + bold title + wrapping message. Height is measured for the menu width, so nothing truncates.
private final class NoticeView: NSView {
    private let icon = NSImageView()
    private let titleLabel = NSTextField(labelWithString: "")
    private let messageLabel = NSTextField(wrappingLabelWithString: "")
    private static let iconW: CGFloat = 16

    init(symbol: String, tint: NSColor, title: String, message: String) {
        super.init(frame: .zero)
        autoresizingMask = [.width]
        icon.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
        icon.symbolConfiguration = .init(pointSize: 13, weight: .semibold)
        icon.contentTintColor = tint
        titleLabel.stringValue = title
        titleLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        messageLabel.stringValue = message
        messageLabel.font = .systemFont(ofSize: 11)
        messageLabel.textColor = .secondaryLabelColor
        messageLabel.isSelectable = true
        [icon, titleLabel, messageLabel].forEach(addSubview)
        frame = NSRect(x: 0, y: 0, width: MenuMetrics.width, height: measuredHeight(for: MenuMetrics.width))
    }
    required init?(coder: NSCoder) { fatalError() }

    private func textWidth(for width: CGFloat) -> CGFloat { width - MenuMetrics.inset * 2 - Self.iconW - 8 }

    private func measuredHeight(for width: CGFloat) -> CGFloat {
        let w = textWidth(for: width)
        let msgH = messageLabel.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: w, height: .greatestFiniteMagnitude)).height
        return 6 + 16 + 2 + ceil(msgH) + 6
    }

    override func layout() {
        super.layout()
        let i = MenuMetrics.inset
        let x = i + Self.iconW + 8
        let w = textWidth(for: bounds.width)
        let msgH = ceil(messageLabel.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: w, height: .greatestFiniteMagnitude)).height)
        icon.frame = NSRect(x: i, y: bounds.height - 6 - 16, width: Self.iconW, height: 16)
        titleLabel.frame = NSRect(x: x, y: bounds.height - 6 - 16, width: w, height: 16)
        messageLabel.frame = NSRect(x: x, y: bounds.height - 6 - 16 - 2 - msgH, width: w, height: msgH)
    }
}

/// NSSwitch draws in its inactive (grey) style inside menus because the menu window is never key.
/// This draws the active look the system menus use: accent track when on, white knob, animated.
private final class PillSwitch: NSControl {
    private let size: NSSize
    private var progress: CGFloat = 0   // 0 = off, 1 = on
    private var anim: Timer?

    private(set) var isOn = false

    init(width: CGFloat, height: CGFloat) {
        size = NSSize(width: width, height: height)
        super.init(frame: NSRect(origin: .zero, size: size))
        setAccessibilityElement(true)
        setAccessibilityRole(.checkBox)
        setAccessibilitySubrole(.switch)
    }
    required init?(coder: NSCoder) { fatalError() }

    override var intrinsicContentSize: NSSize { size }
    override func sizeToFit() { setFrameSize(size) }
    override func accessibilityValue() -> Any? { isOn ? 1 : 0 }
    override func accessibilityPerformPress() -> Bool { flip(); return true }

    /// Animates only when visible, so a freshly opened menu shows the real state immediately.
    func setOn(_ on: Bool) {
        guard on != isOn else { return }
        isOn = on
        if window?.isVisible == true { animate() } else { anim?.invalidate(); progress = on ? 1 : 0; needsDisplay = true }
    }

    private func animate() {
        anim?.invalidate()
        let target: CGFloat = isOn ? 1 : 0
        let start = progress, t0 = CACurrentMediaTime(), dur = 0.16
        let t = Timer(timeInterval: 1.0 / 120, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            let x = min(1, (CACurrentMediaTime() - t0) / dur)
            let e = CGFloat(1 - pow(1 - x, 3))
            self.progress = start + (target - start) * e
            self.needsDisplay = true
            if x >= 1 { timer.invalidate() }
        }
        RunLoop.main.add(t, forMode: .common)
        anim = t
    }

    func flip() {
        isOn.toggle()
        animate()
        sendAction(action, to: target)
    }

    override func mouseDown(with event: NSEvent) {}
    override func mouseUp(with event: NSEvent) {
        if bounds.contains(convert(event.locationInWindow, from: nil)) { flip() }
    }

    override func draw(_ dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: 0.5, dy: 0.5)
        let radius = r.height / 2
        let off = NSColor.labelColor.withAlphaComponent(0.16)
        let on = NSColor.controlAccentColor
        (off.blended(withFraction: progress, of: on) ?? on).setFill()
        NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius).fill()

        let pad: CGFloat = 1.5
        let d = r.height - pad * 2
        let x = r.minX + pad + (r.width - pad * 2 - d) * progress
        let knob = NSRect(x: x, y: r.minY + pad, width: d, height: d)
        NSGraphicsContext.saveGraphicsState()
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.3)
        shadow.shadowBlurRadius = 1.5
        shadow.shadowOffset = NSSize(width: 0, height: -0.5)
        shadow.set()
        NSColor.white.setFill()
        NSBezierPath(ovalIn: knob).fill()
        NSGraphicsContext.restoreGraphicsState()
    }
}
