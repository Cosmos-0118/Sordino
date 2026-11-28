import AudioToolbox
import CoreAudio
import Foundation

/// Monotonic seconds (stops while the Mac sleeps, which is what stall detection wants).
enum Clock {
    private static let scale: Double = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return Double(tb.numer) / Double(tb.denom) / 1e9
    }()
    static func now() -> Double { seconds(mach_absolute_time()) }
    static func seconds(_ ticks: UInt64) -> Double { Double(ticks) * scale }
}

/// High-pass (90 Hz, 2nd-order Butterworth) and a fast peak limiter, run on the output's render thread.
/// Plain-Swift DSP so the output can be a bare AUHAL with no AVAudioEngine graph.
final class VoiceDSP {
    var highPass = true
    var limiter = true

    private var b0: Float = 1, b1: Float = 0, b2: Float = 0, a1: Float = 0, a2: Float = 0
    private var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
    private var env: Float = 0
    private var release: Float = 0.999
    private let threshold: Float = 0.891   // -1 dBFS

    /// Call only while no output is rendering.
    func configure(sampleRate fs: Double) {
        let w0 = 2 * Double.pi * 90 / fs
        let alpha = sin(w0) / (2 * 0.7071)
        let c = cos(w0), a0 = 1 + alpha
        b0 = Float((1 + c) / 2 / a0); b1 = Float(-(1 + c) / a0); b2 = b0
        a1 = Float(-2 * c / a0); a2 = Float((1 - alpha) / a0)
        release = Float(exp(-1 / (0.08 * fs)))
        x1 = 0; x2 = 0; y1 = 0; y2 = 0; env = 0
    }

    func process(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
        let hp = highPass, lim = limiter
        for i in 0..<n {
            let x = p[i]
            // Filter state always advances, so toggling the high-pass never clicks.
            var y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            if abs(y) < 1e-15 { y = 0 }
            x2 = x1; x1 = x; y2 = y1; y1 = y
            var s = hp ? y : x
            if lim {
                let a = abs(s)
                env = a > env ? a : env * release
                if env > threshold { s *= threshold / env }
            }
            p[i] = max(-1, min(1, s))
        }
    }
}

/// Plays the cleaned voice into ONE specific device (the virtual mic) through a raw AUHAL.
///
/// Unlike AVAudioEngine, an AUHAL bound with kAudioOutputUnitProperty_CurrentDevice never follows the
/// system default output, so a device change (a WhatsApp call, AirPods connecting, a sample-rate change)
/// can't move the voice onto the speakers. The AUHAL also converts our mono client format to whatever
/// rate/channels the device runs at, so other apps reconfiguring BlackHole don't force a rebuild.
final class VirtualMicOutput {
    let deviceID: AudioDeviceID
    private var unit: AudioUnit?
    private let ring: RingBuffer
    private let dsp: VoiceDSP
    private let maxFrames = 4096
    /// mach ticks of the last render callback (written on the render thread).
    private(set) var lastRenderTicks: UInt64 = 0

    init(deviceID: AudioDeviceID, sampleRate: Double, ring: RingBuffer, dsp: VoiceDSP) throws {
        self.deviceID = deviceID
        self.ring = ring
        self.dsp = dsp

        var desc = AudioComponentDescription(componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
                                             componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        guard let comp = AudioComponentFindNext(nil, &desc) else { throw Self.err("No HAL output unit", -1) }
        var u: AudioUnit?
        try Self.ok(AudioComponentInstanceNew(comp, &u), "Creating the output")
        guard let u else { throw Self.err("Creating the output", -1) }
        unit = u
        do {
            var on: UInt32 = 1, off: UInt32 = 0
            let u32 = UInt32(MemoryLayout<UInt32>.size)
            try Self.ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &on, u32), "Enabling output")
            try Self.ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &off, u32), "Disabling input")
            var dev = deviceID
            try Self.ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev,
                                             UInt32(MemoryLayout<AudioDeviceID>.size)), "Binding to the virtual mic")

            let channels = UInt32(max(1, min(2, AudioDevices.outputChannels(deviceID))))
            var fmt = AudioStreamBasicDescription(
                mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
                mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: channels,
                mBitsPerChannel: 32, mReserved: 0)
            try Self.ok(AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &fmt,
                                             UInt32(MemoryLayout<AudioStreamBasicDescription>.size)), "Setting the output format")
            var maxF = UInt32(maxFrames)
            AudioUnitSetProperty(u, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &maxF, u32)

            var cb = AURenderCallbackStruct(inputProc: renderProc, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
            try Self.ok(AudioUnitSetProperty(u, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb,
                                             UInt32(MemoryLayout<AURenderCallbackStruct>.size)), "Setting the render callback")
            try Self.ok(AudioUnitInitialize(u), "Initializing the output")
        } catch {
            stop()
            throw error
        }
    }

    deinit { stop() }

    func start() throws {
        guard let unit else { throw Self.err("Output was disposed", -1) }
        lastRenderTicks = mach_absolute_time()
        try Self.ok(AudioOutputUnitStart(unit), "Starting the output")
    }

    /// Idempotent. Disposing waits for any in-flight render callback, so `self` outlives it.
    func stop() {
        guard let u = unit else { return }
        unit = nil
        AudioOutputUnitStop(u)
        AudioUnitUninitialize(u)
        AudioComponentInstanceDispose(u)
    }

    /// True while the unit still sends to the device it was created for.
    var isBound: Bool {
        guard let unit else { return false }
        var dev = AudioDeviceID(0); var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioUnitGetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, &size) == noErr else { return false }
        return dev == deviceID
    }

    fileprivate func render(_ frames: UInt32, _ io: UnsafeMutablePointer<AudioBufferList>?) -> OSStatus {
        lastRenderTicks = mach_absolute_time()
        guard let io else { return noErr }
        let list = UnsafeMutableAudioBufferListPointer(io)
        let n = Int(frames)
        guard list.count > 0, let first = list[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
        guard n <= maxFrames, Int(list[0].mDataByteSize) >= n * 4 else {
            for b in list { if let d = b.mData { memset(d, 0, Int(b.mDataByteSize)) } }
            return noErr
        }
        ring.read(first, n)
        dsp.process(first, n)
        for k in 1..<max(1, list.count) {
            if let d = list[k].mData, Int(list[k].mDataByteSize) >= n * 4 { memcpy(d, first, n * 4) }
        }
        return noErr
    }

    private static func ok(_ s: OSStatus, _ what: String) throws { if s != noErr { throw err(what, s) } }
    private static func err(_ what: String, _ s: OSStatus) -> NSError {
        NSError(domain: "Sordino", code: Int(s), userInfo: [NSLocalizedDescriptionKey: "\(what) failed (\(s))"])
    }
}

private let renderProc: AURenderCallback = { refCon, _, _, _, frames, io in
    Unmanaged<VirtualMicOutput>.fromOpaque(refCon).takeUnretainedValue().render(frames, io)
}
