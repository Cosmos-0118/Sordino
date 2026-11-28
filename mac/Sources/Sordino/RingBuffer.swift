import Foundation
import os

/// Mono float ring buffer bridging the capture clock to the playback clock.
/// Self-corrects drift: drops old audio when too full, re-primes after an underrun.
final class RingBuffer {
    private let buf: UnsafeMutablePointer<Float>
    private let cap: Int
    private let target: Int
    private let maxFill: Int
    private var r = 0
    private var w = 0
    private var primed = false
    var underruns = 0, overflows = 0, lastWriteN = 0, lastReadN = 0
    /// mach ticks of the last write; the pipeline's watchdog uses it to spot a capture that went silent.
    private(set) var lastWriteTicks: UInt64 = mach_absolute_time()
    let sampleRate: Double
    private let lock: UnsafeMutablePointer<os_unfair_lock>

    init(sampleRate: Double) {
        self.sampleRate = sampleRate
        target = Int(sampleRate * 0.04)      // 40 ms cushion
        maxFill = Int(sampleRate * 0.12)     // drop above 120 ms
        cap = Int(sampleRate * 0.5)
        buf = .allocate(capacity: cap)
        buf.initialize(repeating: 0, count: cap)
        lock = .allocate(capacity: 1)
        lock.initialize(to: os_unfair_lock())
    }

    deinit { buf.deallocate(); lock.deallocate() }

    func write(_ p: UnsafePointer<Float>, _ n: Int) {
        os_unfair_lock_lock(lock)
        for i in 0..<n { buf[(w + i) % cap] = p[i] }
        w += n
        lastWriteN = n
        lastWriteTicks = mach_absolute_time()
        if w - r > maxFill { r = w - target; overflows += 1 }
        os_unfair_lock_unlock(lock)
    }

    func read(_ out: UnsafeMutablePointer<Float>, _ n: Int) {
        os_unfair_lock_lock(lock)
        lastReadN = n
        let avail = w - r
        if !primed && avail >= target { primed = true }
        if !primed || avail < n {
            if avail < n && primed { underruns += 1 }
            if avail < n { primed = false }
            os_unfair_lock_unlock(lock)
            out.update(repeating: 0, count: n)
            return
        }
        for i in 0..<n { out[i] = buf[(r + i) % cap] }
        r += n
        os_unfair_lock_unlock(lock)
    }
}
