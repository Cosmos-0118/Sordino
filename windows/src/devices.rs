//! Audio endpoint listing and the "which device is virtual" rules.

use wasapi::{DeviceEnumerator, Direction, Role};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDevice {
    /// Stable endpoint ID (survives renames and reboots; used for selection and routing).
    pub id: String,
    pub name: String,
    pub direction: DirectionKind,
    pub is_virtual: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionKind {
    Capture,
    Render,
}

impl From<DirectionKind> for Direction {
    fn from(d: DirectionKind) -> Self {
        match d {
            DirectionKind::Capture => Direction::Capture,
            DirectionKind::Render => Direction::Render,
        }
    }
}

/// Virtual cable / loopback drivers that must never be offered as a physical microphone, and
/// the only kind of device Sordino will ever play processed audio to.
pub fn is_virtual_name(name: &str) -> bool {
    let n = name.to_lowercase();
    ["cable input", "cable output", "vb-audio", "voicemeeter", "blackhole", "virtual cable", "virtual audio", "hi-fi cable", "vac "]
        .iter()
        .any(|k| n.contains(k))
}

/// For a virtual cable's playback endpoint ("CABLE Input (VB-Audio Virtual Cable)") returns the
/// matching recording endpoint name ("CABLE Output (VB-Audio Virtual Cable)").
pub fn paired_capture_name(render_name: &str) -> String {
    // VB-CABLE Driver Pack 45+ also registers "CABLE In 16ch", which feeds the same "CABLE Output".
    render_name.replace("In 16ch", "Output").replace("Input", "Output").replace("input", "output")
}

/// Lists active endpoints of one kind. COM must already be initialised on this thread.
pub fn list(kind: DirectionKind) -> Vec<AudioDevice> {
    let Ok(en) = DeviceEnumerator::new() else { return vec![] };
    let Ok(coll) = en.get_device_collection(kind.into()) else { return vec![] };
    let mut out = vec![];
    for dev in &coll {
        let Ok(dev) = dev else { continue };
        let (Ok(id), Ok(name)) = (dev.get_id(), dev.get_friendlyname()) else { continue };
        out.push(AudioDevice { is_virtual: is_virtual_name(&name), id, name, direction: kind });
    }
    out
}

/// Physical microphones only (virtual cables are excluded so Sordino can't capture itself).
pub fn microphones() -> Vec<AudioDevice> {
    list(DirectionKind::Capture).into_iter().filter(|d| !d.is_virtual).collect()
}

/// Virtual playback endpoints Sordino may send cleaned audio to.
pub fn virtual_outputs() -> Vec<AudioDevice> {
    list(DirectionKind::Render).into_iter().filter(|d| d.is_virtual).collect()
}

/// The recording side of the virtual cable (what apps pick as their microphone).
pub fn virtual_capture_for(render: &AudioDevice) -> Option<AudioDevice> {
    let want = paired_capture_name(&render.name);
    list(DirectionKind::Capture).into_iter().find(|d| d.is_virtual && d.name == want)
}

/// Current default capture endpoint ID for the console role.
pub fn default_capture_id() -> Option<String> {
    let en = DeviceEnumerator::new().ok()?;
    en.get_default_device_for_role(Direction::Capture, Role::Console).ok()?.get_id().ok()
}

/// The recording-device roles Windows keeps separate defaults for (eConsole, eMultimedia,
/// eCommunications). Games and browsers follow Console; Discord/Teams/Zoom follow Communications.
pub const CAPTURE_ROLES: [i32; 3] = [0, 1, 2];

fn role_of(role: i32) -> Role {
    match role {
        0 => Role::Console,
        1 => Role::Multimedia,
        _ => Role::Communications,
    }
}

/// Default capture endpoint ID for one role (see [`CAPTURE_ROLES`]).
pub fn default_capture_id_for_role(role: i32) -> Option<String> {
    let en = DeviceEnumerator::new().ok()?;
    en.get_default_device_for_role(Direction::Capture, role_of(role)).ok()?.get_id().ok()
}

/// Roles whose current default recording device satisfies `pred`. Checks every role, not just
/// Console, so a role left on a dead virtual mic can't hide behind a real mic on another role.
pub fn capture_roles_where(pred: impl Fn(&AudioDevice) -> bool) -> Vec<i32> {
    let Ok(en) = DeviceEnumerator::new() else { return vec![] };
    CAPTURE_ROLES
        .into_iter()
        .filter(|&r| {
            let Ok(dev) = en.get_default_device_for_role(Direction::Capture, role_of(r)) else { return false };
            let (Ok(id), Ok(name)) = (dev.get_id(), dev.get_friendlyname()) else { return false };
            pred(&AudioDevice { is_virtual: is_virtual_name(&name), id, name, direction: DirectionKind::Capture })
        })
        .collect()
}

/// Current default capture endpoint (console role), if Windows has one.
pub fn default_capture() -> Option<AudioDevice> {
    let en = DeviceEnumerator::new().ok()?;
    let dev = en.get_default_device_for_role(Direction::Capture, Role::Console).ok()?;
    let (id, name) = (dev.get_id().ok()?, dev.get_friendlyname().ok()?);
    Some(AudioDevice { is_virtual: is_virtual_name(&name), id, name, direction: DirectionKind::Capture })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_virtual_devices() {
        assert!(is_virtual_name("CABLE Output (VB-Audio Virtual Cable)"));
        assert!(is_virtual_name("CABLE Input (VB-Audio Virtual Cable)"));
        assert!(is_virtual_name("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)"));
        assert!(!is_virtual_name("Microphone (KT USB Audio)"));
        assert!(!is_virtual_name("Headset (Tribit XSound Go Hands-Free)"));
        assert!(!is_virtual_name("Speakers (Realtek High Definition Audio)"));
    }

    #[test]
    fn pairs_cable_endpoints() {
        assert_eq!(paired_capture_name("CABLE Input (VB-Audio Virtual Cable)"), "CABLE Output (VB-Audio Virtual Cable)");
        assert_eq!(paired_capture_name("CABLE In 16ch (VB-Audio Virtual Cable)"), "CABLE Output (VB-Audio Virtual Cable)");
    }
}
