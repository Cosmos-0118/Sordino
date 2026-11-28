#!/bin/bash
# Sordino one-click installer for macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/Cosmos-0118/Sordino/main/install.sh | bash
#
# or double-click "Install Sordino.command" in a downloaded copy of the repo.
#
# What it does:
#   1. Installs the BlackHole 2ch virtual audio driver (the "virtual mic"), if missing.
#   2. Downloads the latest Sordino release, or builds it from source if no release exists.
#   3. Installs Sordino.app into /Applications and opens it.
set -euo pipefail

REPO="${SORDINO_REPO:-Cosmos-0118/Sordino}"
BRANCH="${SORDINO_BRANCH:-main}"
APP_NAME="Sordino"
INSTALL_DIR="/Applications"
APP_DEST="$INSTALL_DIR/$APP_NAME.app"

BLACKHOLE_VERSION="0.7.1"
BLACKHOLE_URL="https://existential.audio/downloads/BlackHole2ch-${BLACKHOLE_VERSION}.pkg"
BLACKHOLE_SHA256="57b540f27a3e29c37e310e01bee0fdfab76733087e47f997ef9dccf851400dcf"
BLACKHOLE_DRIVER="/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"

# ---------- output helpers ----------
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'; else B='' G='' Y='' R='' D='' N=''; fi
step() { printf "\n${B}==> %s${N}\n" "$*"; }
ok()   { printf "${G}✓${N} %s\n" "$*"; }
warn() { printf "${Y}!${N} %s\n" "$*"; }
die()  { printf "\n${R}✗ %s${N}\n" "$*" >&2; exit 1; }

WORK="$(mktemp -d -t sordino-install)"
trap 'rm -rf "$WORK"; [ -n "${SUDO_KEEPALIVE:-}" ] && kill "$SUDO_KEEPALIVE" 2>/dev/null; true' EXIT

# Directory of this script when run from a checkout (empty when piped through curl | bash).
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ---------- checks ----------
printf "${B}Sordino installer${N}\n${D}Cleaner voice for Discord, WhatsApp, Zoom and games.${N}\n"

[ "$(uname -s)" = "Darwin" ] || die "Sordino only runs on macOS."
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
[ "$MACOS_MAJOR" -ge 13 ] || die "Sordino needs macOS 13 Ventura or newer (you have $(sw_vers -productVersion))."
ok "macOS $(sw_vers -productVersion)"

# Ask for the admin password once, up front, and keep sudo alive for the rest of the run.
SUDO_KEEPALIVE=""
need_sudo() {
    [ -n "$SUDO_KEEPALIVE" ] && return 0
    if ! sudo -n true 2>/dev/null; then
        printf "\nYour Mac password is needed to install the audio driver and the app.\n"
        printf "${D}(Nothing appears while you type — that's normal. Press Return when done.)${N}\n"
        sudo -v || die "Password not accepted."
    fi
    ( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) 2>/dev/null &
    SUDO_KEEPALIVE=$!
}

# ---------- 1. BlackHole ----------
step "Virtual mic (BlackHole 2ch)"
if [ -d "$BLACKHOLE_DRIVER" ]; then
    ok "BlackHole is already installed"
else
    need_sudo
    echo "Downloading BlackHole ${BLACKHOLE_VERSION}…"
    curl -fL --progress-bar -o "$WORK/BlackHole.pkg" "$BLACKHOLE_URL" || die "Couldn't download BlackHole. Check your internet connection."
    echo "$BLACKHOLE_SHA256  $WORK/BlackHole.pkg" | shasum -a 256 -c - >/dev/null 2>&1 \
        || die "BlackHole download failed its integrity check. Please try again."
    sudo installer -pkg "$WORK/BlackHole.pkg" -target / >/dev/null || die "BlackHole installation failed."
    echo "Restarting the macOS audio system so the new device appears…"
    sudo killall coreaudiod 2>/dev/null || true
    sleep 3
    ok "BlackHole installed"
fi

# ---------- 2. Get Sordino.app ----------
step "Getting Sordino"
APP_SRC=""

download_release() {
    local url="https://github.com/$REPO/releases/latest/download/$APP_NAME.zip"
    curl -fsL -o "$WORK/$APP_NAME.zip" "$url" 2>/dev/null || return 1
    # Verify the download against the checksum published with the release; if it is missing or
    # does not match, fall back to building from source instead of running unverified code.
    local expected actual
    expected="$(curl -fsL "$url.sha256" 2>/dev/null | awk 'NR==1 {print tolower($1)}')" || return 1
    actual="$(shasum -a 256 "$WORK/$APP_NAME.zip" | awk '{print tolower($1)}')"
    if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
        warn "The Sordino download failed its integrity check; building from source instead."
        return 1
    fi
    ditto -x -k "$WORK/$APP_NAME.zip" "$WORK/release" || return 1
    [ -d "$WORK/release/$APP_NAME.app" ] || return 1
    APP_SRC="$WORK/release/$APP_NAME.app"
}

ensure_toolchain() {
    if xcode-select -p >/dev/null 2>&1 && xcrun --find swift >/dev/null 2>&1; then return 0; fi
    warn "Apple's Command Line Tools are needed to build Sordino."
    echo "A window will pop up — click ${B}Install${N} and wait for it to finish (5–15 minutes)."
    xcode-select --install >/dev/null 2>&1 || true
    printf "Waiting for the Command Line Tools"
    for _ in $(seq 1 360); do   # up to 30 minutes
        if xcode-select -p >/dev/null 2>&1 && xcrun --find swift >/dev/null 2>&1; then echo; ok "Command Line Tools ready"; return 0; fi
        printf "."; sleep 5
    done
    echo
    die "Command Line Tools didn't finish installing. Run this installer again once they're done."
}

build_from_source() {
    local src=""
    if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/mac/Package.swift" ]; then
        src="$SCRIPT_DIR/mac"
    else
        echo "Downloading source…"
        curl -fsL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" | tar -xz -C "$WORK" \
            || die "Couldn't download the Sordino source."
        src="$(find "$WORK" -maxdepth 3 -name Package.swift -path '*/mac/*' -exec dirname {} \; | head -1)"
        [ -n "$src" ] || die "Downloaded source doesn't contain the Mac app."
    fi
    ensure_toolchain
    echo "Building Sordino (about a minute)…"
    ( cd "$src" && ./build.sh ) >"$WORK/build.log" 2>&1 || { tail -n 30 "$WORK/build.log" >&2; die "Build failed (log above)."; }
    APP_SRC="$src/build/$APP_NAME.app"
}

if [ "${SORDINO_FROM_SOURCE:-0}" != "1" ] && download_release; then
    ok "Downloaded the latest release"
else
    [ "${SORDINO_FROM_SOURCE:-0}" = "1" ] || warn "No prebuilt release found — building from source instead."
    build_from_source
    ok "Built Sordino"
fi

# ---------- 3. Install ----------
step "Installing to $INSTALL_DIR"
if pgrep -x "$APP_NAME" >/dev/null; then
    osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
    sleep 1
    pkill -x "$APP_NAME" 2>/dev/null || true
fi

copy_app() { rm -rf "$APP_DEST" && ditto "$APP_SRC" "$APP_DEST"; }
if [ -w "$INSTALL_DIR" ] && { [ ! -e "$APP_DEST" ] || [ -w "$APP_DEST" ]; }; then
    copy_app
else
    need_sudo
    sudo rm -rf "$APP_DEST" && sudo ditto "$APP_SRC" "$APP_DEST"
    sudo chown -R "$(id -un)":staff "$APP_DEST"
fi
# The app is ad-hoc signed; clear the download flag so Gatekeeper doesn't block it.
xattr -dr com.apple.quarantine "$APP_DEST" 2>/dev/null || true
ok "Installed $APP_DEST"

# ---------- 4. Launch ----------
step "Starting Sordino"
open "$APP_DEST"
ok "Sordino is running — look for the waveform icon in your menu bar"

cat <<EOF

${B}Almost done:${N}
  1. Click ${B}OK${N} when macOS asks for microphone access.
  2. In Discord / WhatsApp / Zoom / your game, set the ${B}microphone${N} (input) to ${B}BlackHole 2ch${N}.
     Keep the ${B}speaker${N} (output) on your normal speakers or headphones — never BlackHole.
  3. In Discord, also turn off Noise Suppression and Automatic Input Sensitivity.

${D}Tip: on laptop speakers without headphones, pick "Voice Processing" in the Sordino menu —
it cancels echo so the other person never hears themselves.${N}

To uninstall later: run uninstall.sh (or "Uninstall Sordino.command").
EOF
