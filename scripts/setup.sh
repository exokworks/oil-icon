#!/usr/bin/env bash
# Install and verify the slicing environment, independently of the caller's cwd.
set -euo pipefail

SKILL_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV="$SKILL_DIR/.venv"
VENV_PYTHON="$VENV/bin/python"
REQUESTED_PYTHON=""
STAGE="checking arguments"

usage() {
    cat <<'HELP'
Usage: setup.sh [--python /path/to/python]

Requires Python 3.12 or 3.13 (3.12 recommended).
Installs all image processing and CPU cutout dependencies, downloads
U2-Net (~176 MB), and verifies a real CPU inference. Network access is
required for packages and the first model download.

An existing .venv is reused and checked, never cleared automatically.
HELP
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

trap 'printf "\nSetup failed while %s. Fix the error above and rerun this command.\n" "$STAGE" >&2' ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 && -n "$2" ]] || fail "--python requires a Python executable"
            REQUESTED_PYTHON="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done

supported_python() {
    "$1" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1
}

if [[ -n "$REQUESTED_PYTHON" ]]; then
    supported_python "$REQUESTED_PYTHON" || fail "--python must run Python 3.12 or 3.13: $REQUESTED_PYTHON"
fi

STAGE="checking the virtual environment"
if [[ -e "$VENV" || -L "$VENV" ]]; then
    [[ -f "$VENV/pyvenv.cfg" && -x "$VENV_PYTHON" ]] || fail "Invalid .venv: $VENV. Move it aside, then rerun setup."
    supported_python "$VENV_PYTHON" || fail "Existing .venv needs Python 3.12 or 3.13. Move it aside, then rerun setup."
    "$VENV_PYTHON" -c 'import pathlib, sys; sys.exit(0 if sys.prefix != sys.base_prefix and pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve() else 1)' "$VENV" || fail "Existing .venv does not resolve to $VENV. Move it aside, then rerun setup."
    if [[ -n "$REQUESTED_PYTHON" ]]; then
        requested_base="$("$REQUESTED_PYTHON" -c 'import os, sys; print(os.path.realpath(sys.base_prefix))')"
        existing_base="$("$VENV_PYTHON" -c 'import os, sys; print(os.path.realpath(sys.base_prefix))')"
        [[ "$requested_base" == "$existing_base" ]] || fail "Existing .venv uses a different Python installation. Move it aside before changing --python."
    fi
    printf 'Reusing %s\n' "$VENV"
else
    selected_python="$REQUESTED_PYTHON"
    if [[ -z "$selected_python" ]]; then
        for candidate in python3.12 python3.13 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && supported_python "$candidate"; then
                selected_python="$candidate"
                break
            fi
        done
    fi
    [[ -n "$selected_python" ]] || fail "No supported Python found. Install Python 3.12 (recommended) or 3.13, then pass --python /path/to/python. Python 3.14 is not supported by this dependency set."
    STAGE="creating .venv"
    "$selected_python" -c 'import ensurepip, venv' || fail "This Python needs venv/ensurepip support before setup can continue."
    "$selected_python" -m venv "$VENV"
fi

"$VENV_PYTHON" -c 'import platform, sys; print(f"Python {platform.python_version()} / {platform.system()} {platform.machine()} / {sys.executable}")'
STAGE="preparing pip"
if ! "$VENV_PYTHON" -m pip --version; then
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

# Keep package-index/proxy settings, but reject configuration that redirects writes.
STAGE="checking pip installation settings"
"$VENV_PYTHON" -m pip config list | "$VENV_PYTHON" -c '
import sys
blocked = []
location_keys = {f"{section}.{option}" for section in ("global", "install", ":env:")
                 for option in ("target", "prefix", "root")}
for line in sys.stdin:
    key, separator, value = line.strip().partition("=")
    if (separator and key in location_keys
            and value not in ("", "\"\"", "\x27\x27")):
        blocked.append(key)
if blocked:
    sys.exit("Error: pip settings may redirect installation outside .venv: " + ", ".join(blocked)
             + ". Remove these settings for setup, then rerun. Their values are not displayed.")
'

STAGE="installing all image dependencies (binary wheels only)"
"$VENV_PYTHON" -m pip install --no-user --disable-pip-version-check --only-binary=:all: -r "$SKILL_DIR/requirements.txt"
STAGE="checking dependency consistency"
"$VENV_PYTHON" -m pip check
STAGE="verifying imports, the U2-Net model and CPU inference"
"$VENV_PYTHON" "$SKILL_DIR/scripts/check_env.py"

printf '\noil-icon environment ready: %s\n' "$VENV"
printf 'Slice with: "%s" "%s/scripts/slice_icons.py" sheet.png out\n' "$VENV_PYTHON" "$SKILL_DIR"
printf 'Agent selects cutout from actual image edges; use the default for separable hard edges and compare rembg with the source and colour candidate when needed.\n'
