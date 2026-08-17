#!/usr/bin/env bash
#
# build_macos.sh -- build the Biophysics GUI into a macOS .app and .dmg.
#
# Usage:
#   ./build_macos.sh
#
# Produces:
#   dist/Bedoya-Kogan.app   -- the application bundle
#   dist/Bedoya-Kogan.dmg   -- a drag-to-install disk image
#
# Requires: Python 3.9+ and Xcode command line tools (for hdiutil/iconutil).
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="Bedoya-Kogan"
VENV=".venv"

echo "==> Setting up virtual environment ($VENV)"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "==> (Re)generating Windows icon is skipped on macOS; using TransparentApp.icns"

echo "==> Building with PyInstaller"
rm -rf build dist
pyinstaller Project2025App.spec --noconfirm --clean

echo "==> Creating DMG installer"
STAGE="$(mktemp -d)/dmg"
mkdir -p "$STAGE"
cp -R "dist/${APP_NAME}.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "dist/${APP_NAME}.dmg"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "dist/${APP_NAME}.dmg"
rm -rf "$STAGE"

echo ""
echo "==> Done."
echo "    App:  dist/${APP_NAME}.app"
echo "    DMG:  dist/${APP_NAME}.dmg"
echo ""
echo "    Note: the build is unsigned. First launch requires right-click > Open"
echo "    (or: System Settings > Privacy & Security > Open Anyway)."
