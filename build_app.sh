#!/bin/bash
set -e

cd "$(dirname "$0")"

rm -rf build dist

pyinstaller VoiceType.spec "$@"

echo ""
echo "Built: dist/VoiceType.app"
echo "Run:   open dist/VoiceType.app"
