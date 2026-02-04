# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for VoiceType menubar app.

Usage:
    pyinstaller VoiceType.spec
"""

import glob
import os
import sys

site_packages = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
)


def _collect_mlx():
    """Collect MLX native libs (.dylib, .metallib, .so) as binaries."""
    mlx_dir = os.path.join(site_packages, "mlx")
    bins = []
    if os.path.isdir(mlx_dir):
        for pattern in ["lib/*.dylib", "lib/*.metallib"]:
            for f in glob.glob(os.path.join(mlx_dir, pattern)):
                bins.append((f, "mlx/lib"))
    return bins


def _collect_onnxruntime():
    ort_capi = os.path.join(site_packages, "onnxruntime", "capi")
    bins = []
    if os.path.isdir(ort_capi):
        for f in glob.glob(os.path.join(ort_capi, "*.dylib")):
            bins.append((f, "onnxruntime/capi"))
    return bins


def _find_portaudio():
    for base in ["/opt/homebrew", "/usr/local"]:
        pa = os.path.join(base, "lib", "libportaudio.2.dylib")
        if os.path.isfile(pa):
            return [(os.path.realpath(pa), ".")]
    return []


extra_binaries = _collect_mlx() + _collect_onnxruntime() + _find_portaudio()

a = Analysis(
    ["voicetype.py"],
    pathex=["."],
    binaries=extra_binaries,
    datas=[
        ("silero_model/silero_vad.onnx", "silero_model"),
        ("simul_whisper", "simul_whisper"),
        ("whisper_streaming", "whisper_streaming"),
        ("token_buffer.py", "."),
        ("simulstreaming_whisper.py", "."),
    ],
    hiddenimports=[
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx.nn.layers",
        "mlx.optimizers",
        "mlx.utils",
        "rumps",
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._darwin",
        "pyaudio",
        "_portaudio",
        "tiktoken",
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
        "huggingface_hub",
        "onnxruntime",
        "numpy",
        "Quartz",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PIL", "pytest", "ruff"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceType",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceType",
)

app = BUNDLE(
    coll,
    name="VoiceType.app",
    bundle_identifier="io.altalt.voicetype",
    info_plist={
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "CFBundleName": "VoiceType",
        "CFBundleDisplayName": "VoiceType",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "NSMicrophoneUsageDescription": "VoiceType needs microphone access for speech-to-text.",
        "NSAccessibilityUsageDescription": (
            "VoiceType needs accessibility permission to type recognized text "
            "and listen for the Option key hotkey."
        ),
    },
)
