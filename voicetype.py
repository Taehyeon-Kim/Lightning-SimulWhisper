#!/usr/bin/env python3
"""
VoiceType - macOS menubar STT app using Lightning-SimulWhisper engine.

Captures microphone audio, runs real-time speech-to-text via SimulWhisper
(MLX + AlignAtt), and types recognized text into the currently focused
input field using CGEvent keyboard simulation.

Usage:
    python voicetype.py [--model_name small] [--lan ko]

Toggle: Press Fn key to start/stop voice recognition.
"""

import sys
import os
import time
import queue
import threading
import logging
import signal
from argparse import Namespace

import numpy as np
import pyaudio
import rumps

from pynput import keyboard as pynput_keyboard

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    kCGHIDEventTap,
    kCGEventKeyDown,
    kCGEventKeyUp,
)

# -- Project imports (Lightning-SimulWhisper engine) --
from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline
from whisper_streaming.vac_online_processor import VACOnlineASRProcessor

logger = logging.getLogger("voicetype")

# ---------------------------------------------------------------------------
# STT Engine – thin wrapper around SimulWhisper
# ---------------------------------------------------------------------------

class STTEngine:
    """Wraps SimulWhisperASR + VACOnlineASRProcessor."""

    SAMPLING_RATE = 16000

    def __init__(self, model_name="small", language="ko", min_chunk_size=1.0):
        logger.info(f"Loading model '{model_name}' for language '{language}' ...")

        # model_path=model_name triggers auto-download from HuggingFace
        # e.g. "small" → mlx-community/whisper-small-mlx
        self.asr = SimulWhisperASR(
            language=language,
            model_path=model_name,
            model_name=model_name,
            cif_ckpt_path=None,
            frame_threshold=25,
            audio_max_len=30.0,
            audio_min_len=0.0,
            segment_length=min_chunk_size,
            beams=1,
            task="transcribe",
            decoder_type="greedy",
            never_fire=True,       # don't truncate last word
            init_prompt=None,
            static_init_prompt=None,
            max_context_tokens=None,
            logdir=None,
            vad_silence_ms=500,
        )

        base_online = SimulWhisperOnline(self.asr)

        # Wrap with VAC for automatic voice activity detection
        self.online = VACOnlineASRProcessor(
            online_chunk_size=min_chunk_size,
            online=base_online,
            vad_silence_ms=500,
        )

        # Warmup with a short silence to speed up first real inference
        warmup_audio = np.zeros(self.SAMPLING_RATE, dtype=np.float32)
        self.asr.warmup(warmup_audio)
        logger.info("Model loaded and warmed up.")

    def feed(self, audio_chunk: np.ndarray) -> tuple[str, bool]:
        """Feed audio and return (recognized_text, is_segment_final).

        When VAD detects end-of-speech, is_segment_final=True means
        this text completes a segment and the next text will be a new one.
        """
        self.online.insert_audio_chunk(audio_chunk)
        # VAC sets is_currently_final in insert_audio_chunk when end-of-speech
        # is detected. process_iter() will then call finish() and reset it.
        will_finalize = self.online.is_currently_final
        _beg, _end, text = self.online.process_iter()
        return (text if text else "", will_finalize)

    def finish(self) -> str:
        """Flush remaining audio and return final text."""
        _beg, _end, text = self.online.finish()
        return text if text else ""

    def reset(self):
        """Reset for a new utterance."""
        self.online.init()


# ---------------------------------------------------------------------------
# Microphone Capture
# ---------------------------------------------------------------------------

class MicCapture:
    """PyAudio microphone capture with callback → queue."""

    RATE = 16000
    CHANNELS = 1
    CHUNK_SEC = 0.04  # 40 ms

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self.queue: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        self.queue.put(audio)
        return (None, pyaudio.paContinue)

    def start(self):
        chunk_frames = int(self.RATE * self.CHUNK_SEC)
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=chunk_frames,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self):
        if self._stream and self._stream.is_active():
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        # drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def destroy(self):
        self.stop()
        self._pa.terminate()


# ---------------------------------------------------------------------------
# Keyboard Injector – CGEvent unicode typing
# ---------------------------------------------------------------------------

class KeyboardInjector:
    """Types text into the active input field via CGEvent keyboard simulation."""

    BACKSPACE_KEYCODE = 51

    def __init__(self):
        self._typed_text = ""

    def type_incremental(self, new_full_text: str):
        """Compare with previously typed text and only type the diff.

        If the new text shares a common prefix with previously typed text,
        delete the divergent tail and type the new suffix (same-segment update).
        If there's no meaningful common prefix, treat it as a new segment
        and just append the new text after what's already on screen.
        """
        if not new_full_text:
            return

        prev = self._typed_text

        # Find common prefix length
        common_len = 0
        for i in range(min(len(prev), len(new_full_text))):
            if prev[i] == new_full_text[i]:
                common_len = i + 1
            else:
                break

        # Heuristic: if less than 30% of the previous text is a common prefix
        # and we'd need to delete a lot, this is likely a new segment.
        # Just append the new text instead of backspacing.
        if prev and common_len < len(prev) * 0.3 and len(prev) - common_len > 3:
            # New segment — append with a space separator
            self._type_unicode(new_full_text)
            self._typed_text = prev + new_full_text
            return

        # Same segment — update incrementally
        delete_count = len(prev) - common_len
        if delete_count > 0:
            self._press_backspace(delete_count)

        new_chars = new_full_text[common_len:]
        if new_chars:
            self._type_unicode(new_chars)

        self._typed_text = new_full_text

    def type_final(self, text: str):
        """Type segment-final text, then reset for next segment."""
        self.type_incremental(text)
        self._typed_text = ""

    def reset(self):
        self._typed_text = ""

    # -- low level --

    def _type_unicode(self, text: str):
        """Type a unicode string via CGEvent."""
        # CGEventKeyboardSetUnicodeString can handle up to ~20 chars at once;
        # we batch in chunks of 16 to be safe.
        BATCH = 16
        for i in range(0, len(text), BATCH):
            chunk = text[i : i + BATCH]
            event_down = CGEventCreateKeyboardEvent(None, 0, True)
            CGEventKeyboardSetUnicodeString(event_down, len(chunk), chunk)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateKeyboardEvent(None, 0, False)
            CGEventKeyboardSetUnicodeString(event_up, len(chunk), chunk)
            CGEventPost(kCGHIDEventTap, event_up)

            time.sleep(0.005)

    def _press_backspace(self, count: int):
        for _ in range(count):
            ev_down = CGEventCreateKeyboardEvent(None, self.BACKSPACE_KEYCODE, True)
            CGEventPost(kCGHIDEventTap, ev_down)
            ev_up = CGEventCreateKeyboardEvent(None, self.BACKSPACE_KEYCODE, False)
            CGEventPost(kCGHIDEventTap, ev_up)
            time.sleep(0.003)


# ---------------------------------------------------------------------------
# VoiceType menubar app
# ---------------------------------------------------------------------------

class VoiceTypeApp(rumps.App):
    """macOS menubar app for real-time STT → keyboard injection."""

    TITLE_IDLE = "[STT Off]"
    TITLE_LISTENING = "[STT On]"

    def __init__(self, model_name="small", language="ko"):
        super().__init__(self.TITLE_IDLE, quit_button="Quit")

        self._model_name = model_name
        self._language = language

        self._engine: STTEngine | None = None
        self._mic = MicCapture()
        self._kb = KeyboardInjector()

        self._recording = False
        self._rec_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Menu items
        self.toggle_item = rumps.MenuItem("Start (⌥)", callback=self._toggle)
        self.status_item = rumps.MenuItem("Status: idle")
        self.status_item.set_callback(None)
        self.menu = [self.toggle_item, self.status_item]

        # Global hotkey listener (Option key alone)
        self._hotkey_listener = None
        self._start_hotkey_listener()

    # -- hotkey --

    def _start_hotkey_listener(self):
        """Listen for Option (⌥) key tap to toggle STT.

        Toggle fires on key release, only if no other key was pressed
        while Option was held (so Option+C etc. won't trigger it).
        """
        self._opt_pressed = False
        self._opt_combo = False  # True if another key was pressed while opt held

        def on_press(key):
            try:
                if key in (pynput_keyboard.Key.alt,
                           pynput_keyboard.Key.alt_l,
                           pynput_keyboard.Key.alt_r):
                    self._opt_pressed = True
                    self._opt_combo = False
                elif self._opt_pressed:
                    # Another key while opt held → not a solo tap
                    self._opt_combo = True
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (pynput_keyboard.Key.alt,
                           pynput_keyboard.Key.alt_l,
                           pynput_keyboard.Key.alt_r):
                    if self._opt_pressed and not self._opt_combo:
                        self._toggle(None)
                    self._opt_pressed = False
                    self._opt_combo = False
            except Exception:
                pass

        self._hotkey_listener = pynput_keyboard.Listener(
            on_press=on_press, on_release=on_release,
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    # -- toggle --

    def _toggle(self, sender):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self._recording:
            return

        # Lazy-load engine on first use
        if self._engine is None:
            self.title = "[Loading...]"
            self.status_item.title = "Status: loading model..."
            # Load in background to not block the main thread
            threading.Thread(target=self._load_and_start, daemon=True).start()
            return

        self._recording = True
        self._stop_event.clear()
        self._kb.reset()
        self._engine.reset()
        self._mic.start()

        self.title = self.TITLE_LISTENING
        self.toggle_item.title = "Stop (⌥)"
        self.status_item.title = "Status: listening..."

        self._rec_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._rec_thread.start()

    @staticmethod
    def _notify(title, subtitle, message):
        """Send macOS notification, silently ignoring failures."""
        try:
            rumps.notification(title, subtitle, message)
        except Exception:
            pass

    def _load_and_start(self):
        try:
            self._engine = STTEngine(
                model_name=self._model_name,
                language=self._language,
            )
            self._notify(
                "VoiceType",
                "Model loaded",
                f"Model '{self._model_name}' ready. Press ⌥ to start.",
            )
            self.title = self.TITLE_IDLE
            self.status_item.title = "Status: idle (model loaded)"
            # Now actually start
            self._start_recording()
        except Exception as e:
            logger.exception("Failed to load model")
            self._notify("VoiceType", "Error", str(e))
            self.title = self.TITLE_IDLE
            self.status_item.title = f"Status: error - {e}"

    def _stop_recording(self):
        if not self._recording:
            return

        self._recording = False
        self._stop_event.set()
        self._mic.stop()

        # Flush remaining text
        if self._engine:
            final_text = self._engine.finish()
            if final_text:
                self._kb.type_final(final_text)
            else:
                self._kb.reset()

        self.title = self.TITLE_IDLE
        self.toggle_item.title = "Start (⌥)"
        self.status_item.title = "Status: idle"

    # -- recognition loop (runs in background thread) --

    def _recognition_loop(self):
        segment_committed = False
        while not self._stop_event.is_set():
            try:
                audio_chunk = self._mic.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                text, is_final = self._engine.feed(audio_chunk)
            except Exception:
                logger.exception("STT feed error")
                continue

            if text:
                logger.info(f"STT: \"{text}\" (final={is_final})")
                try:
                    if is_final:
                        self._kb.type_final(text)
                        segment_committed = True
                    else:
                        if segment_committed:
                            self._kb.reset()
                            segment_committed = False
                        self._kb.type_incremental(text)
                except Exception:
                    logger.exception("Keyboard inject error")

    # -- cleanup --

    @rumps.events.before_quit
    def _cleanup(self):
        self._stop_recording()
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self._mic.destroy()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VoiceType – STT to keyboard")
    parser.add_argument("--model_name", type=str, default="small",
                        help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--lan", type=str, default="ko",
                        help="Language code (ko, en, ja, zh, auto, ...)")
    parser.add_argument("--log_level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = VoiceTypeApp(model_name=args.model_name, language=args.lan)
    app.run()


if __name__ == "__main__":
    main()
