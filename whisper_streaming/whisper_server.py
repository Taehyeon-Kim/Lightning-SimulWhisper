#!/usr/bin/env python3
import argparse
import io
import logging
import os
import socket
import sys
import time

import librosa
import numpy as np
import soundfile

import whisper_streaming.line_packet as line_packet
from whisper_streaming.whisper_online_main import (
    asr_factory,
    load_audio_chunk,
    processor_args,
    set_logging,
)

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


######### Server objects


class Connection:
    """it wraps conn object"""

    PACKET_SIZE = 32000 * 5 * 60  # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        """it doesn't send the same line twice, because it was problematic in online-text-flow-events"""
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


# wraps socket and ASR object, and serves one client connection.
# next client should be served by a new instance of this object
class ServerProcessor:
    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk * SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(
                io.BytesIO(raw_bytes),
                channels=1,
                endian="LITTLE",
                samplerate=SAMPLING_RATE,
                subtype="PCM_16",
                format="RAW",
            )
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self, o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print(f"{beg:1.0f} {end:1.0f} {o[2]}", flush=True, file=sys.stderr)
            return f"{beg:1.0f} {end:1.0f} {o[2]}"
        else:
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        self.online_asr_proc.init()
        iteration = 0
        while True:
            iteration += 1
            t_iter = time.time()
            logger.debug(f"\n[PERF] ========== Iteration {iteration} ==========")

            t_receive = time.time()
            a = self.receive_audio_chunk()
            if a is None:
                break
            logger.debug(
                f"[PERF] Received audio chunk: {time.time() - t_receive:.4f}s (audio len: {len(a) / SAMPLING_RATE:.3f}s)"
            )

            t_insert = time.time()
            self.online_asr_proc.insert_audio_chunk(a)
            logger.debug(f"[PERF] Insert audio chunk: {time.time() - t_insert:.4f}s")

            t_process = time.time()
            o = self.online_asr_proc.process_iter()
            logger.debug(f"[PERF] Process iter: {time.time() - t_process:.4f}s")

            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

            logger.debug(f"[PERF] Total iteration time: {time.time() - t_iter:.4f}s")


def main_server(factory, add_args):
    """
    factory: function that creates the ASR and online processor object from args and logger.
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    """
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument(
        "--warmup-file",
        type=str,
        dest="warmup_file",
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. "
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
    )

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args, logger)

    # setting whisper object by args

    asr, online = asr_factory(args, factory)
    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size

    # warm up the ASR because the very first transcribe takes more time than the others.
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.warmup(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # server loop

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info("Listening on" + str((args.host, args.port)))
        while True:
            conn, addr = s.accept()
            logger.info(f"Connected to client on {addr}")
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, min_chunk)
            proc.process()
            conn.close()
            logger.info("Connection to client closed")
    logger.info("Connection closed, terminating.")
