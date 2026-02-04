import logging
import time

import numpy as np

from whisper_streaming.base import OnlineProcessorInterface
from whisper_streaming.silero_vad_iterator import FixedVADIterator

logger = logging.getLogger(__name__)


class VACOnlineASRProcessor(OnlineProcessorInterface):
    """Wraps OnlineASRProcessor with VAC (Voice Activity Controller).

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds),
    it runs VAD and continuously detects whether there is speech or not.
    When it detects end of speech (non-voice for configurable duration), it makes OnlineASRProcessor to end the utterance immediately.
    """

    def __init__(self, online_chunk_size, online, vad_silence_ms=500):
        self.online_chunk_size = online_chunk_size

        self.online = online

        # VAC:
        vad_model_path = "silero_model/silero_vad.onnx"
        self.vac = FixedVADIterator(
            vad_model_path, min_silence_duration_ms=vad_silence_ms
        )  # configurable silence duration

        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        t_start = time.time()
        res = self.vac(audio)
        logger.debug(
            f"[PERF] VAC processing: {time.time() - t_start:.4f}s (audio len: {len(audio) / self.SAMPLING_RATE:.3f}s)"
        )
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if res is not None:
            frame = next(iter(res.values()))
            if "start" in res and "end" not in res:
                self.status = "voice"
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=frame / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif "end" in res and "start" not in res:
                self.status = "nonvoice"
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = "nonvoice"
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(res["start"] / self.SAMPLING_RATE))
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == "voice":
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM.
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE :]

    def process_iter(self):
        t_start = time.time()
        if self.is_currently_final:
            ret = self.finish()
            logger.debug(f"[PERF] VACOnlineASRProcessor.process_iter() (final): {time.time() - t_start:.4f}s")
            return ret
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            logger.debug(f"[PERF] VACOnlineASRProcessor.process_iter() (process): {time.time() - t_start:.4f}s")
            return ret
        else:
            logger.debug(f"[PERF] VACOnlineASRProcessor.process_iter() (skip): {time.time() - t_start:.4f}s")
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret
