import onnxruntime as ort
import numpy as np

# This is a rewrite of silero-vad's VADIterator to use onnxruntime instead of PyTorch.
# The logic is based on the original Python implementation and the C++ example
# for the ONNX model: https://github.com/snakers4/silero-vad/blob/master/silero_cpp_example/silero-vad-onnx.cpp

class VADIterator:
    def __init__(self,
                 model_path,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 100,
                 windows_frame_size_ms: int = 32
                 ):

        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        # ONNX session
        self.session = ort.InferenceSession(model_path)

        self.sr_per_ms = int(sampling_rate / 1000)
        self.window_size_samples = windows_frame_size_ms * self.sr_per_ms
        self.context_samples = 64 # based on C++ example
        
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((self.context_samples,), dtype=np.float32)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        if not isinstance(x, np.ndarray):
            try:
                x = np.array(x, dtype=np.float32)
            except:
                raise TypeError("Audio cannot be casted to numpy array. Cast it manually")

        if len(x) != self.window_size_samples:
            raise ValueError(f"Input chunk size must be {self.window_size_samples}, but got {len(x)}")

        window_size_samples = len(x)
        self.current_sample += window_size_samples

        # Prepare input for ONNX
        input_data = np.concatenate([self._context, x])
        input_tensor = input_data.reshape(1, -1)

        ort_inputs = {
            'input': input_tensor.astype(np.float32),
            'state': self._state,
            'sr': np.array([self.sampling_rate], dtype=np.int64)
        }

        # Run inference
        ort_outputs = self.session.run(None, ort_inputs)
        speech_prob = ort_outputs[0][0]
        self._state = ort_outputs[1]

        # Update context
        self._context = input_data[-self.context_samples:]

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                print(f"VAD: End of speech detected at sample {self.temp_end}")
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None

#######################
# because Silero now requires exactly 512-sized audio chunks 

import numpy as np
class FixedVADIterator(VADIterator):
    '''It fixes VADIterator by allowing to process any audio length, not only exactly 512 frames at once.
    If audio to be processed at once is long and multiple voiced segments detected, 
    then __call__ returns the start of the first segment, and end (or middle, which means no end) of the last segment. 
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([],dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x) 
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None
