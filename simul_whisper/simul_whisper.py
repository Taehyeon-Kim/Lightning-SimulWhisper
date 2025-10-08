# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

from mlx.core import array
import os

import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


from .mlx_whisper import load_models, decoding, tokenizer
from .config import AlignAttConfig
from .mlx_whisper.audio import log_mel_spectrogram, TOKENS_PER_SECOND, pad_or_trim, N_SAMPLES, N_FRAMES
from .mlx_whisper.timing import median_filter
from .eow_detection import fire_at_boundary, load_cif
import os

from token_buffer import TokenBuffer

import numpy as np
from .generation_progress import *

DEC_PAD = 50257


import sys
import wave
import logging

logger = logging.getLogger(__name__)

class MLXInference:
    def __init__(self, model, parent):
        self.model = model
        self.parent = parent
        self.kv_cache = None

    def logits(self, tokens, audio_features):
        logits, self.kv_cache, cross_attentions = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )

        cross_attentions = [mx.softmax(cross_attention, axis=-1) for cross_attention in cross_attentions]

        self.parent.dec_attns += cross_attentions
        return logits

    def rearrange_kv_cache(self, source_indices):
        if self.kv_cache and source_indices != list(range(len(source_indices))):
            def rotate_only_if_3dim(x):
                if x.shape[0] == 3:
                    return x[source_indices]
                return x
            
            self.kv_cache = tree_map(rotate_only_if_3dim, self.kv_cache)
            
    def reset(self):
        self.kv_cache = None


# New features added to the original version of Simul-Whisper: 
# - large-v3 model support
# - translation support
# - beam search
# - prompt -- static vs. non-static
# - context
class PaddedAlignAttWhisper:
    def __init__(self, cfg: AlignAttConfig) -> None:
        self.logdir_i = 0
        self.log_segments = 0
        if cfg.logdir is not None and not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        
        self.model = load_models.load_model(path_or_hf_repo=cfg.model_path, dtype=mx.float16, model_name=cfg.model_name)
        
        # Check model dtype
        logger.info(f"Model encoder conv1 weight dtype: {self.model.encoder.conv1.weight.dtype}")

        self.decode_options = decoding.DecodingOptions(
            language = cfg.language, 
            without_timestamps = True,
            task=cfg.task
        )
        self.tokenizer_is_multilingual = self.model.is_multilingual
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.detected_language = cfg.language if cfg.language != "auto" else None
        
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = self.model.dims.n_text_layer
        self.cfg = cfg

        # model to detect end-of-word boundary at the end of the segment
        self.CIFLinear, self.always_fire, self.never_fire = load_cif(cfg,
                                                                     n_audio_state=self.model.dims.n_audio_state)
        self.dec_attns = []
        
        self.align_source = {}
        self.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.tolist():
            layer_rank = int(layer_rank)
            heads = self.align_source.get(layer_rank, [])
            heads.append((self.num_align_heads, int(head_id)))
            self.align_source[layer_rank] = heads
            self.num_align_heads += 1


        # tokens to be suppressed from decoding, to prevent hallucinations
        suppress_tokens = [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
                self.tokenizer.no_timestamps,
            ] + list(self.tokenizer.all_language_tokens)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens =  tuple(sorted(set(suppress_tokens)))
        self.suppress_tokens_filter = decoding.SuppressTokens(suppress_tokens, self.model.dims.n_vocab)

        self.segments = []
        self.init_tokens()
        
        self.last_attend_frame = -self.cfg.rewind_threshold

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens
        self.init_context()

        # decoder type: greedy or beam
        if cfg.decoder_type == "greedy":
            self.token_decoder = decoding.GreedyDecoder(0.0, self.tokenizer.eot)
            self.decoder_type = "greedy"
            self.inference = MLXInference(self.model, self)

        elif cfg.decoder_type == "beam":
            self.decoder_type = "beam"
            self.inference = MLXInference(self.model, self)
            self.token_decoder = decoding.BeamSearchDecoder(
                inference=self.inference, 
                eot=self.tokenizer.eot, 
                beam_size=cfg.beam_size
            )

    def create_tokenizer(self, language=None):
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=self.tokenizer_is_multilingual,  
            language=language,
            num_languages=self.model.num_languages,
            task=self.decode_options.task
        )

    def init_context(self):
        kw = {'tokenizer': self.tokenizer, 
              'prefix_token_ids': [self.tokenizer.sot_prev]}
        self.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.context.text += self.cfg.init_prompt

    def init_tokens(self):
        # init tokens (mandatory prompt)
        self.initial_tokens = mx.array(
            self.tokenizer.sot_sequence_including_notimestamps, 
            dtype=mx.int64)[None, :]
        self.initial_token_length = self.initial_tokens.shape[1]
        self.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
        self.tokens = [self.initial_tokens]

    def trim_context(self):
        c = len(self.context.as_token_ids()) - len(self.context.prefix_token_ids)
        l = sum(t.shape[1] for t in self.tokens) + c
        if self.cfg.static_init_prompt is None:
            after = 0
        else:
            after = len(self.cfg.static_init_prompt)
        while c > self.max_context_tokens or l > self.max_text_len - 20:
            t = self.context.trim_words(after=after)
            l -= t
            c -= t
            if t == 0:
                break


    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        return self.inference.logits(tokens, audio_features)
    

    def refresh_segment(self, complete=False):
        self.init_tokens()
        self.last_attend_frame = -self.cfg.rewind_threshold       
        self.detected_language = None
        self.init_context()
        if not complete and len(self.segments) > 2:
            self.segments = self.segments[-2:]
        else:
            self.segments = []
        self.log_segments += 1


    def fire_at_boundary(self, chunked_encoder_feature: mx.array, force_eval=False):
        if self.always_fire: return True
        if self.never_fire: return False

        return fire_at_boundary(chunked_encoder_feature, self.CIFLinear, force_eval)


    def _current_tokens(self):

        toks = self.tokens
        # very first infer: duplicate start of seq to beam_size
        if toks[0].shape[0] == 1 and self.cfg.beam_size > 1:
            toks[0] = mx.repeat(toks[0], repeats=self.cfg.beam_size, axis=0)

        if not self.context.is_empty():
            if self.cfg.beam_size > 1:
                context_toks = self.context.as_tensor_beam(self.cfg.beam_size)
            else:
                context_toks = self.context.as_tensor()

            toks = [context_toks] + toks

        # make it one tensor
        if len(toks) > 1:
            current_tokens = mx.concatenate(toks, axis=1)
        else:
            current_tokens = toks[0]
        
        return current_tokens




    ### audio buffer 

    def segments_len(self):
        segments_len = sum(s.shape[0] for s in self.segments) / 16000
        return segments_len

    def _apply_minseglen(self):
        segments_len = self.segments_len()
        # wait for long enough audio to start
        if segments_len < self.cfg.audio_min_len: 
            return False
        return True

    def insert_audio(self, segment=None):
        if segment is not None:
            self.segments.append(segment)

        removed_len = 0
        # len of audio is bigger than buffer_len. Going to remove the first segment
        segments_len = self.segments_len()
        while len(self.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.last_attend_frame -= int(TOKENS_PER_SECOND*removed_len)
            self.segments = self.segments[1:]
            if len(self.tokens) > 1:
                tokens_to_add_to_context = self.tokens[1][0,:].tolist()
                self.context.append_token_ids(tokens_to_add_to_context)
                self.tokens = [self.initial_tokens] + self.tokens[2:]
        return removed_len

    def _clean_cache(self):
        '''clean the cache that stores the attention matrices and kv_cache.
        It must be called every time after generation with the model.'''
                # cleaning cache
        self.dec_attns = []
        self.inference.reset()
        self.token_decoder.reset()
        mx.clear_cache()


    ### transcription / translation

    def infer(self, is_last=False):
        import time
        
        # Enable forced evaluation for accurate timing when in DEBUG mode
        FORCE_EVAL = logger.isEnabledFor(logging.DEBUG)
        
        t_start = time.time()
        logger.debug(f"[PERF] infer() started")
        
        new_segment = True
        if len(self.segments) == 0:
            self.logdir_save(mx.array([]), [], {})
            logger.debug(f"[PERF] infer() no segments: {time.time()-t_start:.4f}s")
            return [], {}
        if not self._apply_minseglen():
            input_segments = mx.concatenate(self.segments, axis=0)
            self.logdir_save(input_segments, [], {})
            logger.debug(f"[PERF] infer() min seglen not met: {time.time()-t_start:.4f}s")
            return [], {}

        t_concat = time.time()
        # input_segments is concatenation of audio, it's one array
        if len(self.segments) > 1:
            input_segments = mx.concatenate(self.segments, axis=0)
        else:
            input_segments = self.segments[0]
        if FORCE_EVAL:
            mx.eval(input_segments)
        logger.debug(f"[PERF]   audio concat: {time.time()-t_concat:.4f}s")
        
        t_mel = time.time()
        # mel + padding to 30s
        mel_padded = log_mel_spectrogram(input_segments, n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
        # trim to 3000
        mel = pad_or_trim(mel_padded, N_FRAMES, axis=-2)
        # the len of actual audio
        content_mel_len = int((mel_padded.shape[-2] - mel.shape[-2])/2)
        if FORCE_EVAL:
            mx.eval(mel)
        logger.debug(f"[PERF]   mel spectrogram: {time.time()-t_mel:.4f}s")
        
        t_enc = time.time()
        # encode - convert to float16 for efficiency
        mel_input = mel[None, :, :].astype(mx.float16)
        logger.debug(f"[PERF]     mel dtype before encoder: {mel_input.dtype}, shape: {mel_input.shape}")
        encoder_feature = self.model.encoder(mel_input)
        if FORCE_EVAL:
            mx.eval(encoder_feature)
        logger.debug(f"[PERF]     encoder output dtype: {encoder_feature.dtype}")
        logger.debug(f"[PERF]   encoder: {time.time()-t_enc:.4f}s")
        
        if self.cfg.language == "auto" and self.detected_language is None:
            t_lang = time.time()
            _, language_probs = self.model.detect_language(encoder_feature, self.tokenizer) 
            top_lan, p = max(language_probs[0].items(), key=lambda x: x[1])
            self.create_tokenizer(top_lan)
            self.detected_language = top_lan
            self.init_tokens()
            self._clean_cache()
            logger.debug(f"[PERF]   language detection: {time.time()-t_lang:.4f}s")

        t_prep = time.time()
        t_trim = time.time()
        self.trim_context()
        if FORCE_EVAL:
            mx.eval(self.context.as_token_ids())
        logger.debug(f"[PERF]     trim_context: {time.time()-t_trim:.4f}s")
        
        t_current = time.time()
        current_tokens = self._current_tokens()
        if FORCE_EVAL:
            mx.eval(current_tokens)
        logger.debug(f"[PERF]     _current_tokens: {time.time()-t_current:.4f}s")
        
        t_fire = time.time()
        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :], FORCE_EVAL)
        logger.debug(f"[PERF]     fire_at_boundary (CIF): {time.time()-t_fire:.4f}s")
        
        logger.debug(f"[PERF]   context prep & CIF total: {time.time()-t_prep:.4f}s")


        ####################### Decoding loop
        
        num_beams = self.cfg.beam_size if self.decoder_type == "beam" else 1
        sum_logprobs = mx.zeros(num_beams)
        completed = False

        attn_of_alignment_heads = None
        most_attended_frame = None

        token_len_before_decoding = current_tokens.shape[1]
        t_decode_start = time.time()
        logger.debug(f"[PERF]   starting decoding loop (beams={num_beams})")
        
        generation_progress = []
        generation = {
            "starting_tokens": BeamTokens(current_tokens[0,:], num_beams),
            "token_len_before_decoding": token_len_before_decoding,
            "frames_len": content_mel_len,
            "frames_threshold": 4 if is_last else self.cfg.frame_threshold,
            "logits_starting": None,
            "no_speech_prob": None,
            "no_speech": False,
            "progress": generation_progress,
        }

        decode_step = 0
        while not completed and current_tokens.shape[1] < self.max_text_len:
            t_step = time.time()
            decode_step += 1
            generation_progress_loop = []

            tokens_for_logits = current_tokens if new_segment else current_tokens[:,-1:]
            
            t_logits = time.time()
            logits = self.logits(tokens_for_logits, encoder_feature) # B, len(tokens), token dict size
            if FORCE_EVAL:
                mx.eval(logits)
            logger.debug(f"[PERF]     step {decode_step}: logits computation: {time.time()-t_logits:.4f}s")
            
            if new_segment:
                generation["logits_starting"] = Logits(logits[:,:,:])

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = mx.softmax(logits[:, self.sot_index, :], axis=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                generation["no_speech_prob"] = no_speech_probs[0]
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    generation["no_speech"] = True
                    break

            logits = logits[:, -1, :]
            generation_progress_loop.append(("logits_before_suppress",Logits(logits)))

            if new_segment:
                token_ids_to_suppress = self.tokenizer.encode(" ") + [self.tokenizer.eot]
                mask = np.zeros(logits.shape[-1], dtype=np.float32)
                mask[token_ids_to_suppress] = -np.inf
                logits = logits + mx.array(mask, logits.dtype)

            new_segment = False
            logits = self.suppress_tokens_filter.apply(logits, current_tokens)
            generation_progress_loop.append(("logits_after_suppress",Logits(logits)))
            
            t_decoder_update = time.time()
            current_tokens, completed, sum_logprobs = self.token_decoder.update(current_tokens, logits, sum_logprobs)
            if FORCE_EVAL:
                mx.eval(current_tokens, sum_logprobs)
            logger.debug(f"[PERF]     current_tokens dtype: {current_tokens.dtype}, shape: {current_tokens.shape}")
            logger.debug(f"[PERF]     sum_logprobs dtype: {sum_logprobs.dtype}, shape: {sum_logprobs.shape}")
            logger.debug(f"[PERF]     logits dtype: {logits.dtype}, shape: {logits.shape}")
            logger.debug(f"[PERF]     step {decode_step}: decoder update: {time.time()-t_decoder_update:.4f}s")
            
            # logger.info(f"\n--- decoding step {current_tokens.shape[1] - token_len_before_decoding} ---")
            # for i in range(current_tokens.shape[0]):
            #     beam_tokens = current_tokens[i, token_len_before_decoding:].tolist()
            #     beam_text = self.tokenizer.decode(beam_tokens)
            #     try:
            #         logprob = sum_logprobs[i].item()
            #         logger.info(f"Beam {i}: logprob={logprob:.4f} text='{beam_text}'")
            #     except IndexError:
            #         logger.info(f"Beam {i}: text='{beam_text}' (logprob not available)")

            generation_progress_loop.append(("beam_tokens",Tokens(current_tokens[:,-1])))
            generation_progress_loop.append(("sum_logprobs",sum_logprobs.tolist()))
            generation_progress_loop.append(("completed",bool(completed)))

            t_attn = time.time()
            attn_of_alignment_heads = [[] for _ in range(self.num_align_heads)]
            for i, attn_mat in enumerate(self.dec_attns):
                layer_rank = int(i % len(self.model.decoder.blocks))
                align_heads_in_layer = self.align_source.get(layer_rank, [])
                if len(align_heads_in_layer) == 0:
                    continue
                for align_head_rank, head_id in align_heads_in_layer:
                    if num_beams == 1:
                        # For greedy decoder, attn_mat has shape (1, n_heads, seq_len, audio_len)
                        # Extract: [batch=0, head=head_id, :, :] -> (seq_len, audio_len)
                        # Then add batch dim back: (1, seq_len, audio_len)
                        a = attn_mat[0, head_id, :, :]
                        a = mx.expand_dims(a, 0)
                    else:
                        # For beam search, attn_mat has shape (beam_size, n_heads, seq_len, audio_len)
                        a = attn_mat[:, head_id, :, :]
                    attn_of_alignment_heads[align_head_rank].append(a)
            tmp = []
            for mat in attn_of_alignment_heads:
                t = mx.concatenate(mat, axis=1)
                tmp.append(t) 
            attn_of_alignment_heads = mx.stack(tmp, axis=1)
            
            mean = mx.mean(attn_of_alignment_heads, axis=-2, keepdims=True)
            std = mx.sqrt(mx.var(attn_of_alignment_heads, axis=-2, keepdims=True, ddof=0))

            attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std
            
            attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7)
            
            attn_of_alignment_heads = attn_of_alignment_heads.mean(axis=1)
            attn_of_alignment_heads = attn_of_alignment_heads[:,:, :content_mel_len]

            most_attended_frames = mx.argmax(attn_of_alignment_heads[:,-1,:], axis=-1)
            if FORCE_EVAL:
                mx.eval(most_attended_frames)
            generation_progress_loop.append(("most_attended_frames",most_attended_frames.tolist()))
            logger.debug(f"[PERF]     step {decode_step}: attention processing: {time.time()-t_attn:.4f}s")

            most_attended_frame = most_attended_frames[0].item()

            generation_progress.append(dict(generation_progress_loop))
            
            if completed:
                current_tokens = current_tokens[:, :-1]
                break
            
            if not is_last and self.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                    self.last_attend_frame = most_attended_frame
                else:
                    self.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = mx.concatenate(self.tokens, axis=1) if len(self.tokens) > 0 else self.tokens[0]
                    break
            else:
                self.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (4 if is_last else self.cfg.frame_threshold):
                current_tokens = current_tokens[:, :-1]
                break
            
            for i in range(num_beams):
                pass
            
            logger.debug(f"[PERF]     step {decode_step} total: {time.time()-t_step:.4f}s")

        logger.debug(f"[PERF]   total decoding loop ({decode_step} steps): {time.time()-t_decode_start:.4f}s")
        
        tokens_to_split = current_tokens[0, token_len_before_decoding:]
        if fire_detected or is_last:
            new_hypothesis = tokens_to_split.flatten().tolist()
        else:
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_to_split.tolist())
            generation["result"] = {"split_words": split_words[:-1], "split_tokens": split_tokens[:-1]}
            generation["result_truncated"] = {"split_words": split_words[-1:], "split_tokens": split_tokens[-1:]}

            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]  
            else:
                new_hypothesis = []

        
        new_tokens = mx.array([new_hypothesis], dtype=mx.int64)
        if num_beams > 1:
            new_tokens = mx.repeat(new_tokens, repeats=num_beams, axis=0)

        self.tokens.append(new_tokens)

        
        t_clean = time.time()
        self._clean_cache()
        if FORCE_EVAL:
            mx.eval(mx.array([0]))  # Just to measure the cleanup time
        logger.debug(f"[PERF]   cache cleanup: {time.time()-t_clean:.4f}s")

        self.logdir_save(input_segments, new_hypothesis, generation)
        logger.debug(f"[PERF] infer() total: {time.time()-t_start:.4f}s")
        return new_hypothesis, generation

    def logdir_save(self, input_segments, new_hypothesis, generation):
        """The audio and result from each iteration is saved to the logdir for debugging purposes"""

        if self.cfg.logdir is None:
            return

        self.logdir_i += 1

        dir = os.path.join(self.cfg.logdir, f"seg_{self.log_segments:05d}")
        if not os.path.exists(dir):
            os.makedirs(dir)


        wav_path = os.path.join(dir, f"iter_{self.logdir_i:05d}_audio.wav")
        
        audio_np = np.array(input_segments, copy=False)
        if audio_np.dtype != np.int16:
            audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_np

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        text = self.tokenizer.decode(new_hypothesis)
        with open(os.path.join(dir, f"iter_{self.logdir_i:05d}_hypothesis.txt"), "w") as f:
            if generation:
                context = generation["starting_tokens"].as_text(self.tokenizer)
            else:
                context = ""
            f.write(f"CONTEXT+FORCED:\t{context}\n")
            f.write(f"HYPOTHESIS:\t{text}\n")

