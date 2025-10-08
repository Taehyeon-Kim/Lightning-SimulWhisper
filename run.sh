python simulstreaming_whisper_server.py \
  --language ko \
  --host localhost \
  --port 43001 \
  --warmup-file warmup.mp3 \
  --vac \
  --beams 3 \
  -l CRITICAL \
  --model_path small.pt \
  --cif_ckpt_path cif_model/small.pt


python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --beams 3 \
  -l CRITICAL \
  --model_path small.pt \
  --cif_ckpt_path cif_model/small.pt


python simulstreaming_whisper.py test.mp3 --language ko --vac --beams 3 -l INFO --model_path mlx_medium --cif_ckpt_path cif_model/medium.npz --audio_min_len 1.0

python simulstreaming_whisper_server.py --language ko --host localhost --port 43001 --warmup-file warmup.mp3 --vac --beams 3 -l INFO --model_path mlx_medium --cif_ckpt_path cif_model/medium.npz --audio_min_len 1.0  

ffmpeg -f avfoundation -i ":2" -ac 1 -ar 16000 -f s16le -c:a pcm_s16le - | nc localhost 43001           


 python simulstreaming_whisper.py test.mp3 --language ko --vac --beams 3 -l DEBUG --model_path mlx_medium --cif_ckpt_path cif_model/medium.npz --audio_min_len 1.0

python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --vad_silence_ms 1000 \
  --beams 3 \
  -l CRITICAL \
  --cif_ckpt_path cif_model/medium.npz \
  --model_name medium \
  --model_path mlx_medium
