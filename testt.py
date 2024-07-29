from paddlespeech.cli.asr.infer import ASRExecutor

asr = ASRExecutor()
result = asr(audio_file=r"E:\Events\TTS\PaddleSpeech\zh.wav")
print(result)
