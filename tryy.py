import asyncio
import os
import re
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools import RemoteToolkit

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor

from pathlib import Path
import pygame.mixer
import pyaudio
import wave

# Environment setup
os.environ['EB_AGENT_ACCESS_TOKEN'] = '####'  # 连接文心引擎的access_token，需要自己填写
os.environ['EB_AGENT_LOGGING_LEVEL'] = 'info'


def record_audio(filename="input.wav", record_seconds=5, sample_rate=16000, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


async def main():
    llm = ERNIEBot(model="ernie-3.5")  # 初始化大语言模型
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()  # 获取语音合成工具
    agent = FunctionAgent(llm=llm, tools=tts_tool)  # 创建智能体，集成语言模型与工具

    while True:
        # Record audio
        record_audio()

        # Use ASR to convert audio to text
        asr = ASRExecutor()
        audio_file_path = Path('input.wav')
        content = asr(audio_file=audio_file_path)
        print(content, end='')

        config = input('---结果是否正确？[y/n]')
        if config == 'n':
            print('请手动修改结果:', end='')
            content = input('')
        elif config == 'y':
            pass

        # Get response from ERNIEBot with length restriction
        prompt = f"请用简短的句子回答以下问题：{content}"
        result = await agent.run(prompt)
        print(result.text)

        # Clean special characters
        cleaned_text = re.sub(r'[#&*$\/]+', ' ', result.text)

        # Convert text to speech
        tts = TTSExecutor()
        tts(text=cleaned_text, output="output.wav")

        # Play the generated audio
        pygame.mixer.init()
        pygame.mixer.music.load('output.wav')
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.quit()

        config_continue = input('是否继续提问？[y/n]')
        if config_continue == 'n':
            print('即将退出程序')
            break

asyncio.run(main())
