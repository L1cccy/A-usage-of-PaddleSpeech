import asyncio
import os

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools import RemoteToolkit


access_Token = "ed81c466fc077dc5ecf8a84bd4030811b6021be5"
os.environ["EB_AGENT_ACCESS_TOKEN"] = access_Token


async def main():
    llm = ERNIEBot(model="ernie-3.5")  # 初始化大语言模型
    # 这里以语音合成工具为例子，更多的预置工具可参考 https://aistudio.baidu.com/application/center/tool
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()
    agent = FunctionAgent(llm=llm, tools=tts_tool)  # 创建智能体，集成语言模型与工具

    # 与智能体进行通用对话
    result = await agent.run("你好，请自我介绍一下")
    print(f"Agent输出: {result.text}")

    # 请求智能体根据输入文本，自动调用语音合成工具
    result = await agent.run("把上一轮的回答转成语音")
    print(f"Agent输出: {result.text}")

    # 将智能体输出的音频文件写入test1.wav, 可以尝试播放
    audio_file = result.steps[-1].output_files[-1]
    await audio_file.write_contents_to("./test1.wav")

asyncio.run(main())