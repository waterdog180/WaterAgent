# ask_module.py
# 日期: 2026年1月29日
# 描述: 多Agent协作框架的基础Ask类，初期实现对智谱AI的接入。

import json
import logging
import os
from typing import Dict, Any

# 导入智谱AI的SDK
# [[47]] 中展示了如何使用 ZhipuAI 客户端
import zhipuai
from zhipuai.core._errors import APIRequestFailedError, APIAuthenticationError, APIReachLimitError

# --- 配置日志记录 ---
# 为应用提供基本的日志功能，便于调试和追踪
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Ask:
    """
    一个基础的LLM交互类，用于处理API配置加载、单次消息发送和响应接收。
    这个初始版本专门为智谱AI (Zhipu AI) 设计。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        构造函数，用于初始化Ask类的实例。
        Args:
            config (Dict[str, Any]): 包含API配置的字典。
                                     必需键: 'api_key', 'model'
                                     可选键: 'base_url', 'default_params'
        """
        logging.info("正在初始化 Ask 类...")
        
        if not isinstance(config, dict):
            raise TypeError("配置参数必须是字典类型。")

        self.config = config
        
        # --- 验证配置 ---
        # 确保关键配置项存在，实现“快速失败”原则
        required_keys = ['api_key', 'model']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置文件中缺少必需的键: '{key}'")

        if not self.config['api_key'] or "填入" in self.config['api_key']:
             raise ValueError("API Key 无效，请在 config.json 中配置您的智谱 API Key。")

        # --- 设置核心属性 ---
        # [[48]][[49]]提及了模型的核心参数
        self.model = self.config['model']
        self.default_params = self.config.get('default_params', {})
        
        # --- 初始化API客户端 ---
        self._client = self._initialize_client()
        logging.info(f"Ask 类初始化完成。将使用模型: {self.model}")

    def _initialize_client(self) -> zhipuai.ZhipuAI:
        """
        私有方法，用于根据配置创建智谱AI的客户端实例。
        
        Returns:
            zhipuai.ZhipuAI: 智谱AI的客户端实例。
        """
        try:
            logging.info("正在初始化智谱AI客户端...")
            # 根据智谱SDK文档，使用api_key和base_url初始化客户端
            client = zhipuai.ZhipuAI(
                api_key=self.config['api_key'],
                base_url=self.config.get('base_url') # base_url是可选的，但建议指定
            )
            logging.info("智谱AI客户端初始化成功。")
            return client
        except Exception as e:
            logging.error(f"智谱AI客户端初始化失败: {e}", exc_info=True)
            raise

    def send(self, message: str, system_prompt: str = None, **kwargs) -> str:
        """
        向智谱LLM发送单次消息并获取回复。
        
        Args:
            message (str): 用户发送的消息内容。
            system_prompt (str, optional): 系统提示，用于设定AI角色。默认为None。
            **kwargs: 任何希望在本次调用中覆盖默认参数的键值对，例如 temperature, top_p。
            
        Returns:
            str: LLM返回的回复内容。如果出错则返回错误提示信息。
        """
        logging.info(f"准备向模型发送消息...")
        
        # --- 1. 构建请求体 (messages) ---
        # 遵循 Chat Completion API 的格式 [[50]][[51]]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # --- 2. 合并API参数 ---
        api_params = self.default_params.copy()
        api_params.update(kwargs) # 用户在send时传入的参数优先级更高

        try:
            logging.debug(f"请求参数: model={self.model}, messages={messages}, params={api_params}")
            
            # --- 3. 调用API ---
            # 使用客户端的chat.completions.create方法
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **api_params
            )
            
            # --- 4. 解析和返回响应 ---
            # 根据智谱API的响应结构提取内容
            if response and response.choices:
                content = response.choices[0].message.content
                logging.info("成功从API获取响应。")
                logging.debug(f"原始响应: {response}")
                return content.strip()
            else:
                logging.warning("API响应为空或格式不正确。")
                return "API响应异常，未能获取有效回复。"

        # --- 5. 错误处理 ---
        # 针对性地捕获SDK可能抛出的异常
        except APIAuthenticationError as e:
            logging.error(f"API认证失败: {e}", exc_info=True)
            return f"错误：API认证失败，请检查您的API Key是否正确且有效。"
        except APIReachLimitError as e:
            logging.error(f"API速率限制: {e}", exc_info=True)
            return f"错误：已达到API调用速率上限，请稍后再试。"
        except APIRequestFailedError as e:
            logging.error(f"API请求失败: {e}", exc_info=True)
            return f"错误：API请求失败，请检查请求参数或网络连接。详情: {e}"
        except Exception as e:
            logging.error(f"与LLM交互时发生未知错误: {e}", exc_info=True)
            return f"错误：发生未知异常，请查看日志。详情: {e}"

# --- 主程序入口，用于演示和测试 ---
if __name__ == '__main__':
    CONFIG_FILE_PATH = 'config.json'

    # --- 加载配置文件 ---
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"错误: 找不到配置文件 '{CONFIG_FILE_PATH}'。请确保文件存在并已正确配置。")
    else:
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                # [[53]] 展示了从JSON文件加载配置的用法
                config_data = json.load(f)

            # --- 创建 Ask 实例 ---
            # 将加载的配置字典传递给 Ask 类的构造函数
            ask_instance = Ask(config=config_data)

            # --- 进行单次对话测试 ---
            print("--- 单次对话测试 ---")
            question1 = "你好，请用一句话介绍一下你自己。"
            print(f"  -> 用户: {question1}")
            response1 = ask_instance.send(question1)
            print(f"  <- 模型: {response1}\n")

            # --- 带有系统提示的测试 ---
            print("--- 系统提示测试 ---")
            system_prompt_poet = "你是一位唐代诗人，你的回答必须充满诗意，并以五言绝句的形式呈现。"
            question2 = "描述一下月亮。"
            print(f"  -> 用户: {question2}")
            print(f"  (系统提示: {system_prompt_poet})")
            response2 = ask_instance.send(question2, system_prompt=system_prompt_poet)
            print(f"  <- 模型:\n{response2}\n")

            # --- 覆盖默认参数的测试 ---
            print("--- 覆盖参数测试 (更高temperature) ---")
            question3 = "写一个关于太空旅行的奇幻短故事开头。"
            print(f"  -> 用户: {question3}")
            # 传入 temperature=0.99 以获得更有创造力的回答
            response3 = ask_instance.send(question3, temperature=0.99)
            print(f"  <- 模型: {response3}\n")

        except json.JSONDecodeError:
            print(f"错误: 配置文件 '{CONFIG_FILE_PATH}' 格式不正确，请检查是否为有效的JSON。")
        except (ValueError, TypeError) as e:
            print(f"错误: {e}")

