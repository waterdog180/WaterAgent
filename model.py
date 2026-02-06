import asyncio
import json
import random
import uuid
from typing import Any, Dict, List

class ModelPool:
    """
    基于面向对象的异步模型池管理类（含唯一 ID 机制）。

    核心功能:
    - 读取 JSON 配置文件并为每个配置生成唯一的序号 ID。
    - 对内聚合管理模型并发数，对外屏蔽模型差异。
    - 提供无参数的异步申请接口 `acquire_config`，返回包含 `config_id` 的配置字典。
    - 提供简化的异步释放接口 `release_config_by_id`，仅需传入 `config_id`。
    """

    def __init__(self, config_path: str):
        """
        初始化模型池并加载配置。

        Args:
            config_path: JSON 配置文件的本地路径。

        Raises:
            FileNotFoundError: 配置文件不存在。
            ValueError: JSON 格式错误或缺少必要字段。
        """
        # 私有属性
        self._model_configs: Dict[str, List[Dict[str, Any]]] = {}
        self._model_max_concurrency: Dict[str, int] = {}
        self._model_available_concurrency: Dict[str, int] = {}
        self._id_to_model: Dict[str, str] = {}  # ID 到模型的映射
        self._condition = asyncio.Condition()

        self._load_and_process_config(config_path)

    def _load_and_process_config(self, config_path: str) -> None:
        """
        加载 JSON 文件并为每个配置生成唯一 ID。
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"无法解析JSON文件，请检查格式: {config_path}")

        if not isinstance(configs, list):
            raise ValueError("配置文件根结构必须是JSON数组")

        for config in configs:
            if not isinstance(config, dict):
                raise ValueError("配置文件数组中的元素必须是JSON对象")
                
            model = config.get("model")
            max_concy = config.get("max_concy")

            # 核心字段校验
            if not isinstance(model, str) or not model:
                raise ValueError(f"配置项缺少或无效的 'model' 字段: {config}")
            if not isinstance(max_concy, int) or max_concy < 0:
                raise ValueError(f"配置项 'max_concy' 必须为非负整数: {config}")
            
            # 为配置生成唯一 ID（使用 UUID 确保全局唯一性）
            unique_id = str(uuid.uuid4())
            config["config_id"] = unique_id

            # 建立 ID 与模型的映射关系
            self._id_to_model[unique_id] = model

            # 按 model 维度聚合数据
            self._model_configs.setdefault(model, []).append(config)
            self._model_max_concurrency[model] = self._model_max_concurrency.get(model, 0) + max_concy
            self._model_available_concurrency[model] = self._model_available_concurrency.get(model, 0) + max_concy

    def _get_total_available_concurrency(self) -> int:
        """计算当前池内所有模型的总可用并发数。"""
        return sum(self._model_available_concurrency.values())

    async def acquire_config(self) -> Dict[str, Any]:
        """
        异步申请一个无指定的API配置。

        Returns:
            包含唯一 `config_id` 的完整API配置字典。
        """
        async with self._condition:
            # 当所有模型并发耗尽时，异步阻塞等待
            while self._get_total_available_concurrency() == 0:
                print(f"[{asyncio.current_task().get_name()}] 池内无可用并发，进入阻塞等待...")
                await self._condition.wait()
            
            # 简单轮询策略：找到第一个有可用并发的模型
            selected_model = None
            for model, available in self._model_available_concurrency.items():
                if available > 0:
                    selected_model = model
                    break
            
            if selected_model is None:
                raise RuntimeError("内部状态错误：并发数为0但未进入阻塞。")

            # 扣减并发数
            self._model_available_concurrency[selected_model] -= 1
            
            # 随机返回该模型下的一个配置
            config_to_return = random.choice(self._model_configs[selected_model])
            
            print(f"[{asyncio.current_task().get_name()}] 成功申请到配置，模型: {selected_model}。 "
                  f"池内总可用并发: {self._get_total_available_concurrency()}")
            
            return config_to_return

    async def release_config_by_id(self, config_id: str) -> None:
        """
        异步释放一个配置的并发数（基于唯一 ID）。

        Args:
            config_id: acquire_config 返回的配置中的唯一标识符。

        Raises:
            ValueError: ID 格式错误、ID 不存在或释放操作导致并发数溢出。
        """
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("释放配置时，config_id 必须是非空字符串。")
        
        async with self._condition:
            # 校验 ID 是否存在
            if config_id not in self._id_to_model:
                raise ValueError(f"提供的 config_id '{config_id}' 未在池中注册。")
            
            model = self._id_to_model[config_id]

            # 检查是否已达到最大并发数，防止重复释放
            if self._model_available_concurrency[model] >= self._model_max_concurrency[model]:
                raise ValueError(f"模型 '{model}' 的并发数已满，无法释放更多。")
            
            # 恢复可用并发数
            self._model_available_concurrency[model] += 1
            
            print(f"[{asyncio.current_task().get_name()}] 成功释放配置，模型: {model} (ID: {config_id})。 "
                  f"池内总可用并发: {self._get_total_available_concurrency()}")
            
            # 唤醒等待任务
            self._condition.notify()


# ==================== 测试用例 ====================

async def main():
    # 1. 准备测试配置文件
    config_data = [
        {
            "provider": "zhipu",
            "api_key": "zhipu_key_1",
            "model": "glm-4-flash",
            "max_concy": 1
        },
        {
            "provider": "zhipu",
            "api_key": "zhipu_key_2",
            "model": "glm-4-flash",
            "max_concy": 1
        },
        {
            "provider": "openai",
            "api_key": "openai_key_1",
            "model": "gpt-4o-mini",
            "max_concy": 2
        }
    ]
    config_path = "temp_model_config_with_id.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 2. 初始化模型池
    print("--- 初始化模型池 ---")
    model_pool = ModelPool(config_path)
    print("模型池初始化完成。")

    # 3. 定义 worker 协程（使用 ID 进行释放）
    async def worker(worker_id: int,t):
        print(f"[Worker-{worker_id}] 开始申请...")
        config = await model_pool.acquire_config()
        print(f"[Worker-{worker_id}] 申请成功，获得 config_id: {config['config_id']}")
        
        # 模拟使用耗时
        await asyncio.sleep(t)
        
        # 使用 ID 进行释放
        print(f"[Worker-{worker_id}] 使用完毕，准备释放...")
        await model_pool.release_config_by_id(config["config_id"])
        print(f"[Worker-{worker_id}] 释放完成。")

    # 4. 启动多个 worker 进行测试（总并发为4，启动6个 worker 测试阻塞）
    tasks = [
        asyncio.create_task(worker(i,i), name=f"Worker-{i}")
        for i in range(1, 7)
    ]
    await asyncio.gather(*tasks)

    print("\n所有任务执行完毕。")

if __name__ == "__main__":
    asyncio.run(main())
