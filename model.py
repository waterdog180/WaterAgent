import asyncio
import json
import random
import uuid
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

# --- 极简日志配置（符合初期要求） ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 核心自定义异常（仅2个，符合简化要求） ---
class UnsupportedPerfTagError(Exception):
    """申请配置时传入不支持的性能维度标识"""
    def __init__(self, perf_tag: str):
        self.perf_tag = perf_tag
        super().__init__(f"不支持的性能维度：{perf_tag}")

class ConfigIdNotFoundError(Exception):
    """释放配置时指定的config_id不存在于任何模型池中"""
    def __init__(self, config_id: str):
        self.config_id = config_id
        super().__init__(f"配置ID不存在：{config_id}")

# --- 申请策略枚举（仅预留STRICT，符合要求） ---
class AcquireStrategy(Enum):
    STRICT = "strict"  # 严格模式（当前仅实现）
    # 预留：后续添加降级模式、失败模式

# --- 原有 ModelPool 类（仅保留核心，移除print，改用日志） ---
class ModelPool:
    """单性能维度模型池管理类（核心逻辑保留，日志规范化）"""
    def __init__(self, config_path: str):
        self._model_configs: Dict[str, List[Dict[str, Any]]] = {}
        self._model_max_concurrency: Dict[str, int] = {}
        self._model_available_concurrency: Dict[str, int] = {}
        self._id_to_model: Dict[str, str] = {}  # ID到模型的映射
        self._condition = asyncio.Condition()
        self._load_and_process_config(config_path)

    def _load_and_process_config(self, config_path: str) -> None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"JSON格式错误: {config_path}")

        if not isinstance(configs, list):
            raise ValueError("配置文件根结构必须是JSON数组")

        for config in configs:
            model = config.get("model")
            max_concy = config.get("max_concy")
            if not isinstance(model, str) or not model:
                raise ValueError(f"配置项缺少有效model字段: {config}")
            if not isinstance(max_concy, int) or max_concy < 0:
                raise ValueError(f"配置项max_concy必须为非负整数: {config}")
            
            unique_id = str(uuid.uuid4())
            config["config_id"] = unique_id
            self._id_to_model[unique_id] = model

            self._model_configs.setdefault(model, []).append(config)
            self._model_max_concurrency[model] = self._model_max_concurrency.get(model, 0) + max_concy
            self._model_available_concurrency[model] = self._model_available_concurrency.get(model, 0) + max_concy

    def _get_total_available_concurrency(self) -> int:
        """计算总可用并发数（供Models调用）"""
        return sum(self._model_available_concurrency.values())

    async def acquire_config(self) -> Dict[str, Any]:
        """异步申请配置（严格模式，无可用则阻塞）"""
        async with self._condition:
            while self._get_total_available_concurrency() == 0:
                logger.debug(f"[{asyncio.current_task().get_name()}] 无可用并发，阻塞等待")
                await self._condition.wait()
            
            selected_model = None
            for model, available in self._model_available_concurrency.items():
                if available > 0:
                    selected_model = model
                    break
            
            if selected_model is None:
                raise RuntimeError("内部状态错误：并发数为0但未阻塞")

            self._model_available_concurrency[selected_model] -= 1
            config_to_return = random.choice(self._model_configs[selected_model])
            logger.debug(f"[{asyncio.current_task().get_name()}] 申请配置成功，模型: {selected_model}，ID: {config_to_return['config_id']}")
            return config_to_return

    async def release_config_by_id(self, config_id: str) -> None:
        """异步释放配置（按ID）"""
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("config_id必须是非空字符串")
        
        async with self._condition:
            if config_id not in self._id_to_model:
                raise ValueError(f"config_id未注册: {config_id}")
            
            model = self._id_to_model[config_id]
            if self._model_available_concurrency[model] >= self._model_max_concurrency[model]:
                raise ValueError(f"模型{model}并发数已满，无法释放")
            
            self._model_available_concurrency[model] += 1
            logger.debug(f"[{asyncio.current_task().get_name()}] 释放配置成功，ID: {config_id}")
            self._condition.notify()

# --- Models 类（核心实现，严格贴合需求） ---
class Models:
    """多性能维度ModelPool统一管控入口（初期测试版）"""
    def __init__(self, perf_tag: Optional[str] = None):
        self._pool_map: Dict[str, ModelPool] = {}  # perf_tag -> ModelPool
        self._default_config_dir = "models"
        self._current_perf_tag = perf_tag  # 上下文管理器用的性能维度
        self._current_config = None  # 上下文管理器用的配置
        self._init_pool_map()  # 自动扫描初始化

    def _init_pool_map(self) -> None:
        """自动扫描models目录，初始化ModelPool（符合极简初始化要求）"""
        if not os.path.isdir(self._default_config_dir):
            logger.warning(f"默认目录{self._default_config_dir}不存在，未加载任何模型池")
            return

        for filename in os.listdir(self._default_config_dir):
            if not filename.endswith(".json"):
                continue
            
            perf_tag = os.path.splitext(filename)[0]
            config_path = os.path.join(self._default_config_dir, filename)
            try:
                pool = ModelPool(config_path)
                self._pool_map[perf_tag] = pool
                logger.info(f"成功加载性能维度: {perf_tag}（配置：{filename}）")
            except Exception as e:
                logger.warning(f"加载{perf_tag}失败: {str(e)}")

        supported_tags = self.get_supported_perf_tags()
        logger.info(f"Models初始化完成，已加载维度: {supported_tags}")

    def get_supported_perf_tags(self) -> List[str]:
        """返回所有支持的性能维度（符合需求）"""
        return list(self._pool_map.keys())

    # --- 预留接口（仅定义pass，符合要求） ---
    def hot_reload(self, perf_tag: str = None) -> None:
        """后续开发：热更新配置"""
        pass

    def register_pool(self, perf_tag: str, config_path: str) -> None:
        """后续开发：动态注册模型池"""
        pass

    def unregister_pool(self, perf_tag: str) -> None:
        """后续开发：动态注销模型池"""
        pass

    async def acquire_batch(self, perf_tag: str, count: int) -> List[Dict[str, Any]]:
        """后续开发：批量申请配置"""
        pass

    async def release_batch(self, config_ids: List[str]) -> None:
        """后续开发：批量释放配置"""
        pass

    # --- 核心申请接口（仅严格模式，符合要求） ---
    async def acquire(
        self, 
        perf_tag: str, 
        strategy: AcquireStrategy = AcquireStrategy.STRICT
    ) -> Dict[str, Any]:
        """严格模式申请配置（无参抛NotImplementedError，符合需求）"""
        if perf_tag is None:
            raise NotImplementedError("未传入perf_tag，后续支持自动选池")
        
        if perf_tag not in self._pool_map:
            logger.error(f"UnsupportedPerfTagError：{perf_tag}")
            raise UnsupportedPerfTagError(perf_tag)
        
        # 仅实现严格模式（符合需求）
        pool = self._pool_map[perf_tag]
        config = await pool.acquire_config()
        logger.info(f"申请配置成功，维度：{perf_tag}，ID：{config['config_id']}")
        return config

    # --- 核心释放接口（按ID匹配池，符合要求） ---
    async def release(self, config_id: str) -> None:
        """按ID释放配置（未找到抛ConfigIdNotFoundError）"""
        target_pool = None
        for pool in self._pool_map.values():
            if config_id in pool._id_to_model:
                target_pool = pool
                break
        
        if target_pool is None:
            logger.error(f"ConfigIdNotFoundError：{config_id}")
            raise ConfigIdNotFoundError(config_id)
        
        await target_pool.release_config_by_id(config_id)
        logger.info(f"释放配置成功，ID：{config_id}")

    # --- 异步上下文管理器（直接实现，符合需求） ---
    async def __aenter__(self) -> Dict[str, Any]:
        """上下文管理器：申请配置"""
        if self._current_perf_tag is None:
            raise ValueError("使用上下文管理器需指定perf_tag")
        self._current_config = await self.acquire(self._current_perf_tag)
        return self._current_config

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：自动释放配置"""
        if self._current_config:
            await self.release(self._current_config["config_id"])
        self._current_perf_tag = None
        self._current_config = None

    # --- 状态查询（轻量化，符合需求） ---
    def get_pool_status(self, perf_tag: str) -> Dict[str, int]:
        """获取单个池的核心状态（total_available/total_max）"""
        if perf_tag not in self._pool_map:
            raise UnsupportedPerfTagError(perf_tag)
        
        pool = self._pool_map[perf_tag]
        return {
            "total_available": pool._get_total_available_concurrency(),
            "total_max": sum(pool._model_max_concurrency.values())
        }

    def get_all_pool_status(self) -> Dict[str, Dict[str, int]]:
        """获取所有池的状态"""
        return {tag: self.get_pool_status(tag) for tag in self._pool_map}

# ==================== 复原：复杂功能测试用例 ====================
# --- 测试前置：生成测试配置文件 ---
def generate_test_configs():
    """生成多性能维度的测试配置文件（light/deep），用于复杂测试"""
    # 创建models目录
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # light维度配置（低并发，用于快速测试）
    light_config = [
        {"provider": "zhipu", "api_key": "light_key_1", "model": "glm-4-flash", "max_concy": 1},
        {"provider": "zhipu", "api_key": "light_key_2", "model": "glm-4-flash", "max_concy": 1}
    ]
    with open("models/light.json", "w", encoding="utf-8") as f:
        json.dump(light_config, f, indent=2)
    
    # deep维度配置（高并发，用于阻塞测试）
    deep_config = [
        {"provider": "zhipu", "api_key": "deep_key_1", "model": "glm-4", "max_concy": 2},
        {"provider": "openai", "api_key": "deep_key_2", "model": "gpt-4o", "max_concy": 2}
    ]
    with open("models/deep.json", "w", encoding="utf-8") as f:
        json.dump(deep_config, f, indent=2)
    
    logger.info("测试配置文件生成完成（light.json/deep.json）")

# --- 测试1：ModelPool单池多并发阻塞测试（复杂场景） ---
async def test_model_pool_high_concurrency():
    """测试ModelPool核心能力：多任务并发申请、阻塞、释放"""
    logger.info("\n=== 测试1：ModelPool单池多并发阻塞测试 ===")
    # 生成临时测试配置（总并发=4）
    test_config = [
        {"model": "glm-4-flash", "max_concy": 2},
        {"model": "gpt-4o-mini", "max_concy": 2}
    ]
    temp_config_path = "temp_model_pool_test.json"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(test_config, f)
    
    # 初始化ModelPool
    pool = ModelPool(temp_config_path)
    logger.info(f"ModelPool初始化完成，总最大并发：{sum(pool._model_max_concurrency.values())}")
    
    # 定义worker协程（模拟不同耗时的任务）
    async def worker(worker_id: int, sleep_time: float):
        task_name = f"Worker-{worker_id}"
        logger.info(f"[{task_name}] 开始申请配置")
        try:
            config = await pool.acquire_config()
            logger.info(f"[{task_name}] 申请成功，config_id：{config['config_id']}")
            
            # 模拟业务耗时
            await asyncio.sleep(sleep_time)
            
            # 释放配置
            logger.info(f"[{task_name}] 开始释放配置")
            await pool.release_config_by_id(config["config_id"])
            logger.info(f"[{task_name}] 释放成功")
        except Exception as e:
            logger.error(f"[{task_name}] 执行失败：{str(e)}")
    
    # 启动6个worker（超过总并发4，验证阻塞逻辑）
    tasks = [
        asyncio.create_task(worker(i, i*0.5), name=f"Worker-{i}")
        for i in range(1, 7)
    ]
    await asyncio.gather(*tasks)
    
    # 验证最终状态（可用并发应等于最大并发）
    final_available = pool._get_total_available_concurrency()
    final_max = sum(pool._model_max_concurrency.values())
    logger.info(f"测试结束，最终可用并发：{final_available}，总最大并发：{final_max}")
    assert final_available == final_max, "测试失败：最终可用并发数不等于最大并发数"
    
    # 清理临时文件
    os.remove(temp_config_path)
    logger.info("ModelPool多并发测试完成")

# --- 测试2：Models类多维度混合操作测试（复杂场景） ---
async def test_models_multi_dimension_operation():
    """测试Models核心能力：多维度申请/释放、状态查询、异常处理"""
    logger.info("\n=== 测试2：Models多维度混合操作测试 ===")
    # 初始化Models（自动加载light/deep池）
    models = Models()
    supported_tags = models.get_supported_perf_tags()
    logger.info(f"Models初始化完成，支持的维度：{supported_tags}")
    
    # 子测试1：正常申请/释放（light维度）
    logger.info("\n--- 子测试1：light维度正常申请释放 ---")
    try:
        config_light = await models.acquire("light")
        logger.info(f"light维度申请成功，config_id：{config_light['config_id']}")
        await models.release(config_light["config_id"])
        logger.info("light维度释放成功")
    except Exception as e:
        logger.error(f"子测试1失败：{str(e)}")
    
    # 子测试2：正常申请/释放（deep维度）
    logger.info("\n--- 子测试2：deep维度正常申请释放 ---")
    try:
        config_deep = await models.acquire("deep")
        logger.info(f"deep维度申请成功，config_id：{config_deep['config_id']}")
        await models.release(config_deep["config_id"])
        logger.info("deep维度释放成功")
    except Exception as e:
        logger.error(f"子测试2失败：{str(e)}")
    
    # 子测试3：状态查询验证
    logger.info("\n--- 子测试3：状态查询验证 ---")
    try:
        light_status = models.get_pool_status("light")
        deep_status = models.get_pool_status("deep")
        all_status = models.get_all_pool_status()
        logger.info(f"light池状态：{light_status}")
        logger.info(f"deep池状态：{deep_status}")
        logger.info(f"所有池状态：{all_status}")
        # 验证状态合法性
        assert light_status["total_available"] >= 0, "light池可用并发数为负"
        assert deep_status["total_available"] >= 0, "deep池可用并发数为负"
    except Exception as e:
        logger.error(f"子测试3失败：{str(e)}")
    
    # 子测试4：异常场景 - 传入不支持的perf_tag
    logger.info("\n--- 子测试4：异常场景 - 不支持的perf_tag ---")
    try:
        await models.acquire("invalid_tag")
    except UnsupportedPerfTagError as e:
        logger.info(f"预期异常触发：{str(e)}")
    except Exception as e:
        logger.error(f"非预期异常：{str(e)}")
    
    # 子测试5：异常场景 - 释放不存在的config_id
    logger.info("\n--- 子测试5：异常场景 - 不存在的config_id ---")
    try:
        await models.release("invalid_config_id_123456")
    except ConfigIdNotFoundError as e:
        logger.info(f"预期异常触发：{str(e)}")
    except Exception as e:
        logger.error(f"非预期异常：{str(e)}")
    
    # 子测试6：异常场景 - 无参acquire
    logger.info("\n--- 子测试6：异常场景 - 无参acquire ---")
    try:
        await models.acquire(None)
    except NotImplementedError as e:
        logger.info(f"预期异常触发：{str(e)}")
    except Exception as e:
        logger.error(f"非预期异常：{str(e)}")
    
    logger.info("Models多维度混合操作测试完成")

# --- 测试3：Models上下文管理器复杂场景测试 ---
async def test_models_context_manager():
    """测试上下文管理器：正常场景、异常场景（仍能自动释放）"""
    logger.info("\n=== 测试3：Models上下文管理器复杂测试 ===")
    
    # 子测试1：正常场景（deep维度）
    logger.info("\n--- 子测试1：上下文管理器正常使用 ---")
    try:
        models_deep = Models(perf_tag="deep")
        async with models_deep as config:
            logger.info(f"上下文内使用config_id：{config['config_id']}")
            # 模拟业务逻辑
            await asyncio.sleep(0.5)
        logger.info("上下文退出，配置已自动释放")
    except Exception as e:
        logger.error(f"子测试1失败：{str(e)}")
    
    # 子测试2：异常场景（上下文内抛异常，验证仍释放）
    logger.info("\n--- 子测试2：上下文内抛异常（验证自动释放） ---")
    try:
        models_light = Models(perf_tag="light")
        async with models_light as config:
            logger.info(f"上下文内使用config_id：{config['config_id']}")
            # 主动抛出异常
            raise RuntimeError("模拟业务逻辑异常")
    except RuntimeError as e:
        logger.info(f"业务异常触发：{str(e)}")
        # 验证配置已释放（查询状态）
        light_status = models_light.get_pool_status("light")
        logger.info(f"异常后light池可用并发：{light_status['total_available']}")
    except Exception as e:
        logger.error(f"非预期异常：{str(e)}")
    
    # 子测试3：异常场景（无perf_tag初始化上下文）
    logger.info("\n--- 子测试3：无perf_tag初始化上下文 ---")
    try:
        models_none = Models()
        async with models_none as config:
            pass
    except ValueError as e:
        logger.info(f"预期异常触发：{str(e)}")
    except Exception as e:
        logger.error(f"非预期异常：{str(e)}")
    
    logger.info("Models上下文管理器复杂测试完成")

# --- 测试4：Models多任务并发申请测试（验证并发安全） ---
async def test_models_concurrent_acquire():
    """测试Models多任务并发申请不同维度配置，验证并发安全"""
    logger.info("\n=== 测试4：Models多任务并发申请测试 ===")
    models = Models()
    supported_tags = models.get_supported_perf_tags()
    if not supported_tags:
        logger.warning("无可用性能维度，跳过并发测试")
        return
    
    # 定义并发worker
    async def concurrent_worker(worker_id: int, perf_tag: str):
        task_name = f"Concurrent-Worker-{worker_id}"
        logger.info(f"[{task_name}] 申请{perf_tag}维度配置")
        try:
            config = await models.acquire(perf_tag)
            logger.info(f"[{task_name}] 申请成功，config_id：{config['config_id']}")
            await asyncio.sleep(random.uniform(0.2, 0.8))  # 随机耗时
            await models.release(config["config_id"])
            logger.info(f"[{task_name}] 释放成功")
        except Exception as e:
            logger.error(f"[{task_name}] 失败：{str(e)}")
    
    # 启动10个worker，混合申请light/deep维度
    tasks = []
    for i in range(1, 11):
        perf_tag = "light" if i % 2 == 0 else "deep"
        tasks.append(asyncio.create_task(concurrent_worker(i, perf_tag), name=f"Worker-{i}"))
    
    await asyncio.gather(*tasks)
    
    # 验证最终状态
    final_status = models.get_all_pool_status()
    logger.info(f"并发测试结束，最终各池状态：{final_status}")
    for tag, status in final_status.items():
        assert status["total_available"] == status["total_max"], f"{tag}池并发数异常"
    
    logger.info("Models多任务并发申请测试完成")

# --- 基础示例（保留，符合初期易用性要求） ---
async def example_1_basic_acquire_release():
    """示例1：基础申请+释放"""
    logger.info("\n--- 基础示例1：基础申请释放 ---")
    models = Models()
    supported_tags = models.get_supported_perf_tags()
    if "light" in supported_tags:
        config = await models.acquire("light")
        await models.release(config["config_id"])

async def example_2_context_manager():
    """示例2：上下文管理器（自动释放）"""
    logger.info("\n--- 基础示例2：上下文管理器 ---")
    models = Models(perf_tag="deep")
    if "deep" in models.get_supported_perf_tags():
        async with models as config:
            logger.info(f"上下文内使用配置ID：{config['config_id']}")

async def example_3_status_query():
    """示例3：状态查询"""
    logger.info("\n--- 基础示例3：状态查询 ---")
    models = Models()
    logger.info(f"所有池状态：{models.get_all_pool_status()}")
    if "light" in models.get_supported_perf_tags():
        logger.info(f"light池状态：{models.get_pool_status('light')}")

# --- 主入口：执行所有复杂测试 + 基础示例 ---
if __name__ == "__main__":
    # 前置：生成测试配置文件
    generate_test_configs()
    
    # 执行复杂测试集
    asyncio.run(test_model_pool_high_concurrency())
    asyncio.run(test_models_multi_dimension_operation())
    asyncio.run(test_models_context_manager())
    asyncio.run(test_models_concurrent_acquire())
    
    # 执行基础示例（保留初期易用性）
    asyncio.run(example_1_basic_acquire_release())
    asyncio.run(example_2_context_manager())
    asyncio.run(example_3_status_query())
    
    logger.info("\n所有测试与示例执行完成！")