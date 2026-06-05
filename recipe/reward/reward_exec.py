import json
import re
import logging
import subprocess
import tempfile
import os
from typing import Dict, Any
# -----------------------------
# 日志配置
# -----------------------------
LOGLEVEL = os.environ.get("REWARD_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="[%(levelname)s][%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Reward function initialized, loglevel=%s", LOGLEVEL)

# -----------------------------
# IMPORT PROMPT (可选)
# -----------------------------
IMPORT_PROMPT = '''from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')
'''

# -----------------------------
# 单个测试执行函数
# -----------------------------
def _run_single_test(solution_code, test_input, expected_output, timeout=2):
    try:
        full_code = (
            IMPORT_PROMPT
            + solution_code
            + "\n"
            + f"print(run({repr(test_input)}))"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(full_code)
            f.flush()

            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        if result.returncode != 0:
            return False, result.stderr.strip()

        pred = result.stdout.strip()
        exp = expected_output.strip()

        return pred == exp, f"pred={pred}, exp={exp}"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

# -----------------------------
# 主 reward 函数
# -----------------------------
def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    timeout: int = 2,
    **kwargs
) -> float:
    """
    reward function for inputs/outputs dataset

    Args:
        solution_str: 模型输出的完整文本
        ground_truth: dict with "inputs" and "outputs"
        timeout: 每个测试最大秒数

    Returns:
        float: 1.0 if all tested cases passed, else 0.0
    """
    MAX_TESTS = int(os.environ.get("REWARD_MAX_TESTS", -1))

    try:
        # 提取代码块
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            logger.warning("No python code block found")
            return 0.0

        solution = solutions[-1]
        logger.debug("Extracted solution code, length=%d", len(solution))

        # 解析 ground truth
        if isinstance(ground_truth, str):
            gt_data = json.loads(ground_truth)
        else:
            gt_data = ground_truth

        inputs = gt_data["inputs"]
        outputs = gt_data["outputs"]

        logger.info("Total test cases: %d", len(inputs))

        # 只跑前 K 个测试
        if MAX_TESTS > 0:
            logger.warning("Limiting test cases to first %d (DEBUG MODE)", MAX_TESTS)
            inputs = inputs[:MAX_TESTS]
            outputs = outputs[:MAX_TESTS]

        # 执行测试
        passed_all = True
        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            logger.debug("Running test %d", idx)
            success, message = _run_single_test(solution, inp, out, timeout=timeout)
            if not success:
                logger.info("Test %d FAILED: %s", idx, message)
                passed_all = False
                break
            else:
                logger.debug("Test %d PASSED", idx)

        return 1.0 if passed_all else 0.0

    except Exception as e:
        logger.exception("Fatal error in reward execution")
        return 0.0

# -----------------------------
# 简单测试示例
# -----------------------------
if __name__ == "__main__":
    # 环境变量可控制日志和最大测试数
    os.environ["REWARD_MAX_TESTS"] = "1"
    os.environ["REWARD_LOGLEVEL"] = "DEBUG"

    sample_solution = '''
```python
def run(inp):
    # 简单示例，返回输入长度
    return len(inp)```'''
    sample_ground_truth = {
    "inputs": [[1,2,3,4], [5,6]],
    "outputs": ["4", "2"]
    }
    score = compute_reward(sample_solution, sample_ground_truth)
    print("Reward score:", score)