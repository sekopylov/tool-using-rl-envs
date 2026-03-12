from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERL_ROOT = ROOT / "verl-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from base.data import Data
from envs.meeting_ops_env import MeetingOpsEnv
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse


_RUNTIME_KEY = "meeting_runtime"
_TRACE_KEY = "meeting_trace"


class MeetingOpsTool(BaseTool):
    """MeetingOps environment operation tool (one tool per operation name)."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config=config, tool_schema=tool_schema)
        self.max_steps = config.get("max_steps", 8)

    def _init_runtime(self, create_kwargs: dict) -> dict:
        payload = json.loads(create_kwargs["ground_truth_json"])

        data = Data.from_json(payload)
        env = MeetingOpsEnv(max_steps=self.max_steps)
        env.reset(data)

        return {
            "state": copy.deepcopy(env._state),
            "done": env._done,
            "step_idx": env._step_idx,
            "pending_confirmation": env._pending_confirmation,
            "notified_meetings": sorted(env._notified_meetings),
            "world_changes": 0,
            "notify_calls": 0,
            "finish_calls": 0,
        }

    def _restore_env(self, runtime: dict) -> MeetingOpsEnv:
        env = MeetingOpsEnv(max_steps=self.max_steps)
        env._state = copy.deepcopy(runtime["state"])
        env._done = runtime["done"]
        env._step_idx = runtime["step_idx"]
        env._pending_confirmation = runtime["pending_confirmation"]
        env._notified_meetings = set(runtime["notified_meetings"])
        return env

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        _ = instance_id
        agent_data = kwargs["agent_data"]

        op_name = self.name
        create_kwargs = agent_data.tools_kwargs[op_name]["create_kwargs"]

        runtime = agent_data.extra_fields.get(_RUNTIME_KEY)
        if runtime is None:
            runtime = self._init_runtime(create_kwargs)
            agent_data.extra_fields[_RUNTIME_KEY] = runtime

        env = self._restore_env(runtime)

        op_args = parameters
        action = "TOOL_CALL " + json.dumps({"name": op_name, "args": op_args}, ensure_ascii=False)
        obs, env_reward, done, info = env.step(action)

        goal_satisfied = env.goal_satisfied()
        success = done and goal_satisfied

        tool_reward = env_reward
        if done:
            tool_reward += 1.0 if success else -1.0

        world_change = info["world_change"]
        invalid_action = info["invalid_action"]
        world_changes = runtime["world_changes"] + int(world_change)
        notify_calls = runtime["notify_calls"] + int(op_name == "notify_participants")
        finish_calls = runtime["finish_calls"] + int(op_name == "finish_task")

        agent_data.extra_fields[_RUNTIME_KEY] = {
            "state": copy.deepcopy(env._state),
            "done": env._done,
            "step_idx": env._step_idx,
            "pending_confirmation": env._pending_confirmation,
            "notified_meetings": sorted(env._notified_meetings),
            "world_changes": world_changes,
            "notify_calls": notify_calls,
            "finish_calls": finish_calls,
        }

        trace = agent_data.extra_fields.setdefault(_TRACE_KEY, [])
        trace.append(
            {
                "step": env._step_idx,
                "op": op_name,
                "args": op_args,
                "obs": obs,
                "env_reward": float(env_reward),
                "tool_reward": float(tool_reward),
                "done": done,
                "goal_satisfied": goal_satisfied,
                "invalid_action": invalid_action,
                "world_change": world_change,
            }
        )

        agent_data.extra_fields.update(
            {
                "meeting_done": done,
                "meeting_success": success,
                "meeting_steps": env._step_idx,
                "meeting_goal_satisfied": goal_satisfied,
                "meeting_last_op": op_name,
                "meeting_world_changes": world_changes,
                "meeting_notify_calls": notify_calls,
                "meeting_finish_calls": finish_calls,
            }
        )

        response = {
            "observation": obs,
            "done": done,
            "goal_satisfied": goal_satisfied,
            "invalid_action": invalid_action,
            "world_change": world_change,
        }
        return ToolResponse(text=json.dumps(response, ensure_ascii=False)), float(tool_reward), info
