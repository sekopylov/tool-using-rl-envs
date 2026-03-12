from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from envs.meeting_ops_env import MeetingOpsEnv


OPS = [
    "get_meeting",
    "get_calendar",
    "get_room_availability",
    "find_common_slots",
    "move_meeting",
    "cancel_meeting",
    "notify_participants",
    "finish_task",
]

SYSTEM_PROMPT = """
You are a meeting operations agent.
Solve each task through interactive multi-turn tool usage.

Strict rules for EVERY assistant turn:
- Output exactly ONE tool call and nothing else.
- Use this exact format:
<tool_call>
{"name":"TOOL_NAME","arguments":{...}}
</tool_call>
- JSON must be valid (double quotes, proper colons, no trailing text).
- Never emit two <tool_call> blocks in one turn.
- After each <tool_response>, decide the next single tool call.
- Call finish_task only after goal_satisfied is true.

Valid finish call format:
<tool_call>
{"name":"finish_task","arguments":{}}
</tool_call>
""".strip()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def format_user_prompt(item: Data, env: MeetingOpsEnv) -> str:
    payload = {
        "task": item.question,
        "available_tools": OPS,
        "initial_state": item.initial_state,
        "tool_docs": env.describe_tools(),
    }
    return json.dumps(payload, ensure_ascii=False)


def to_verl_row(item: Data, env: MeetingOpsEnv, split: str, index: int) -> dict[str, Any]:
    gt = json.dumps(item.to_json(), ensure_ascii=False)
    episode_id = str((item.metadata or {}).get("episode_id", f"{split}_{index}"))
    shared_create_kwargs = {"ground_truth_json": gt}

    return {
        "data_source": "meeting_ops",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(item, env)},
        ],
        "ability": "tool_use",
        "reward_model": {"style": "rule", "ground_truth": [gt]},
        "extra_info": {
            "split": split,
            "index": index,
            "difficulty": item.difficulty,
            "episode_id": episode_id,
            "need_tools_kwargs": True,
            "tools_kwargs": {op: {"create_kwargs": shared_create_kwargs} for op in OPS},
        },
    }


def write_data_jsonl(path: Path, data_rows: list[Data]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data_rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_train(env: MeetingOpsEnv, train_size: int, difficulties: list[int], seed: int, max_attempts: int) -> list[Data]:
    out: list[Data] = []
    rng = random.Random(seed)
    for i in range(train_size):
        d = difficulties[i % len(difficulties)]
        sample = env.generate(
            num_of_questions=1,
            max_attempts=max_attempts,
            difficulty=d,
            seed=rng.randint(0, 10**9),
        )[0]
        out.append(sample)
    return out


def build_eval_buckets(
    env: MeetingOpsEnv,
    eval_per_difficulty: int,
    difficulties: list[int],
    seed: int,
    max_attempts: int,
) -> dict[int, list[Data]]:
    out: dict[int, list[Data]] = {}
    rng = random.Random(seed)
    for d in difficulties:
        rows: list[Data] = []
        for _ in range(eval_per_difficulty):
            rows.append(
                env.generate(
                    num_of_questions=1,
                    max_attempts=max_attempts,
                    difficulty=d,
                    seed=rng.randint(0, 10**9),
                )[0]
            )
        out[d] = rows
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/meetingops_v1")
    p.add_argument("--train-size", type=int, default=2500)
    p.add_argument("--eval-per-difficulty", type=int, default=100)
    p.add_argument("--train-difficulties", default="1,2,3,4,5")
    p.add_argument("--eval-difficulties", default="1,2,3,4,5")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=31415)
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = MeetingOpsEnv(max_steps=args.max_steps)
    train_diffs = parse_int_list(args.train_difficulties)
    eval_diffs = parse_int_list(args.eval_difficulties)

    train_data = build_train(env, args.train_size, train_diffs, args.train_seed, args.max_attempts)
    eval_buckets = build_eval_buckets(env, args.eval_per_difficulty, eval_diffs, args.eval_seed, args.max_attempts)

    train_rows = [to_verl_row(x, env, "train", i) for i, x in enumerate(train_data)]
    val_rows: list[dict[str, Any]] = []

    for d in eval_diffs:
        bucket = eval_buckets[d]
        write_data_jsonl(out_dir / f"eval_d{d}.jsonl", bucket)
        val_rows.extend(to_verl_row(x, env, f"eval_d{d}", i) for i, x in enumerate(bucket))

    write_parquet(out_dir / "train.parquet", train_rows)
    write_parquet(out_dir / "val.parquet", val_rows)

    stats = {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_difficulties": train_diffs,
        "eval_difficulties": eval_diffs,
        "eval_per_difficulty": args.eval_per_difficulty,
        "max_steps": args.max_steps,
        "train_parquet": str(out_dir / "train.parquet"),
        "val_parquet": str(out_dir / "val.parquet"),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
