# MeetingOps Tool-Using RL (Research in LLMs, HW4)

## 1) Environment (envs/meeting_ops_env.py)

MeetingOpsEnv is the core simulator used both for data generation and online rollout execution.

### 1.1 State and Goal Model

The environment state includes:
1. people
2. rooms
3. meetings
4. goal
5. notified_meetings

Runtime fields track progression and termination:
1. _done
2. _step_idx
3. _pending_confirmation
4. _notified_meetings

Goal satisfaction (goal_satisfied) requires:
1. target meeting ID/day/start/room match the goal,
2. notification done when require_notify=true,
3. no remaining participant/room conflicts.

### 1.2 Tool Operations Supported

Environment supports these actions through TOOL_CALL:
1. get_meeting
2. get_calendar
3. get_room_availability
4. find_common_slots
5. move_meeting
6. cancel_meeting
7. notify_participants
8. finish_task

Read tools return observations. Write tools mutate world state with argument and conflict checks.

### 1.3 Step API and Termination

1. reset(data) loads initial_state and returns initial JSON observation.
2. step(action) accepts free text or TOOL_CALL {...} and returns (obs, reward, done, info).
3. Episode ends when:
   - finish_task is called, or
   - max steps is reached.

### 1.4 Environment Reward Signals

Per-step environment rewards:
1. invalid action: -0.1
2. read-only tool call: -0.01
3. state-changing call (move_meeting/cancel_meeting): -0.02
4. turn without tool calling: -0.01
5. finish_task: +0.1 if goal satisfied, otherwise -0.2

## 2) Tasks and Dataset Generation

### 2.1 Task Construction

Tasks are generated from the environment by scripts/build_meetingops_dataset.py.

Each sample encodes:
1. a user goal (reschedule M1 + notify + finish)
2. initial world state
3. metadata (difficulty, episode ID)
4. reward ground truth payload

### 2.2 Dataset Outputs

The builder writes:
1. train.parquet
2. val.parquet
3. `eval_d*.jsonl`
4. stats.json

Dataset directory:
1. data/meetingops_v1

- train size: 500
- val size: 500

Inside `eval_d*.jsonl` real tasks can be seen.

### 2.3 How Difficulty Works

Difficulty is explicitly controllable during generation.

Control flags in build_meetingops_dataset.py:
1. --train-difficulties (default 1,2,3,4,5)
2. --eval-difficulties (default 1,2,3,4,5)

Selection logic:
1. train: round-robin over --train-difficulties
2. validation/test: eval_per_difficulty samples for each eval difficulty

In MeetingOpsEnv._difficulty_params, input difficulty is clamped to 1..10, then mapped to complexity:
1. num_people = min(3 + d // 2, 8)
2. num_rooms = 1 + d // 4
3. num_noise_meetings = 1 + d
4. num_days = 1 if d <= 5 else 2

Target-meeting participant count also increases with difficulty (2 + d // 3).

Separate seeds are used for train and eval (--train-seed, --eval-seed)

## 3) VERL Async Rollout Collection

This pipeline uses verl async agent-loop rollout.

### 3.1 Key Async Configuration

1. actor_rollout_ref.rollout.mode=async
2. actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
3. actor_rollout_ref.rollout.agent.num_workers=2
4. actor_rollout_ref.rollout.multi_turn.format=hermes
5. actor_rollout_ref.rollout.multi_turn.tool_config_path=config/meetingops_tool_config.yaml

### 3.2 How Async Rollout Works Here

1. VERL samples a batch from parquet rows.
2. tool_agent runs multi-turn generation.
3. Tool calls are parsed from model output.
4. meetingops_tool.py executes one environment operation per call using episode payload (ground_truth_json).
5. Tool responses are injected back into the chat context.
6. Trajectory-level reward is computed by compute_score function in scripts/reward_meetingops_wrapper.py.
7. GRPO updates actor weights.

### 3.3 Custom Integration

1. Tool runtime bridge: meetingops_tool.py
2. Tool schemas: config/meetingops_tool_config.yaml
3. Reward function: scripts/reward_meetingops_wrapper.py

## 4) Training Runs and Results

### 4.1 Main Completed Run

1. run dir: runs/grpo_qwen3_1p7b_meetingops_main
2. model: Qwen/Qwen3-1.7B
3. GPUs: 0,1
4. trainer: GRPO (algorithm.adv_estimator=grpo)
5. completed global steps: 60

### 4.2 Final Validation Metrics

1. Tool-step reward in meetingops_tool.py: tool_reward = env_reward; on terminal done, add +1.0 for success or -1.0 for failed completion, so successful completion is strongly preferred.
2. Trajectory score in scripts/reward_meetingops_wrapper.py starts from sum(tool_rewards) and then applies: +2.0 if success; else -1.0 if done but failed; else -0.6 if not done (favor successful completion and penalize incomplete/failed trajectories).
3. Additional wrapper shaping terms: `+0.15 * min(world_changes, 2)` (reward doing necessary world updates but cap to avoid farming edits), `+0.15 * min(notify_hits, 1)` (reward notification evidence once), `-0.03 * steps` (encourage shorter plans), `-0.20 * extra_unexecuted_tool_blocks` (penalize emitting tool blocks that were not actually executed), and if finish_task is mentioned but episode is not done, `-0.20 * min(finish_mentions, 3)` (penalize spurious finish claims).
4. Validation metrics are aggregated in verl-main/verl/trainer/ppo/ray_trainer.py (`_validate`). Because of this score is a shaped trajectory metric and is not bounded by the raw +0.1 env terminal reward.

### 4.3 Baseline vs Trained

I did trainer.val_before_train=False and didn't save metrics for step 0. So comparison below uses step 5 as baseline proxy.

| Metric | Baseline Proxy | Trained (step 60) | Delta |
|---|---:|---:|---:|
| acc | 0.4800 | 0.5960 | +0.1160 |
| success | 0.4800 | 0.5960 | +0.1160 |
| score | 0.8398 | 1.2888 | +0.4490 |
| done | 0.8840 | 1.0000 | +0.1160 |


Validation generation examples are available in runs/grpo_qwen3_1p7b_meetingops_main/val_generations; only a curated subset is included here due to memory limits.

## 5) Reproducibility and Commands

VERL pin:
1. repo: https://github.com/volcengine/verl.git
2. commit: d9d94b4da93fbacc06bb546609171c67c0a674aa

sync training
```bash
VENV=~/.venv CUDA_DEVICES=0,1 bash scripts/run_grpo_meetingops_main.sh
```
