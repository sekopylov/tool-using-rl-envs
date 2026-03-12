from __future__ import annotations


def compute_score(data_source, solution_str, ground_truth, extra_info=None, shaping=True, **kwargs):
    _ = data_source
    _ = ground_truth
    _ = kwargs

    extra = extra_info or {}
    rollout = extra.get("rollout_reward_scores", {})

    tool_rewards = rollout.get("tool_rewards", extra.get("tool_rewards", []))
    steps = rollout.get("meeting_steps", extra.get("meeting_steps", len(tool_rewards)))

    inferred_done = 1 if any(abs(r) >= 0.8 for r in tool_rewards) else 0
    inferred_success = 1 if any(r >= 0.8 for r in tool_rewards) else 0

    done = rollout.get("meeting_done", extra.get("meeting_done", inferred_done))
    success = rollout.get("meeting_success", extra.get("meeting_success", inferred_success))

    output = solution_str or ""
    world_changes = output.count("world_change")
    notify_hits = output.count("Participants notified.")
    tool_tag_count = output.count("<tool_call>")
    finish_mentions = output.count("finish_task")

    tool_reward_sum = float(sum(tool_rewards))
    extra_unexecuted = max(0, tool_tag_count - len(tool_rewards))

    if shaping:
        score = tool_reward_sum
        score += 2.0 if success else (-1.0 if done else -0.6)
        score += 0.15 * min(world_changes, 2)
        score += 0.15 * min(notify_hits, 1)
        score -= 0.03 * max(0, steps)
        score -= 0.20 * extra_unexecuted
        if finish_mentions > 0 and done == 0:
            score -= 0.20 * min(finish_mentions, 3)
    else:
        score = 1.0 if success else -1.0

    return {
        "score": float(score),
        "acc": float(1 if success else 0),
        "success": float(1 if success else 0),
        "done": float(1 if done else 0),
        "steps": float(steps),
        "tool_calls": float(len(tool_rewards)),
        "tool_reward_sum": float(tool_reward_sum),
        "world_changes": float(world_changes),
        "notify_hits": float(notify_hits),
        "finish_mentions": float(finish_mentions),
        "tool_tag_count": float(tool_tag_count),
        "extra_unexecuted_tool_blocks": float(extra_unexecuted),
    }
