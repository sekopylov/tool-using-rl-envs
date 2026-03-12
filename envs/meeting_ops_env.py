from __future__ import annotations

import copy
import json
import random
import re
from typing import Any, Optional

from base.data import Data
from base.tool_env import ToolEnv


_CONFIRM_RE = re.compile(r"\b(confirm|confirmation|approve|approved|подтверж|разрешаю|ok to proceed)\b", re.IGNORECASE)


class MeetingOpsEnv(ToolEnv):
    def __init__(self, name: str = "meeting_ops", max_steps: int = 8):
        super().__init__(name)
        self.max_steps = max_steps
        self._slots = [
            "09:00",
            "09:30",
            "10:00",
            "10:30",
            "11:00",
            "11:30",
            "12:00",
            "14:00",
            "14:30",
            "15:00",
            "15:30",
            "16:00",
        ]
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self._state: dict[str, Any] = {}
        self._done = False
        self._step_idx = 0
        self._pending_confirmation = False
        self._notified_meetings: set[str] = set()

    def describe_tools(self) -> str:
        return (
            "Tools:\n"
            "- get_meeting {meeting_id}\n"
            "- get_calendar {person_id, day?}\n"
            "- get_room_availability {day, duration_min?}\n"
            "- find_common_slots {participants, day, duration_min?}\n"
            "- move_meeting {meeting_id, day, start, room_id} [state-changing]\n"
            "- cancel_meeting {meeting_id} [state-changing]\n"
            "- notify_participants {meeting_id, message}\n"
            "- finish_task {}"
        )

    def current_known_entities(self) -> set[str]:
        out: set[str] = set()
        for p in self._state.get("people", []):
            out.add(str(p.get("person_id", "")))
        for r in self._state.get("rooms", []):
            out.add(str(r.get("room_id", "")))
        for m in self._state.get("meetings", []):
            out.add(str(m.get("meeting_id", "")))
            for pid in m.get("participants", []):
                out.add(str(pid))
            out.add(str(m.get("room_id", "")))
        return {x for x in out if x}

    def reset(self, data: Data) -> str:
        self._reset_runtime()
        self._state = copy.deepcopy(data.initial_state or {})
        self._notified_meetings = set(str(x) for x in self._state.get("notified_meetings", []))

        goal = self._state.get("goal", {})
        meetings = self._state.get("meetings", [])
        short_meetings = []
        for m in meetings[:8]:
            if m.get("canceled"):
                continue
            short_meetings.append(
                {
                    "meeting_id": m.get("meeting_id"),
                    "day": m.get("day"),
                    "start": m.get("start"),
                    "room_id": m.get("room_id"),
                    "participants": m.get("participants", []),
                }
            )

        obs = {
            "task": data.question,
            "goal": goal,
            "people": [p.get("person_id") for p in self._state.get("people", [])],
            "rooms": [r.get("room_id") for r in self._state.get("rooms", [])],
            "meetings": short_meetings,
            "tool_instructions": self.describe_tools(),
            "action_format": "Either free-text or TOOL_CALL {\"name\":...,\"args\":{...}}",
        }
        return json.dumps(obs, ensure_ascii=False)

    def _meeting_by_id(self, meeting_id: str) -> Optional[dict[str, Any]]:
        for m in self._state.get("meetings", []):
            if str(m.get("meeting_id")) == str(meeting_id):
                return m
        return None

    def _person_exists(self, person_id: str) -> bool:
        return any(str(p.get("person_id")) == str(person_id) for p in self._state.get("people", []))

    def _room_exists(self, room_id: str) -> bool:
        return any(str(r.get("room_id")) == str(room_id) for r in self._state.get("rooms", []))

    def _busy(self, person_id: str, day: int, start: str, ignore_meeting_id: Optional[str] = None) -> bool:
        for m in self._state.get("meetings", []):
            if m.get("canceled"):
                continue
            if ignore_meeting_id and str(m.get("meeting_id")) == str(ignore_meeting_id):
                continue
            if int(m.get("day")) == int(day) and str(m.get("start")) == str(start):
                if str(person_id) in [str(x) for x in m.get("participants", [])]:
                    return True
        return False

    def _room_busy(self, room_id: str, day: int, start: str, ignore_meeting_id: Optional[str] = None) -> bool:
        for m in self._state.get("meetings", []):
            if m.get("canceled"):
                continue
            if ignore_meeting_id and str(m.get("meeting_id")) == str(ignore_meeting_id):
                continue
            if int(m.get("day")) == int(day) and str(m.get("start")) == str(start) and str(m.get("room_id")) == str(room_id):
                return True
        return False

    def _any_conflicts(self) -> bool:
        meetings = [m for m in self._state.get("meetings", []) if not m.get("canceled")]
        for i in range(len(meetings)):
            for j in range(i + 1, len(meetings)):
                a = meetings[i]
                b = meetings[j]
                if int(a.get("day")) != int(b.get("day")) or str(a.get("start")) != str(b.get("start")):
                    continue
                if str(a.get("room_id")) == str(b.get("room_id")):
                    return True
                pa = {str(x) for x in a.get("participants", [])}
                pb = {str(x) for x in b.get("participants", [])}
                if pa & pb:
                    return True
        return False

    def goal_satisfied(self) -> bool:
        goal = self._state.get("goal", {})
        meeting_id = str(goal.get("meeting_id", ""))
        target_day = int(goal.get("target_day", -1))
        target_start = str(goal.get("target_start", ""))
        target_room = str(goal.get("target_room_id", ""))
        require_notify = bool(goal.get("require_notify", True))

        m = self._meeting_by_id(meeting_id)
        if m is None or m.get("canceled"):
            return False
        if int(m.get("day")) != target_day:
            return False
        if str(m.get("start")) != target_start:
            return False
        if str(m.get("room_id")) != target_room:
            return False
        if require_notify and meeting_id not in self._notified_meetings:
            return False
        if self._any_conflicts():
            return False
        return True

    def _parse_tool_call(self, action: str) -> tuple[Optional[str], dict[str, Any], bool]:
        prefix = "TOOL_CALL"
        if not action.startswith(prefix):
            return None, {}, False
        payload = action[len(prefix) :].strip()
        try:
            data = json.loads(payload)
        except Exception:
            return None, {}, True
        if not isinstance(data, dict):
            return None, {}, True
        name = data.get("name")
        args = data.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return None, {}, True
        return name, args, False

    def step(self, action: str):
        if self._done:
            return "Episode already finished.", 0.0, True, {"invalid_action": True, "action_type": "none"}

        self._step_idx += 1
        action = (action or "").strip()
        info: dict[str, Any] = {
            "step": self._step_idx,
            "invalid_action": False,
            "action_type": "free_text",
            "tool_name": None,
            "world_change": False,
            "explicit_confirmation": False,
            "observed_entities": [],
        }

        if action.startswith("TOOL_CALL"):
            name, args, parse_failed = self._parse_tool_call(action)
            info["action_type"] = "tool_call"
            if parse_failed or name is None:
                info["invalid_action"] = True
                obs = "Invalid TOOL_CALL format."
                reward = -0.1
            else:
                info["tool_name"] = name
                obs, reward, done, extra_info = self._apply_tool(name, args)
                info.update(extra_info)
                self._done = self._done or done
        else:
            explicit = bool(_CONFIRM_RE.search(action))
            info["explicit_confirmation"] = explicit
            if explicit:
                self._pending_confirmation = True
            obs = "Acknowledged. Continue with diagnostics or propose next action."
            reward = -0.01

        if self._step_idx >= self.max_steps and not self._done:
            self._done = True
            obs = f"{obs} Max steps reached."

        return obs, float(reward), bool(self._done), info

    def _apply_tool(self, name: str, args: dict[str, Any]):
        info = {
            "invalid_action": False,
            "world_change": False,
            "observed_entities": [],
            "used_confirmation": False,
        }

        if name == "get_meeting":
            meeting_id = str(args.get("meeting_id", ""))
            m = self._meeting_by_id(meeting_id)
            if m is None:
                info["invalid_action"] = True
                return "Meeting not found.", -0.1, False, info
            obs = json.dumps(m, ensure_ascii=False)
            info["observed_entities"] = [meeting_id, str(m.get("room_id", "")), *[str(x) for x in m.get("participants", [])]]
            return obs, -0.01, False, info

        if name == "get_calendar":
            person_id = str(args.get("person_id", ""))
            day = args.get("day", None)
            if not self._person_exists(person_id):
                info["invalid_action"] = True
                return "Person not found.", -0.1, False, info
            cal = []
            for m in self._state.get("meetings", []):
                if m.get("canceled"):
                    continue
                if person_id not in [str(x) for x in m.get("participants", [])]:
                    continue
                if day is not None and int(m.get("day")) != int(day):
                    continue
                cal.append(
                    {
                        "meeting_id": m.get("meeting_id"),
                        "day": m.get("day"),
                        "start": m.get("start"),
                        "room_id": m.get("room_id"),
                    }
                )
            info["observed_entities"] = [person_id, *[str(x.get("meeting_id")) for x in cal], *[str(x.get("room_id")) for x in cal]]
            return json.dumps(cal, ensure_ascii=False), -0.01, False, info

        if name == "get_room_availability":
            day = int(args.get("day", 1))
            free_by_room: dict[str, list[str]] = {}
            for r in self._state.get("rooms", []):
                room_id = str(r.get("room_id"))
                busy = set()
                for m in self._state.get("meetings", []):
                    if m.get("canceled"):
                        continue
                    if int(m.get("day")) == day and str(m.get("room_id")) == room_id:
                        busy.add(str(m.get("start")))
                free_by_room[room_id] = [s for s in self._slots if s not in busy]
            info["observed_entities"] = list(free_by_room.keys())
            return json.dumps({"day": day, "free_by_room": free_by_room}, ensure_ascii=False), -0.01, False, info

        if name == "find_common_slots":
            participants = [str(x) for x in args.get("participants", [])]
            day = int(args.get("day", 1))
            if not participants or any(not self._person_exists(p) for p in participants):
                info["invalid_action"] = True
                return "Invalid participants.", -0.1, False, info
            free = []
            for slot in self._slots:
                if any(self._busy(p, day, slot) for p in participants):
                    continue
                free.append(slot)
            info["observed_entities"] = participants
            return json.dumps({"day": day, "participants": participants, "common_slots": free}, ensure_ascii=False), -0.01, False, info

        if name == "move_meeting":
            meeting_id = str(args.get("meeting_id", ""))
            day = args.get("day", None)
            start = str(args.get("start", ""))
            room_id = str(args.get("room_id", ""))
            m = self._meeting_by_id(meeting_id)
            if m is None or day is None or not start or not room_id:
                info["invalid_action"] = True
                return "Invalid move_meeting args.", -0.1, False, info
            if not self._room_exists(room_id):
                info["invalid_action"] = True
                return "Room not found.", -0.1, False, info
            if start not in self._slots:
                info["invalid_action"] = True
                return "Unsupported start slot.", -0.1, False, info
            day = int(day)
            participants = [str(x) for x in m.get("participants", [])]
            if any(self._busy(p, day, start, ignore_meeting_id=meeting_id) for p in participants):
                info["invalid_action"] = True
                return "Cannot move: participant conflict.", -0.1, False, info
            if self._room_busy(room_id, day, start, ignore_meeting_id=meeting_id):
                info["invalid_action"] = True
                return "Cannot move: room conflict.", -0.1, False, info

            m["day"] = day
            m["start"] = start
            m["room_id"] = room_id
            info["world_change"] = True
            if self._pending_confirmation:
                info["used_confirmation"] = True
                self._pending_confirmation = False
            return "Meeting moved.", -0.02, False, info

        if name == "cancel_meeting":
            meeting_id = str(args.get("meeting_id", ""))
            m = self._meeting_by_id(meeting_id)
            if m is None:
                info["invalid_action"] = True
                return "Meeting not found.", -0.1, False, info
            m["canceled"] = True
            info["world_change"] = True
            if self._pending_confirmation:
                info["used_confirmation"] = True
                self._pending_confirmation = False
            return "Meeting canceled.", -0.02, False, info

        if name == "notify_participants":
            meeting_id = str(args.get("meeting_id", ""))
            m = self._meeting_by_id(meeting_id)
            if m is None:
                info["invalid_action"] = True
                return "Meeting not found.", -0.1, False, info
            self._notified_meetings.add(meeting_id)
            return "Participants notified.", -0.01, False, info

        if name == "finish_task":
            self._done = True
            goal_ok = self.goal_satisfied()
            terminal_reward = 0.1 if goal_ok else -0.2
            return "Task finished.", terminal_reward, True, info

        info["invalid_action"] = True
        return "Unknown tool.", -0.1, False, info

    def _difficulty_params(self, difficulty: int) -> dict[str, int]:
        d = max(1, min(10, int(difficulty)))
        return {
            "num_people": min(3 + d // 2, 8),
            "num_rooms": 1 + d // 4,
            "num_noise_meetings": 1 + d,
            "num_days": 1 if d <= 5 else 2,
        }

    def _generate_one(self, rng: random.Random, difficulty: int, episode_id: str, **kwargs: Any) -> Data:
        params = self._difficulty_params(difficulty)
        for k in list(params.keys()):
            if k in kwargs and kwargs[k] is not None:
                params[k] = int(kwargs[k])

        people = [{"person_id": f"P{i+1}", "name": f"Person {i+1}"} for i in range(params["num_people"])]
        rooms = [{"room_id": f"R{i+1}", "capacity": 6} for i in range(params["num_rooms"])]

        participants_count = min(max(2, 2 + difficulty // 3), len(people))
        main_participants = sorted(rng.sample([p["person_id"] for p in people], participants_count))

        goal_day = rng.randint(1, params["num_days"])
        target_start = rng.choice(self._slots)
        current_start = rng.choice([s for s in self._slots if s != target_start])
        target_room = rng.choice(rooms)["room_id"]
        current_room = rng.choice([r["room_id"] for r in rooms])

        meetings: list[dict[str, Any]] = [
            {
                "meeting_id": "M1",
                "day": goal_day,
                "start": current_start,
                "duration_min": 30,
                "room_id": current_room,
                "participants": main_participants,
                "priority": "high",
                "canceled": False,
            }
        ]

        for i in range(params["num_noise_meetings"]):
            tries = 0
            while tries < 20:
                tries += 1
                mid = f"M{i+2}"
                day = rng.randint(1, params["num_days"])
                start = rng.choice(self._slots)
                room_id = rng.choice(rooms)["room_id"]
                pcount = rng.randint(1, min(3, len(people)))
                part = sorted(rng.sample([p["person_id"] for p in people], pcount))

                # Keep target slot solvable.
                if day == goal_day and start == target_start:
                    if room_id == target_room or set(part) & set(main_participants):
                        continue

                meetings.append(
                    {
                        "meeting_id": mid,
                        "day": day,
                        "start": start,
                        "duration_min": 30,
                        "room_id": room_id,
                        "participants": part,
                        "priority": "normal",
                        "canceled": False,
                    }
                )
                break

        goal = {
            "meeting_id": "M1",
            "target_day": goal_day,
            "target_start": target_start,
            "target_room_id": target_room,
            "require_notify": True,
        }

        user_goal = (
            f"Reschedule meeting M1 to day {goal_day} at {target_start} in room {target_room}. "
            f"Keep participants unchanged, notify participants, then finish task."
        )

        initial_state = {
            "people": people,
            "rooms": rooms,
            "meetings": meetings,
            "goal": goal,
            "notified_meetings": [],
        }

        return Data(
            question=user_goal,
            answer="goal_satisfied",
            difficulty=difficulty,
            initial_state=initial_state,
            metadata={"episode_id": episode_id, "difficulty": difficulty},
        )

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> list[Data]:
        d = max(1, min(10, int(difficulty if difficulty is not None else 1)))
        base_seed = int(kwargs.get("seed", 42))

        out: list[Data] = []
        attempts = 0
        while len(out) < num_of_questions and attempts < max_attempts * num_of_questions:
            attempts += 1
            episode_id = f"d{d}_seed{base_seed}_idx{len(out)}_att{attempts}"
            rng = random.Random(base_seed + attempts * 1009)
            out.append(self._generate_one(rng=rng, difficulty=d, episode_id=episode_id, **kwargs))
        return out
