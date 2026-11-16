"""Future action scheduling with measured submit-to-boundary hiding windows."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from openpi_client import base_policy


def validate_schedule(execute_steps, overlap_steps):
    if (type(execute_steps) is not int or type(overlap_steps) is not int
            or not 1 <= overlap_steps < execute_steps <= 10 or overlap_steps > 9):
        raise ValueError("Require integer 1 <= overlap_steps < execute_steps <= 10")


class FutureActionChunkBroker(base_policy.BasePolicy):
    """Predict after queued actions, then execute actions[:E] without output skips.

    Clock origin for observations is env.step return, when supplied by the caller.
    Records are appended by the caller after a result is joined, never concurrently.
    """

    def __init__(self, policy, *, execute_steps=5, overlap_steps=2, record_timing=True):
        validate_schedule(execute_steps, overlap_steps)
        self.policy = policy
        self.execute_steps = execute_steps
        self.overlap_steps = overlap_steps
        self.pre_steps = execute_steps - overlap_steps
        self.record_timing = record_timing
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="future-inference")
        self._current = None
        self._index = 0
        self._pending = None
        self._closed = False
        self._request_index = 0
        self.inference_records = []
        self.switch_wait_ms = 0.0
        self.overlap_hidden_ms = 0.0

    def _request(self, observation, record):
        record["start_ns"] = time.perf_counter_ns()
        result = self.policy.infer(observation)
        record["ready_ns"] = time.perf_counter_ns()
        actions = np.asarray(result["actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected ten finite 7D Future actions, got {actions.shape}")
        record["server_model_ms"] = result.get("policy_timing", {}).get("infer_ms")
        record["client_e2e_ms"] = (record["ready_ns"]-record["start_ns"])/1e6
        record["submit_to_ready_ms"] = (record["ready_ns"]-record["submit_ns"])/1e6
        return actions[:self.execute_steps].copy(), record

    def _new_record(self, observation, observation_ns, bootstrap):
        record = dict(index=self._request_index, observation_ns=observation_ns,
            submit_ns=time.perf_counter_ns(), bootstrap=bootstrap, terminal_prefetch=False,
            context_steps=len(observation["action_context"]), predicted_steps=10,
            executed_steps=self.execute_steps, overlap_steps=self.overlap_steps,
            boundary_ns=None, first_new_action_ns=None)
        self._request_index += 1
        return record

    def infer(self, obs, *, observation_ns=None):
        if self._closed:
            raise RuntimeError("Broker is closed")
        observation_ns = time.perf_counter_ns() if observation_ns is None else observation_ns
        if self._current is None:
            observation = copy.deepcopy(obs)
            observation["action_context"] = np.zeros((0, 7), dtype=np.float32)
            self._current, self._active_record = self._request(observation,
                self._new_record(observation, observation_ns, True))
            if self.record_timing:
                self.inference_records.append(self._active_record)
        elif self._index == self.execute_steps:
            if self._pending is None:
                raise RuntimeError("Missing pending Future request")
            boundary = observation_ns
            join_started = time.perf_counter_ns()
            try:
                self._current, self._active_record = self._pending.result()
            finally:
                self._pending = None
            joined = time.perf_counter_ns()
            record = self._active_record
            record["boundary_ns"] = boundary
            record["hide_window_ms"] = (boundary-record["submit_ns"])/1e6
            record["uncovered_ms"] = max(0, record["ready_ns"]-boundary)/1e6
            record["missed_deadline"] = record["ready_ns"] > boundary
            record["join_wait_ms"] = (joined-join_started)/1e6
            record["boundary_to_join_ms"] = (joined-boundary)/1e6
            self.switch_wait_ms += record["join_wait_ms"]
            self.overlap_hidden_ms += max(0, min(boundary,record["ready_ns"])-record["start_ns"])/1e6
            if self.record_timing:
                self.inference_records.append(record)
            self._index = 0
        if self._index == self.pre_steps and self._pending is None:
            observation = copy.deepcopy(obs)
            observation["action_context"] = self._current[self._index:].copy()
            if observation["action_context"].shape != (self.overlap_steps, 7):
                raise RuntimeError("Queued actions do not match overlap")
            record = self._new_record(observation, observation_ns, False)
            self._pending = self._executor.submit(self._request, observation, record)
        action = self._current[self._index].copy()
        self._index += 1
        return {"actions": action}

    def mark_action_started(self, timestamp_ns=None):
        """Call immediately before env.step; annotate first executed new action."""
        if self._index == 1 and self._active_record["first_new_action_ns"] is None:
            stamp = time.perf_counter_ns() if timestamp_ns is None else timestamp_ns
            self._active_record["first_new_action_ns"] = stamp
            self._active_record["observation_age_ms"] = (stamp-self._active_record["observation_ns"])/1e6

    def _drain(self):
        if self._pending is not None:
            try:
                _, record = self._pending.result()
                record["terminal_prefetch"] = True
                if self.record_timing:
                    self.inference_records.append(record)
            finally:
                self._pending = None

    def reset(self):
        try:
            self._drain()
        finally:
            self._current = None
            self._index = 0
            self.policy.reset()

    def close(self):
        if not self._closed:
            try:
                self._drain()
            finally:
                self._executor.shutdown(wait=True)
                self._closed = True
