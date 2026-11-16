import threading
import unittest

import numpy as np

from openpi_client.future_action_chunk_broker import FutureActionChunkBroker


class FakePolicy:
    def __init__(self):
        self.requests=[]
        self.started=threading.Event()
        self.release=threading.Event()

    def infer(self,obs):
        index=len(self.requests)
        self.requests.append(obs)
        if index:
            self.started.set()
            if not self.release.wait(5):raise RuntimeError("test timed out")
        return {"actions":np.repeat((index*100+np.arange(10))[:,None],7,axis=1)}

    def reset(self):pass


class ScheduleTest(unittest.TestCase):
    def test_all_supported_schedules_preserve_context_and_future_alignment(self):
        for execute,overlap in [(5,2),(5,1),(10,3),(10,2)]:
            with self.subTest(execute=execute,overlap=overlap):
                policy=FakePolicy()
                broker=FutureActionChunkBroker(policy,execute_steps=execute,overlap_steps=overlap)
                try:
                    got=[]
                    for i in range(execute):
                        got.append(broker.infer({"state":np.full(8,i)})["actions"][0])
                        broker.mark_action_started()
                        if i==execute-overlap:self.assertTrue(policy.started.wait(5))
                    np.testing.assert_array_equal(got,np.arange(execute))
                    np.testing.assert_array_equal(policy.requests[1]["action_context"][:,0],np.arange(execute-overlap,execute))
                    np.testing.assert_array_equal(policy.requests[1]["state"],np.full(8,execute-overlap))
                    policy.release.set()
                    self.assertEqual(broker.infer({"state":np.zeros(8)})["actions"][0],100)
                    broker.mark_action_started()
                    record=broker.inference_records[1]
                    self.assertEqual(record["context_steps"],overlap)
                    self.assertGreaterEqual(record["observation_age_ms"],0)
                    self.assertAlmostEqual(record["uncovered_ms"],max(0,
                        record["submit_to_ready_ms"]-record["hide_window_ms"]),places=5)
                finally:
                    policy.release.set();broker.close()

    def test_terminal_prefetch_excluded_even_if_already_ready(self):
        policy=FakePolicy();policy.release.set()
        broker=FutureActionChunkBroker(policy)
        try:
            for _ in range(5):broker.infer({})
            broker.reset()
            self.assertTrue(broker.inference_records[-1]["terminal_prefetch"])
            self.assertIsNone(broker.inference_records[-1]["first_new_action_ns"])
        finally:broker.close()

    def test_worker_errors_propagate(self):
        class Broken(FakePolicy):
            def infer(self,obs):
                if len(obs["action_context"]):raise RuntimeError("worker failed")
                return super().infer(obs)
        broker=FutureActionChunkBroker(Broken())
        try:
            for _ in range(5):broker.infer({})
            with self.assertRaisesRegex(RuntimeError,"worker failed"):broker.infer({})
        finally:broker.close()

    def test_invalid_schedules(self):
        for e,o in [(5,0),(5,5),(11,1),(5,1.5)]:
            with self.assertRaises(ValueError):FutureActionChunkBroker(None,execute_steps=e,overlap_steps=o)


if __name__=="__main__":unittest.main()
