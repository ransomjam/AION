import unittest
from aion.inference.stopping import (
    EosTokenCriteria,
    MaxNewTokensCriteria,
    StopSequenceCriteria,
    StoppingCriteriaList,
)


class TestMaxNewTokensCriteria(unittest.TestCase):

    def test_stops_at_limit(self):
        c = MaxNewTokensCriteria(3)
        self.assertFalse(c.should_stop([1, 2], 2, "ab"))
        self.assertTrue(c.should_stop([1, 2, 3], 3, "abc"))

    def test_stops_beyond_limit(self):
        c = MaxNewTokensCriteria(2)
        self.assertTrue(c.should_stop([1, 2, 3], 3, "abc"))


class TestEosTokenCriteria(unittest.TestCase):

    def test_stops_on_eos(self):
        c = EosTokenCriteria(eos_token_id=0)
        self.assertTrue(c.should_stop([1, 2, 0], 0, "ab<eos>"))

    def test_does_not_stop_on_other_token(self):
        c = EosTokenCriteria(eos_token_id=0)
        self.assertFalse(c.should_stop([1, 2, 3], 3, "abc"))


class TestStopSequenceCriteria(unittest.TestCase):

    def test_stops_on_suffix_match(self):
        c = StopSequenceCriteria(["END", "\n\n"])
        self.assertTrue(c.should_stop([1, 2], 2, "hello END"))

    def test_stops_on_newline_suffix(self):
        c = StopSequenceCriteria(["\n\n"])
        self.assertTrue(c.should_stop([1], 1, "text\n\n"))

    def test_does_not_stop_on_non_suffix(self):
        c = StopSequenceCriteria(["END"])
        self.assertFalse(c.should_stop([1, 2], 2, "END hello"))

    def test_empty_sequences_never_stops(self):
        c = StopSequenceCriteria([])
        self.assertFalse(c.should_stop([1, 2, 3], 3, "anything"))


class TestStoppingCriteriaList(unittest.TestCase):

    def test_empty_list_never_stops(self):
        sl = StoppingCriteriaList()
        self.assertFalse(sl([1, 2], 2, "ab"))

    def test_stops_when_any_fires(self):
        sl = StoppingCriteriaList([
            MaxNewTokensCriteria(10),
            EosTokenCriteria(0),
        ])
        self.assertTrue(sl([1, 0], 0, "a<eos>"))

    def test_does_not_stop_when_none_fire(self):
        sl = StoppingCriteriaList([
            MaxNewTokensCriteria(10),
            EosTokenCriteria(0),
        ])
        self.assertFalse(sl([1, 2], 2, "ab"))

    def test_len(self):
        sl = StoppingCriteriaList([MaxNewTokensCriteria(5), EosTokenCriteria(0)])
        self.assertEqual(len(sl), 2)

    def test_append(self):
        sl = StoppingCriteriaList()
        sl.append(MaxNewTokensCriteria(5))
        self.assertEqual(len(sl), 1)


if __name__ == "__main__":
    unittest.main()
