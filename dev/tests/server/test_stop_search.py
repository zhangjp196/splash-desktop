import unittest

from server.output import StopSearch


class StopSearchTests(unittest.TestCase):
    def test_finds_a_stop_across_chunks_and_reports_how_far_back(self):
        search = StopSearch(("needle", "second"))
        self.assertIsNone(search.search("haystack nee"))
        self.assertEqual(search.held, 3)
        self.assertEqual(search.search("dle more"), ("needle", 11))

    def test_the_earliest_matching_stop_wins(self):
        search = StopSearch(("ab", "b"))
        self.assertEqual(search.search("ab"), ("ab", 2))

    def test_no_match_holds_the_longest_prefix_of_a_stop(self):
        search = StopSearch(("abc", "bc"))
        self.assertIsNone(search.search("ab"))
        self.assertEqual(search.held, 2)
        self.assertIsNone(search.search("x"))
        self.assertEqual(search.held, 0)

    def test_a_long_stop_is_matched_character_by_character(self):
        stop = "a" * 16384 + "!"
        search = StopSearch((stop,))
        self.assertIsNone(search.search("a" * 16384))
        self.assertEqual(search.held, 16384)
        self.assertEqual(search.search("!"), (stop, len(stop)))


if __name__ == "__main__":
    unittest.main()
