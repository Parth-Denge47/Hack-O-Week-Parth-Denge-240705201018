"""Tests for Week 3. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from python_essentials import (Inventory, Item, PerishableItem, Point, fibonacci, flatten,  # noqa: E402
                               read_in_chunks, running_average, squares_of_evens, unique_initials)


class TestWeek3(unittest.TestCase):
    def test_inventory_merges_duplicates(self):
        inv = Inventory()
        inv.add(Item("Pen", 10, 5))
        inv.add(Item("Pen", 10, 3))
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv["Pen"].qty, 8)

    def test_perishable_discount(self):
        self.assertEqual(PerishableItem("Milk", 10, 2, days_left=1).total_value(), 10)
        self.assertEqual(PerishableItem("Milk", 10, 2, days_left=5).total_value(), 20)

    def test_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            Item("Bad", -1)

    def test_point_add(self):
        self.assertEqual(Point(1, 2) + Point(3, 4), Point(4, 6))

    def test_comprehensions(self):
        self.assertEqual(squares_of_evens(7), [0, 4, 16, 36])
        self.assertEqual(unique_initials(["ant", "Apple", "bee"]), {"A", "B"})
        self.assertEqual(flatten([[1], [2, 3]]), [1, 2, 3])

    def test_generators(self):
        self.assertEqual(list(fibonacci(20)), [0, 1, 1, 2, 3, 5, 8, 13])
        self.assertEqual(list(read_in_chunks([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(running_average([2, 4])), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
