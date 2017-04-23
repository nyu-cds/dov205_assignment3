"""
    Danny Vilela

    This program is used to provide a series of unit tests for our parallel
    sort implementation in `parallel_sorter.py`. To run these tests from the
    terminal, run the following from the project's root directory

        $ python -m unittest discover
"""

import unittest
from parallel_sorter import *


class ParallelSortTest(unittest.TestCase):

    def test_generation(self):
        """Verify that our data is generated according to specification."""

        # Standard case: low <= high, normal size.
        vals = generate_data(0, 10, 10)
        self.assertEqual(len(vals), 10)

        # Low >= high, normal size.
        vals = generate_data(10, 0, 10)
        self.assertEqual(len(vals), 10)

        # Low >= high, bad size.
        with self.assertRaises(ValueError):
            generate_data(10, 0, -1)

        # NoneType low, high, normal size.
        # Expected: program defaults to low=0, high=100, size=100.
        vals = generate_data(None, None, 10)
        self.assertEqual(len(vals), 100)

        # Normal low, high, NoneType size.
        # Expected: program defaults to low=0, high=100, size=100.
        vals = generate_data(0, 10, None)
        self.assertEqual(len(vals), 100)

    def test_merge(self):
        """Verify that our data is properly merged."""

        # Standard case: :a and :b are valid.
        vals = sorted(generate_data(0, 100, 100))
        a, b = vals[::2], vals[1::2]
        result = merge(a, b)
        self.assertEqual(len(result), len(a) + len(b))
        self.assertEqual(result, sorted(vals))

        # :a is empty, :b is valid.
        a, b = [], vals[1::2]
        result = merge(a, b)
        self.assertEqual(len(result), len(b))
        self.assertEqual(result, sorted(b))

        # :a is None, :b is valid.
        # Expected behavior: merge() raises a ValueError.
        a, b = None, vals[1::2]
        with self.assertRaises(ValueError):
            merge(a, b)

        # :a is None, :b is None.
        # Expected behavior: merge() raises a ValueError.
        a = b = None
        with self.assertRaises(ValueError):
            merge(a, b)

        # :a is a list of ints, :b is a list of floats.
        a = vals[::2]
        b = [i * 1.0 for i in vals[1::2]]
        result = merge(a, b)
        self.assertEqual(len(result), len(a) + len(b))
        
    def test_input(self):
        """Verify that our input size param is properly evaluated."""

        # Standard cast: param is valid.
        argv = ['parallel_sorter.py', 100]
        result = get_opt_input(argv)
        self.assertEqual(argv[1], result)

        argv[1] = str(argv[1])
        result = get_opt_input(argv)
        self.assertEqual(int(argv[1]), result)

        # :size is not provided.
        # Expected: get_opt_input defaults to size of 10000.
        argv = ['parallel_sorter.py']
        result = get_opt_input(argv)
        self.assertEqual(result, 10000)

        # :size is not easily cast to int.
        # Expected: get_opt_input defaults to size of 10000.
        argv = ['parallel_sorter.py', 'ten']
        result = get_opt_input(argv)
        self.assertEqual(result, 10000)

