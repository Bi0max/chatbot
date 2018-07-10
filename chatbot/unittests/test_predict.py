from unittest import TestCase

import numpy as np

from chatbot.predict import argmax_b_2d


class TestPredict(TestCase):
    def setUp(self):
        pass

    def test_argmax_b_2d(self):
        a = np.arange(10).reshape(5, 2)
        print(a)
        b = 3
        indices = argmax_b_2d(a, b)
        print(indices)
        self.assertTupleEqual(indices.shape, (b, 2))
        answer = np.array([[4, 1], [4, 0], [3, 1]])
        self.assertTrue((indices == answer).all())
        print()
        print(indices)
        print()
        print(a[indices])