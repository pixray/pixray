import unittest
from pixray import *

#TODO: Refactor pixray so that arguments can be mocked more cleanly.

class TestPixrayMethods(unittest.TestCase):
    def setupTestParser(self):
        parser = argparse.ArgumentParser()
        return setup_parser(parser)
    
    def apply_overlay_all_true(self):
        parser = self.setupTestParser()
        args = ['--overlay_image', 'asdf', '--overlay_every', 1, '--overlay_offset', 1, '--overlay_until', 100]
        result = parser.parse_args(args)
        self.assertEqual(apply_overlay(args, 10))