import unittest
from pixray import *

class TestPixrayMethods(unittest.TestCase):
    def setup_test_parser(self):
        parser = argparse.ArgumentParser()
        return setup_parser(parser)
    
    def get_args_for_apply_overlay_test(self, overlay_image, overlay_every, overlay_offset, overlay_until):
        parser = self.setup_test_parser()
        args = ['--overlay_image', overlay_image, '--overlay_every', str(overlay_every), '--overlay_offset', str(overlay_offset), '--overlay_until', str(overlay_until)]
        return parser.parse_args(args)

    def test_apply_overlay_all_true(self):
        args = self.get_args_for_apply_overlay_test('image.png', 1, 0, 100)
        self.assertEqual(apply_overlay(args, 10), True)

    def test_apply_overlay_no_overlay_image(self):
        args = self.get_args_for_apply_overlay_test(None, 1, 0, 100)
        args.overlay_image = None
        self.assertEqual(apply_overlay(args, 10), False)

    def test_apply_overlay_not_at_offset(self):
        args = self.get_args_for_apply_overlay_test('image.png', 5, 10, 100)
        self.assertEqual(apply_overlay(args, 10), False)

    def test_apply_overlay_less_than_overlay_until(self):
        args = self.get_args_for_apply_overlay_test('image.png', 1, 0, 5)
        self.assertEqual(apply_overlay(args, 10), False)

if __name__ == '__main__':
    unittest.main()
