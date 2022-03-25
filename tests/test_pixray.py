import unittest
from pixray import *

class TestPixrayMethods(unittest.TestCase):
    def setup_test_parser(self):
        parser = argparse.ArgumentParser()
        return setup_parser(parser)
    
    def get_args_for_apply_overlay_test(self, overlay_image, overlay_every, overlay_offset, overlay_until):
        settings = { 
                     '--overlay_image': overlay_image,
                     '--overlay_every': overlay_every,
                     '--overlay_offset': overlay_offset,
                     '--overlay_until': overlay_until
        }

        return self.parse_dictionary_to_args(settings)

    def parse_dictionary_to_args(self, settings_dict):
        parser = self.setup_test_parser()
        args = []
        for key, value in settings_dict.items():
            if value is not None:
                args.append(key)
                args.append(value)
            
        args = parser.parse_args(args)
        args.overlay_offset = parse_unit(args.overlay_offset, args.iterations, "overlay_offset")
        args.overlay_until = parse_unit(args.overlay_until, args.iterations, "overlay_until")
        args.overlay_every = parse_unit(args.overlay_every, args.iterations, "overlay_every")
        return args

    #region apply_overlay
    def test_apply_overlay_all_true(self):
        args = self.get_args_for_apply_overlay_test('image.png', '1i', '0i', '100i')
        self.assertEqual(apply_overlay(args, 10), True)

    def test_apply_overlay_no_overlay_image(self):
        args = self.get_args_for_apply_overlay_test(None, '1i', '0i', '100i')
        self.assertEqual(apply_overlay(args, 10), False)

    def test_apply_overlay_not_at_offset(self):
        args = self.get_args_for_apply_overlay_test('image.png', '5i', '10i', '100i')
        self.assertEqual(apply_overlay(args, 10), False)

    def test_apply_overlay_overlay_until_none(self):
        args = self.get_args_for_apply_overlay_test('image.png', '5i', '10i', None)
        self.assertEqual(apply_overlay(args, 10), False)

    def test_apply_overlay_less_than_overlay_until(self):
        args = self.get_args_for_apply_overlay_test('image.png', '1i', '0i', '5i')
        self.assertEqual(apply_overlay(args, 10), False)
    #endregion apply_overlay

    #region get_learning_rate_drops
    def test_get_learning_rate_drops_empty(self):
        self.assertEqual(get_learning_rate_drops(None, 300), [])

    def test_get_learning_rate_drops_single(self):
        self.assertEqual(get_learning_rate_drops([75], 300), [224])

    def test_get_learning_rate_drops_multi(self):
        self.assertEqual(get_learning_rate_drops([50, 22.5], 300), [149, 67])
    #endregion get_learning_rate_drops


if __name__ == '__main__':
    unittest.main()

