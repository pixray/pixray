import unittest
from util import *

class TestUtilMethods(unittest.TestCase):
    #region get_file_path
    def test_get_file_path_no_backslash(self):
        self.assertEqual(get_file_path('/testpath', 'testfile', '.png'), '/testpath/testfile.png')
    
    def test_get_file_path_with_backslash(self):
        self.assertEqual(get_file_path('/testpath/', 'testfile', '.png'), '/testpath/testfile.png')

    def test_get_file_path_filename_contains_backslash(self):
        self.assertRaises(ValueError, get_file_path, '/testpath/', '\\test\\filename.png', '.png')

    def test_get_file_path_filename_contains_slash(self):
        self.assertRaises(ValueError, get_file_path, '/testpath/', '/test/filename.png', '.png')

    def test_get_file_path_no_path(self):
        self.assertEqual(get_file_path('', 'testfile', '.png'), 'testfile.png')

    def test_get_file_no_filename_raises_error(self):
        self.assertRaises(ValueError, get_file_path, '/testpath/', None, '.png')

    def test_get_file_empty_filename_raises_error(self):
        self.assertRaises(ValueError, get_file_path, '/testpath/', ' ', '.png')
    
    def test_get_file_path_replaces_suffix(self):
        self.assertEqual(get_file_path('/testpath', 'testfile.png', '.mp4'), '/testpath/testfile.mp4')
    #endregion get_file_path

    #region parse_unit
    def test_parse_unit_valid_iterations(self):
        self.assertEqual(parse_unit('200iterations', 500, 'overlay_until'), 200)
    
    def test_parse_unit_valid_iterations_space(self):
        self.assertEqual(parse_unit('200 i', 500, 'overlay_until'), 200)

    def test_parse_unit_valid_percentage(self):
        self.assertEqual(parse_unit('50%', 500, 'overlay_until'), 250)

    def test_parse_unit_valid_percentage_space(self):
        self.assertEqual(parse_unit('33 percent', 500, 'overlay_until'), 165)

    def test_parse_unit_valid_invalid(self):
        self.assertRaises(ValueError, parse_unit, ' percent', 500, 'overlay_until')

    def test_parse_unit_none(self):
        self.assertEqual(parse_unit(None, 500, 'overlay_until'), None)

    def test_parse_unit_robust_format(self):
        self.assertEqual(parse_unit('200 iterATions    ', 500, 'overlay_until'), 200)
    #endregion parse_unit

if __name__ == '__main__':
    unittest.main()