import unittest
from util import *

class TestUtilMethods(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()