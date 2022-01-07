import unittest
from utils.fileutils import *

class TestFileUtilMethods(unittest.TestCase):

    def test_get_file_name(self):
        # Happy path
        self.assertEqual(get_output_file_name('test.png', '.json'), 'test.json')
        
        # Dot in path
        self.assertEqual(get_output_file_name('/home/user.name/test.png', '.log'), '/home/user.name/test.log')

        # Empty path
        self.assertEqual(get_output_file_name('', ''), None)

        # No extension
        self.assertEqual(get_output_file_name('test', '.json'), 'test.json')


if __name__ == '__main__':
    unittest.main()
