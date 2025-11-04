import logging
from unittest import TestCase
from package_name.utils import get_file_paths_list


LOG = logging.getLogger(__name__)


class PackageNameTest(TestCase):

    def setUp(self):
        ...
    
    def test_get_file_paths_list_function(self):
        file_paths = get_file_paths_list('./')
        LOG.debug(str(file_paths))
        self.assertIn('./pyproject.toml', file_paths)
        self.assertIn('./tests', file_paths)
        self.assertIn('./requirements.txt', file_paths)
        self.assertIn('./Makefile', file_paths)
        self.assertIn('./pytest.ini', file_paths)
        self.assertIn('./src', file_paths)
        self.assertIn('./logging.conf', file_paths)
        self.assertIn('./README.md', file_paths)
        self.assertIn('./CHANGELOG.md', file_paths)
        self.assertIn('./.gitignore', file_paths)
