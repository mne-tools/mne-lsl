# auto import subpackages
from pycnbi.utils import q_common as qc
import os
ROOT = qc.parse_path(os.path.realpath(__file__)).dir
for d in qc.get_dir_list(ROOT):
    if os.path.exists('%s/__init__.py' % d):
        exe_package = 'import pycnbi.%s' % d.replace(ROOT + '/', '')
        exec(exe_package)
