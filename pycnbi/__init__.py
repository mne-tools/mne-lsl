# auto import subpackages
from pycnbi.utils import q_common as qc
import os
PYCNBI_ROOT = qc.parse_path(os.path.realpath(__file__)).dir
for d in qc.get_dir_list(PYCNBI_ROOT):
    if os.path.exists('%s/__init__.py' % d):
        exe_package = 'import pycnbi.%s' % d.replace(PYCNBI_ROOT + '/', '')
        exec(exe_package)
