import os
import pycnbi

# auto import subpackages
from pycnbi.utils import q_common as qc
os.chdir(qc.parse_path(__file__).dir)
for d in qc.get_dir_list('.'):
    if os.path.exists('%s/__init__.py' % d):
        exe_package = 'import pycnbi.%s' % d[2:]
        exec(exe_package)
