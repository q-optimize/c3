import datetime
import os
import re

_PATH_INIT = os.path.join(os.getcwd(), 'c3', '__init__.py')
_PATH_SETUP = os.path.join(os.getcwd(), 'setup.py')

# get today date
now = datetime.datetime.now()
now_date = now.strftime("%Y%m%d")

print(f"Update init - replace version by {now_date}")
with open(_PATH_INIT, 'r') as fp:
    init = fp.read()
init = re.sub(r'__version__ = [\d\.\w\'"]+', f'__version__ = "{now_date}"', init)
with open(_PATH_INIT, 'w') as fp:
    fp.write(init)

print(f"Update setup - replace version by {now_date} and update to c3-toolset-nightly")
with open(_PATH_SETUP, 'r') as fp:
    setup = fp.read()
setup = re.sub(r'version=[\d\.\w\'"]+', f'version="{now_date}"', setup)
setup = re.sub(r'name="c3-toolset"', f'name="c3-toolset-nightly"', setup)
with open(_PATH_SETUP, 'w') as fp:
    fp.write(setup)