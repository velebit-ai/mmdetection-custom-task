#! /usr/bin/env python3
"""mmdet lib fixes"""

import shutil

from mmdet.apis import test

print(f"Replacing {test.__file__} with fixes/test.py")
shutil.copy("fixes/test.py", test.__file__)
