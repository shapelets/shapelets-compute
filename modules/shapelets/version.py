# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

from ._version import get_versions

__ALL__ = ['version', 'full_version', 'git_revision', 'release']

version_info = get_versions()
version: str = version_info['version']
full_version: str = version_info['version']
git_revision: str = version_info['full-revisionid']
release: bool = 'dev0' not in version

del get_versions, version_info
