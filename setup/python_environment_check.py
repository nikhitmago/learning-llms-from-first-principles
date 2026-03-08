# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib.metadata
import platform
import sys
from importlib.metadata import PackageNotFoundError, import_module
from os.path import dirname, exists, join, realpath

from packaging.version import parse as version_parse

if version_parse(platform.python_version()) < version_parse("3.9"):
    print("[FAIL] We recommend Python 3.9 or newer but" " found version %s" % (sys.version))
else:
    print("[OK] Your Python version is %s" % (platform.python_version()))


def get_packages(pkgs):
    versions = []
    for p in pkgs:
        try:
            imported = import_module(p)
            try:
                version = (
                    getattr(imported, "__version__", None)
                    or getattr(imported, "version", None)
                    or getattr(imported, "version_info", None)
                )
                if version is None:
                    # If common attributes don"t exist, use importlib.metadata
                    version = importlib.metadata.version(p)
                versions.append(version)
            except PackageNotFoundError:
                # Handle case where package is not installed
                versions.append("0.0")
        except ImportError:
            # Fallback if importlib.import_module fails for unexpected reasons
            versions.append("0.0")
    return versions


def get_requirements_dict():
    PROJECT_ROOT = dirname(realpath(__file__))
    PROJECT_ROOT_UP_ONE = dirname(PROJECT_ROOT)
    TOML_FILE = join(PROJECT_ROOT_UP_ONE, "pyproject.toml")
    
    if not exists(TOML_FILE):
        return {}

    d = {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
            
        with open(TOML_FILE, "rb") as f:
            toml_data = tomllib.load(f)
            dependencies = toml_data.get("project", {}).get("dependencies", [])
            for dep in dependencies:
                # Basic parsing for "name>=version" or "name>=version,<version"
                # This is simplified but covers current project needs
                import re
                match = re.match(r"([a-zA-Z0-9_\-]+)\s*(>=|==|<|>|<=)\s*([0-9\.]+)(?:,\s*(<|>|<=|>=|==)\s*([0-9\.]+))?", dep)
                if match:
                    name = match.group(1)
                    lower = match.group(3)
                    upper = match.group(5)
                    if upper:
                        d[name] = (lower, upper)
                    else:
                        d[name] = lower
    except Exception:
        # Fallback to manual parsing if toml parser is not available or fails
        with open(TOML_FILE, "r") as f:
            in_dependencies = False
            for line in f:
                if "dependencies =" in line:
                    in_dependencies = True
                    continue
                if in_dependencies:
                    if "]" in line:
                        break
                    line = line.strip().strip(",").strip('"').strip("'")
                    if not line:
                        continue
                    # Simple split on >= or ==
                    import re
                    match = re.split(r">=|==|<", line)
                    if len(match) >= 2:
                        name = match[0].strip()
                        version = match[1].strip()
                        # Check for upper bound
                        if "," in version:
                            version = version.split(",")[0].strip()
                        d[name] = version
    return d


def check_packages(d):
    versions = get_packages(d.keys())

    for (pkg_name, suggested_ver), actual_ver in zip(d.items(), versions):
        if isinstance(suggested_ver, tuple):
            lower, upper = suggested_ver[0], suggested_ver[1]
        else:
            lower = suggested_ver
            upper = None
        if actual_ver == "N/A":
            continue
        actual_ver = version_parse(actual_ver)
        lower = version_parse(lower)
        if upper is not None:
            upper = version_parse(upper)
        if actual_ver < lower and upper is None:
            print(f"[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {lower}")
        elif actual_ver < lower:
            print(f"[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {lower} and < {upper}")
        elif upper is not None and actual_ver >= upper:
            print(f"[FAIL] {pkg_name} {actual_ver}, please downgrade to >= {lower} and < {upper}")
        else:
            print(f"[OK] {pkg_name} {actual_ver}")


def main():
    d = get_requirements_dict()
    check_packages(d)


if __name__ == "__main__":
    main()
