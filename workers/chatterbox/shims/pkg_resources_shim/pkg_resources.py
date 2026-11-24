"""Minimal `pkg_resources` subset shim for resemble-perth.

Setuptools 81+ removed the vendored `pkg_resources` package, which broke
resemble-perth's only use of it: `resource_filename("perth.perth_net",
"pretrained")`. This module provides the `resource_*` helpers on top of
`importlib.resources`, the documented replacement. It intentionally does not
emulate distributions, entry points, or working-set APIs.
"""

from importlib import resources as _resources

__all__ = [
    "resource_bytes",
    "resource_exists",
    "resource_filename",
    "resource_isdir",
    "resource_listdir",
    "resource_stream",
]


def resource_filename(package_or_requirement, resource_name):
    return str(_resources.files(package_or_requirement) / resource_name)


def resource_exists(package_or_requirement, resource_name):
    return (_resources.files(package_or_requirement) / resource_name).exists()


def resource_isdir(package_or_requirement, resource_name):
    return (_resources.files(package_or_requirement) / resource_name).is_dir()


def resource_listdir(package_or_requirement, resource_name):
    return sorted(
        resource.name for resource in (_resources.files(package_or_requirement) / resource_name).iterdir()
    )


def resource_stream(package_or_requirement, resource_name):
    return (_resources.files(package_or_requirement) / resource_name).open("rb")


def resource_bytes(package_or_requirement, resource_name):
    return (_resources.files(package_or_requirement) / resource_name).read_bytes()
