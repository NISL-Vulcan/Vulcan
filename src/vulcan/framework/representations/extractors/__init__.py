import contextlib
import functools
import itertools
import shutil
import subprocess
import warnings
from typing import Optional

try:
    from .extractors import ClangDriver, LLVM_VERSION  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ClangDriver = None  # type: ignore[assignment]
    LLVM_VERSION = "unknown"
    _EXTENSION_AVAILABLE = False
else:
    _EXTENSION_AVAILABLE = True


@functools.lru_cache()
def clang_binary_path():
    """Find the clang compiler binary, trying to match the version that the native extension was compiled with.

    Prints a warning if the binary that was found is a different version.
    Raises RuntimeException if no clang compiler is found at all.
    """
    if not _EXTENSION_AVAILABLE:
        raise ModuleNotFoundError(
            "native extension 'vulcan.framework.representations.extractors.extractors' is not available; "
            "build/install it to enable clang/LLVM extractors."
        )

    best_score = None
    best_version = None
    best_path = None

    components = LLVM_VERSION.split('.')
    for suffix_len in reversed(range(len(components) + 1)):
        suffix = '.'.join(components[:suffix_len])
        path = shutil.which(f"clang-{suffix}" if suffix else "clang")
        if path is None:
            continue

        llvm_config_path = subprocess.run(
            [path, "-print-prog-name=llvm-config"],
            check=True, stdout=subprocess.PIPE
        ).stdout.decode().strip()

        this_version = subprocess.run(
            [llvm_config_path, "--version"],
            check=True, stdout=subprocess.PIPE
        ).stdout.decode().strip()

        if this_version == LLVM_VERSION:
            return path

        score = sum(1 for _ in itertools.takewhile(lambda x: x[0] == x[1], zip(this_version, LLVM_VERSION)))
        if best_score is None or score > best_score:
            best_score = score
            best_version = this_version
            best_path = path

    if not best_path:
        raise RuntimeError("cannot find clang compiler binary in PATH")

    warnings.warn(f"found clang compiler at {best_path} for LLVM {best_version}, but native extension was compiled "
                  f"against LLVM {LLVM_VERSION}")
    return best_path


@contextlib.contextmanager
def clang_driver_scoped_options(clang_driver, additional_include_dir: Optional[str] = None, filename: Optional[str] = None):
    """A context manager to set and restore options for the clang driver in a local scope.

    >>> with clang_driver_scoped_options(clang_driver, filename="foo"):
    ...     # clang_driver's file name is set to foo in this scope
    ...     pass
    ... # filename is restored to previous value after the with block
    """
    prev_filename = None
    if filename is not None:
        prev_filename = clang_driver.getFileName()
        clang_driver.setFileName(filename)

    if additional_include_dir:
        clang_driver.addIncludeDir(
            additional_include_dir, ClangDriver.IncludeDirType.User
        )

    try:
        yield clang_driver
    finally:
        if prev_filename is not None:
            clang_driver.setFileName(prev_filename)
        if additional_include_dir:
            clang_driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

