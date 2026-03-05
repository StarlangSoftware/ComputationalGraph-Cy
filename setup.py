from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")


def discover_extensions() -> list[Extension]:
    """Build one extension per .pyx file under ComputationalGraph/."""
    extensions: list[Extension] = []
    for source in (ROOT / "ComputationalGraph").rglob("*.pyx"):
        module_name = ".".join(source.relative_to(ROOT).with_suffix("").parts)
        extensions.append(Extension(module_name, [str(source)]))
    return extensions


setup(
    name="NlpToolkit-ComputationalGraph",
    version="1.0.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    url="https://github.com/StarlangSoftware/ComputationalGraph-Py",
    license="",
    author="olcaytaner",
    author_email="olcay.yildiz@ozyegin.edu.tr",
    description="Computational Graph library",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=["NlpToolkit-Math"],
    ext_modules=cythonize(
        discover_extensions(),
        compiler_directives={"language_level": "3"},
    ),
)
