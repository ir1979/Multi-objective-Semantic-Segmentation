from typing import List

from setuptools import find_packages, setup


def _read_requirements() -> List[str]:
    requirements_path = "requirements.txt"
    with open(requirements_path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="building-segmentation-moo",
    version="1.0.0",
    description="Multi-objective optimization for building segmentation.",
    packages=find_packages(),
    python_requires=">=3.7,<3.8",
    install_requires=_read_requirements(),
)
