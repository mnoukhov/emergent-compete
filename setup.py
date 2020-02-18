from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = []
    for line in f:
        if not line.startswith('-f'):
            requirements.append(line)

with open('notebook_requirements.txt') as f:
    notebook_reqs = []
    for line in f:
        if not line.startswith('-f'):
            notebook_reqs.append(line)

setup(
    name = 'emergent-compete',
    version = '0.0.1',
    packages = find_packages(),
    install_requires=requirements,
    extras_require={'notebook': notebook_reqs},
)
