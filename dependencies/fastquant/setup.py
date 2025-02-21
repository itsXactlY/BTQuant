import setuptools

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="fastquant",
    version="0.1.8.2",
    author="Lorenzo Ampil",
    author_email="lorenzo.ampil@gmail.com",
    description="Bringing data driven investments to the mainstream",
    long_description_content_type="text/markdown",
    url="https://github.com/enzoampil/fastquant",
    packages=setuptools.find_packages(exclude=["docs", "tests"]),
    package_data={
        "fastquant": ["data/*"],
        "fastquant.strategys": ["*"],
        "fastquant.strategys.*": ["*"]
        "fastquant.strategys/custom_indicators.*": ["*"]
    },
    include_package_data=True,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)