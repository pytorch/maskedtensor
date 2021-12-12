import setuptools
import distutils.command.clean
import shutil
import os
import glob
import subprocess
import io


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


version = "0.0.3"
package_name = "maskedtensor"

cwd = os.path.dirname(os.path.abspath(__file__))

print("Building wheel {}-{}".format(package_name, version))


readme = open("README.md").read()


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


# Commented out sections we may need later on to enable C++ extension
setuptools.setup(
    name=package_name,
    version=version,
    author="Christian Puhrsch",
    author_email="cpuhrsch@fb.com",
    description="MaskedTensors for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/cpuhrsch/maskedtensor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
    cmdclass={
        "clean": clean,
        # "build_ext": BuildExtension.with_options(no_python_abi_suffix=True,),
    },
    # install_requires=requirements,
    # ext_modules=get_extensions(),
)
