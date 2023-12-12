from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="multicam_birdseye",
    version="1.0.0",
    description="Birds Eye View from Multiple Vehicle Camera Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayushgoel24/Birds-Eye-View-from-Multiple-Vehicle-Camera-Images",
    author="Ayush Goel",
    author_email="ayush.goel2427@gmail.com",
    # package_dir={"": "src"},  # Optional
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'seaborn',
        'numpy',
        'pandas',
        'tqdm'
    ]
)