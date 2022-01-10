from setuptools import setup

setup(
    name="swin_transformer_v2",
    version="0.1",
    url="https://github.com/ChristophReich1996/Swin-Transformer-V2",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="PyTorch Swin Transformer V2",
    packages=["swin_transformer_v2"],
    install_requires=["torch>=1.7.0", "timm>=0.4.12"],
)
