from setuptools import setup

setup(
    name="pnm",
    version="0.0.1",
    packages=["pnm"],
    include_package_data=True,
    url="https://github.com/pstwh/pnm",
    keywords="phonetics, machine learning, neural network, english, speech, practice",
    package_data={"pnm": ["artifacts/*.onnx", "artifacts/*.bin", "artifacts/*.parquet"]},
    python_requires=">=3.5, <4",
    install_requires=[
        "pandas==2.2.3",
        "scipy==1.13.1",
        "pyaudio==0.2.14",
        "pyarrow==18.1.0",
    ],
    extras_require={
        "cpu": [
            "onnxruntime==1.18.1",
        ],
        "gpu": [
            "onnxruntime-gpu==1.18.1",
        ],
    },
    entry_points={
        "console_scripts": ["pnm=pnm.cli:main"],
    },
    description="Convert audio to phonetic text and practice improving your speech accent.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
