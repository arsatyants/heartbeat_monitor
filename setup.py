from setuptools import setup, find_packages

setup(
    name="heartbeat_monitor",
    version="0.1.0",
    description="rPPG heartbeat detection via Raspberry Pi IMX500 camera",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "opencv-python>=4.8",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "heartbeat-monitor=main:run",
        ]
    },
)
