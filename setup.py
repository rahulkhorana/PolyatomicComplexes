from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="polyatomic_complexes",
        version="1.0.0",
        packages=find_packages(),
        install_requirements=["torch", "pandas", "numpy"],
    )
