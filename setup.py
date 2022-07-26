import setuptools

packages = setuptools.find_packages(where = "./Functions")

setuptools.setup(
        name = "AnalysisTopGNN", 
        version = "1.0", 
        packages = packages, 
        long_description = open("README.md").read(), 
        package_dir = {pkg : f"Functions/{pkg.replace('.', '/')}" for pkg in packages}, 
    )

