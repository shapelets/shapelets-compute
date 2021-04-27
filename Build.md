# Build Instructions



## Windows Pre-Requisites
Ensure you have:

- git 
- python 3.8
- cmake
- 7 zip
- Windows 10 SDK or Visual Studio with C++/CLI
- Optional components:
    - VCPKG, whose root can be found on environment VCPKG_ROOT. 
    - ArrayFire installation

### Chocolatey setup 

    choco install git cmake 7zip
    choco install python --version=3.8.0


## Instructions
1. Download the latest version of the repository by executing:

    git clone --recurse-submodules https://gitlab.com/shapelets/shapelets/python-client.git

2. Ensure all python dependencies are present in your environment: ```pip install -r requirements.txt```

3. Run ```setup.py```.  
If you planning to develop, use the ```develop``` option, that is, ```python setup.py develop```, which will set up the necessary entries in the pip library folders to point to your development environment.  Once setup with ```develop``` option, further builds and changes will be automatically picked up.

4. Run tests to ensure everything is running as expected by issuing ```pytest``` on the root folder.





    
