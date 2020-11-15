
set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=OFF -DCMAKE_BUILD_TYPE=Release  -DMSVC_STATIC_CRT=ON
