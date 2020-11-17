# How-to-use-OpenBLAS-in-Microsoft-Visual-Studio

https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio

## download 
https://github.com/xianyi/OpenBLAS/releases
OpenBLAS-0.3.12.tar.gz

## prepair

- open anaconda command prompt
conda update -n base conda
conda config --add channels conda-forge
conda install -y cmake flang clangdev perl libflang ninja

"c:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvars64.bat"

or

"c:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat"

tar xf OpenBLAS-0.3.12.tar.gz

set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
mkdir build
cd build

- arg "--DMSVC_STATIC_CRT=ON" is  to use static crt.     
    cl /MT  
    is to use static libc   

    cl /MD 
    is to use msvcrt.dll   

cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release -DMSVC_STATIC_CRT=ON

cmake --build . --config Release

## install
cmake --install . --prefix c:\opt -v

## Test
tes.c
```
    #include <cblas.h>
    #include <stdio.h>
    void main()
    {
        int i=0;
        double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};         
        double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};  
        double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5}; 
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);
        for(i=0; i<9; i++)
            printf("%lf ", C[i]);
        printf("\n");
    }
```

## compile
cl /MT tes.c -Ic:\opt\include\openblas build\lib\Release\openblas.lib

