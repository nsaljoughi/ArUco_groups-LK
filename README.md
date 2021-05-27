# Augmented reality module with ArUco markers
<p>This repository contains the code of a C++ module for augmented reality with groups of ArUco markers. </p>
<p>The aim of the module is to develop a method that estimates the camera's pose as accurately as possible and that is able to deal with different issues that can arise in the real world. The method is based on the combination of different markers and their tracking in time to stabilize the estimated pose; when all markers are lost we exploit FAST features tracking to estimate the LK optical flow. </p>

## Structure of the repo
+ **src/**          main and header files
+ **utils/**        contains scripts for calibration, markers' generation and plotting
+ **old_versions/** old versions of the main script
+ **graphs/**       schemes of the control logic
+ **videos/**       videos for testing (at the moment just one example)

## How to compile
After changing the paths in CMakeLists.txt, make a build directory and compile:
```
mkdir build && cd build
cmake -GNinja ..
ninja
```
OR
```
cmake ..
make -j8
```
## Example
```
./ar_stable -c=2160x3840.yml -l=0.54 -o=0.04 -d=11 -u=1
```

## Graph of the control logic
![plot](./graphs/scheme1.jpg)
