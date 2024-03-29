cmake_minimum_required(VERSION 2.8)
project(PamaAR)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
CMAKE_POLICY(SET CMP0072 NEW)

set("OpenCV_DIR" "/path/to/opencv/build/")
set("EIGEN3_INCLUDE_DIR" "/path/to/eigen")
set("CMAKE_MODULE_PATH" "/path/to/eigen/cmake")

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

file(GLOB ar_stab
	"src/ar_stable.cpp"
	"src/*.h"
)

add_executable(create_board "utils/create_board.cpp")
add_executable(ar_stable ${ar_stab})
target_link_libraries(ar_stable ${OpenCV_LIBS} ${OPENGL_LIBS} ${GLUT_LIBS})
target_link_libraries(create_board ${OpenCV_LIBS} ${OPENGL_LIBS} ${GLUT_LIBS})
