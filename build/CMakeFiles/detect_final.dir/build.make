# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nico/pama_marker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nico/pama_marker/build

# Include any dependencies generated for this target.
include CMakeFiles/detect_final.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detect_final.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detect_final.dir/flags.make

CMakeFiles/detect_final.dir/detect_markers_final.cpp.o: CMakeFiles/detect_final.dir/flags.make
CMakeFiles/detect_final.dir/detect_markers_final.cpp.o: ../detect_markers_final.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nico/pama_marker/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detect_final.dir/detect_markers_final.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect_final.dir/detect_markers_final.cpp.o -c /home/nico/pama_marker/detect_markers_final.cpp

CMakeFiles/detect_final.dir/detect_markers_final.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect_final.dir/detect_markers_final.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nico/pama_marker/detect_markers_final.cpp > CMakeFiles/detect_final.dir/detect_markers_final.cpp.i

CMakeFiles/detect_final.dir/detect_markers_final.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect_final.dir/detect_markers_final.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nico/pama_marker/detect_markers_final.cpp -o CMakeFiles/detect_final.dir/detect_markers_final.cpp.s

# Object files for target detect_final
detect_final_OBJECTS = \
"CMakeFiles/detect_final.dir/detect_markers_final.cpp.o"

# External object files for target detect_final
detect_final_EXTERNAL_OBJECTS =

detect_final: CMakeFiles/detect_final.dir/detect_markers_final.cpp.o
detect_final: CMakeFiles/detect_final.dir/build.make
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_gapi.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_stitching.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_alphamat.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_aruco.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_bgsegm.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_bioinspired.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_ccalib.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_dnn_objdetect.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_dnn_superres.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_dpm.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_face.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_freetype.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_fuzzy.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_hfs.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_img_hash.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_intensity_transform.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_line_descriptor.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_mcc.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_quality.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_rapid.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_reg.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_rgbd.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_saliency.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_stereo.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_structured_light.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_superres.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_surface_matching.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_tracking.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_videostab.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_viz.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_wechat_qrcode.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_xfeatures2d.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_xobjdetect.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_xphoto.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_shape.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_highgui.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_datasets.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_plot.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_text.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_ml.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_phase_unwrapping.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_optflow.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_ximgproc.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_video.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_videoio.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_dnn.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_imgcodecs.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_objdetect.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_calib3d.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_features2d.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_flann.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_photo.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_imgproc.so.4.5.1
detect_final: /home/nico/packages/opencv/opencv-master/build/lib/libopencv_core.so.4.5.1
detect_final: CMakeFiles/detect_final.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nico/pama_marker/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable detect_final"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detect_final.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detect_final.dir/build: detect_final

.PHONY : CMakeFiles/detect_final.dir/build

CMakeFiles/detect_final.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detect_final.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detect_final.dir/clean

CMakeFiles/detect_final.dir/depend:
	cd /home/nico/pama_marker/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nico/pama_marker /home/nico/pama_marker /home/nico/pama_marker/build /home/nico/pama_marker/build /home/nico/pama_marker/build/CMakeFiles/detect_final.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detect_final.dir/depend

