LDIR_OPENCV=-L/home/fabio/Libraries/opencv-3.3.0_w_contrib/release/installed/lib
LIBS_OPENCV=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_video -lopencv_calib3d -lopencv_aruco -lopencv_videoio
CFLAGS_OPENCV=-I/home/fabio/Libraries/opencv-3.3.0_w_contrib/release/installed/include

CXX_FLAGS=-std=c++11 -g

detect_markers: detect_markers.o
	c++ $(CXX_FLAGS) -o detect_markers detect_markers.o $(LDIR_OPENCV) $(LIBS_OPENCV)

detect_markers.o:
	c++ $(CXX_FLAGS) -c detect_markers.cpp $(CFLAGS_OPENCV)

clean:
	rm -f detect_markers *.o
	
