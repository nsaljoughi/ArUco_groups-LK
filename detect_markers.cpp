/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection";
const char* keys  =
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
	"{f        |       | Use stabilization filtering (=1) or not (=0) }";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

static Mat cvcloud_load()
{
    Mat cloud(1, 7708, CV_64FC3);
    ifstream ifs("arrow.ply");

    string str;
    for(size_t i = 0; i < 13; ++i)
        getline(ifs, str);

    Point3d* data = cloud.ptr<cv::Point3d>();
    float dummy1, dummy2, dummy3;
    for(size_t i = 0; i < 7708; ++i)
        ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2 >> dummy3;

    //cloud *= 5.0f;
    return cloud;
}

// Transform Rodrigues rotation vector into a quaternion
Vec4d vec2quat(Vec3d vec) {
	Vec4d q;
	double ang = sqrt( vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] );
	q[0] = vec[0] / ang * sin(ang / 2);
	q[1] = vec[1] / ang * sin(ang / 2);
	q[2] = vec[2] / ang * sin(ang / 2);
	q[3] = cos(ang / 2);

	return q;
}

// Transform quaternion into a Rodrigues rotation vector
Vec3d quat2vec(Vec4d quat) {
	Vec3d v;
	double ang = 2*acos(quat[3]);
	v[0] = quat[0] / sqrt(1 - quat[3]*quat[3]) * ang;
	v[1] = quat[1] / sqrt(1 - quat[3]*quat[3]) * ang;
	v[2] = quat[2] / sqrt(1 - quat[3]*quat[3]) * ang;
	
	return v;
}

// Compute average of two quaternions using 
// eq. (18) of https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf, 
// that provides a closed-form solution for averaging two quaternions
Vec4d avgQuat(Vec4d q1, Vec4d q2, double w1 = 1, double w2 = 1) {
	Vec4d q3;
	double zed = sqrt( (w1-w2)*(w1-w2) + 4*w1*w2*(q1.dot(q2))*(q1.dot(q2)) );
	q3 = ((w1-w2+zed)*q1 + 2*w2*(q1.dot(q2))*q2);
	double norm = sqrt( q3[0]*q3[0] + q3[1]*q3[1] + q3[2]*q3[2] + q3[3]*q3[3] );
	q3 = q3 / norm;
	return q3;
}

void generateMarkers() {
    Mat marker1, marker2, marker3, marker4;
    Ptr<aruco::Dictionary> dict1 = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::Dictionary> dict2 = aruco::getPredefinedDictionary(aruco::DICT_7X7_250);
    aruco::drawMarker(dict1, 1, 300,  marker1, 1);
    aruco::drawMarker(dict1, 1, 400, marker2, 1);
    aruco::drawMarker(dict2, 1, 300, marker3, 1);
    aruco::drawMarker(dict2, 1, 400, marker4, 1);
    imwrite("marker1.png", marker1);
    imwrite("marker2.png", marker2);
    imwrite("marker3.png", marker3);
    imwrite("marker4.png", marker4);
}

/**
 */
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }
    
    Mat bunny_cloud = cvcloud_load();

    bunny_cloud = 0.1 * bunny_cloud;

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");
    bool stabilFilt = parser.has("f");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

    int camId = parser.get<int>("ci");

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    
    
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }
    

    VideoCapture inputVideo;//("/home/nicola/Desktop/pama_ar/video.mp4");
    
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
        // cout << "success" << endl;
    } else {
        inputVideo.open(0);
        waitTime = 10;
        // cout << "failed" << endl;
    }


    if(!inputVideo.isOpened()) {
    	cout << "Video could not be opened..." << endl;
    	return -1;
    }

    int frame_width = inputVideo.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = inputVideo.get(CAP_PROP_FRAME_HEIGHT);

    double totalTime = 0;
    int totalIterations = 0;

    vector<Point3d> rectangle3D;
    vector<Point2d> rectangle2D, rectangle2D2, rectangle2D3;

    rectangle3D.push_back(Point3d(-0.2, 0.1, 0));
    rectangle3D.push_back(Point3d(0.2, 0.1, 0));
    rectangle3D.push_back(Point3d(0.2, 0.3, 0));
    rectangle3D.push_back(Point3d(-0.2, 0.3, 0));

    Vec3d rvec_store, tvec_store, tvec2_store, tvec3_store;
    int frame_id = 0;
    int lost_id = 0;

    // save to video
    VideoWriter cap("demo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
		    inputVideo.get(CAP_PROP_FPS), Size(frame_width, frame_height));
    // save results to file
    ofstream resultfile;
    if (stabilFilt) {
	    resultfile.open("results_filt.txt");
	    if (resultfile.is_open()) {
		    resultfile << "Filtered resulting transformations \n";
	    }
	    else cout << "Unable to open result file" << endl;
    }
    else {
	    resultfile.open("results_unfilt.txt");
	    if (resultfile.is_open()) {
		    resultfile << "Unfiltered resulting transformations \n";
	    }
	    else cout << "Unable to open result file" << endl;
    }

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);
        
	cout << "Frame " << frame_id << endl;

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;
	Vec3d tvecs2, tvecs3; 

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
	
        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

	// parameters for transformation weighting
	double alpha = 0.5;
	double alpha2 = 0.7;

        // draw results
        image.copyTo(imageCopy);
 
        if(ids.size() > 0) {
	    lost_id = 0; // after a while, let go of history
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if(estimatePose) {
                
                for(unsigned int i = 0; i < ids.size(); i++)
                {
		    // If the video just started, no previous data...	
		    if (frame_id == 0) {
			    rvec_store = rvecs[i];
			    tvec_store = tvecs[i];
		    }
		    

		    // Display some infos
		    Vec4d q1 = vec2quat(rvecs[i]);
		    Vec4d q2 = vec2quat(rvec_store);
		    Vec4d q3 = avgQuat(q1, q2);
		    Vec3d v3 = quat2vec(q3);

		    Vec3d diff = tvec_store - tvecs[i];
		    Vec4d diff_rot = q2 - q1;
		    float diff_mag = sqrt( diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] );
		    float diff_rot_mag = sqrt( diff_rot[0]*diff_rot[0] + diff_rot[1]*diff_rot[1] + 
				    diff_rot[2]*diff_rot[2] + diff_rot[3]*diff_rot[3] );

		    if (!stabilFilt && resultfile.is_open()) {
			    resultfile << frame_id << "," << diff_rot_mag << "," << diff_mag << "\n";
		    }
		  
		    cout << "x(t-1) = " << tvec_store << endl;
		    cout << "x(t) = " << tvecs[i] << endl;
		    cout << "r(t-1) = " << rvec_store << endl;
		    cout << "r(t) = " << rvecs[i] << endl;
		    cout << "q(t-1) = " << q2 << endl;
		    cout << "q(t) = " << q1 << endl;
		    cout << "||x(t-1) - x(t)|| = " << diff_mag << endl;
		    cout << "||q(t-1) - q(t)|| = " << diff_rot_mag << endl;

		    if (stabilFilt) {
			    if (diff_mag > 0.1 || diff_mag < 0.0001 || diff_rot_mag > 1.5) {
				    rvecs[i] = rvec_store;
				    tvecs[i] = tvec_store;
			    }
			    
			    // Use quaternions to make a weighted average of rotations
			    Vec4d quat1, quat2, quat_avg;
			    quat1 = vec2quat(rvecs[i]);
			    quat2 = vec2quat(rvec_store);
			    quat_avg = avgQuat(quat1, quat2, (1-alpha2), alpha2);
			    rvecs[i] = quat2vec(quat_avg);
			    
			    for (int j=0; j<3; j++) {
				    tvecs[i][j] = (1.0f - alpha)*tvecs[i][j] + alpha*tvec_store[j];
			    } 
			    
			    cout << "(1-alpha)*x(t-1) + alpha*x(t) = " << tvecs[i] << endl;
			    cout << "(1-alpha)*r(t-1) + alpha*r(t) = " << rvecs[i] << endl;
		    }
		    
		    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    markerLength * 0.5f);

		    // Display two more arrows to test offset
		    tvecs2[0] = tvecs[i][0] + 0.3;
		    tvecs2[1] = tvecs[i][1];
		    tvecs2[2] = tvecs[i][2];
		    tvecs3[0] = tvecs[i][0] - 0.35;
		    tvecs3[1] = tvecs[i][1];
		    tvecs3[2] = tvecs[i][2];

                    projectPoints(bunny_cloud, rvecs[i], tvecs[i], camMatrix, distCoeffs, rectangle2D);
		    projectPoints(bunny_cloud, rvecs[i], tvecs2, camMatrix, distCoeffs, rectangle2D2);
		    projectPoints(bunny_cloud, rvecs[i], tvecs3, camMatrix, distCoeffs, rectangle2D3);

                    for (unsigned int j = 0; j < rectangle2D.size(); j++)
                        circle(imageCopy, rectangle2D[j], 1, Scalar(255,0,0), -1);
		    for (unsigned int j = 0; j < rectangle2D2.size(); j++)
                        circle(imageCopy, rectangle2D2[j], 1, Scalar(0,255,0), -1);
		    for (unsigned int j = 0; j < rectangle2D3.size(); j++)
                        circle(imageCopy, rectangle2D3[j], 1 , Scalar(0,0,255), -1);

                    cout << "R: " << rvecs[i] << "; T: " << tvecs[i] << "\n" << endl; // the translation is with respect to the principal point
		    if (stabilFilt) {
			    q1 = vec2quat(rvecs[i]);
			    q2 = vec2quat(rvec_store);
			    diff = tvec_store - tvecs[i];
			    diff_rot = q2 - q1;
			    diff_mag = sqrt( diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] );
			    diff_rot_mag = sqrt( diff_rot[0]*diff_rot[0] + diff_rot[1]*diff_rot[1] + 
					    diff_rot[2]*diff_rot[2] + diff_rot[3]*diff_rot[3] );
			    
			    if (resultfile.is_open()) {
				    resultfile << frame_id << "," << diff_rot_mag << "," << diff_mag << "\n";
			    }
		    }

		    // Display some infos on video frame
		    string txt0, txt1, txt2, txt3;
		    float dist1, dist2, dist3;
		    // Compute distances of objects from camera
		    dist1 = sqrt( tvecs[i][0]*tvecs[i][0] + tvecs[i][1]*tvecs[i][1] + tvecs[i][2]*tvecs[i][2] );
		    dist2 = sqrt( tvecs2[0]*tvecs2[0] + tvecs2[1]*tvecs2[1] + tvecs2[2]*tvecs2[2] );
		    dist3 = sqrt( tvecs3[0]*tvecs3[0] + tvecs3[1]*tvecs3[1] + tvecs3[2]*tvecs3[2] );
		    txt0 = "Marker distance: " + to_string(dist1) + " m";
		    txt1 = "First arrow: " + to_string(dist1) + " m";
		    txt2 = "Second arrow: " +  to_string(dist2) + " m";
		    txt3 = "Third arrow: " +  to_string(dist3) + " m";
		    putText(imageCopy, txt0, Point2f(50, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8, false);
		    putText(imageCopy, txt1, Point2f(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8, false);
		    putText(imageCopy, txt2, Point2f(50, 140), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, 8, false);
		    putText(imageCopy, txt3, Point2f(50, 180), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8, false);


		    // Save the computed estimation for the next frame
		    rvec_store = rvecs[i];
		    tvec_store = tvecs[i];
		    tvec2_store = tvecs2;
		    tvec3_store = tvecs3;
                }
            }
        }

	else if (ids.size() == 0) {
		//if (lost_id > 5) {
		//	break;
		//}
		//else {
			lost_id += 1;
			cout << "Lost since " << lost_id << endl;
			
			projectPoints(bunny_cloud, rvec_store, tvec_store, camMatrix, distCoeffs, rectangle2D);
			projectPoints(bunny_cloud, rvec_store, tvec2_store, camMatrix, distCoeffs, rectangle2D2);
			projectPoints(bunny_cloud, rvec_store, tvec3_store, camMatrix, distCoeffs, rectangle2D3);
			
			if (stabilFilt && resultfile.is_open()) {
				resultfile << frame_id << "," << float(0.0) << "," << float(0.0) << "\n";
			}
			if (!stabilFilt && resultfile.is_open()) {
				resultfile << frame_id << "," << float(0.0) << "," << float(0.0) << "\n";
			}

			for (unsigned int j = 0; j < rectangle2D.size(); j++)
				circle(imageCopy, rectangle2D[j], 1, Scalar(255,0,0), -1);
			for (unsigned int j = 0; j < rectangle2D2.size(); j++)
				circle(imageCopy, rectangle2D2[j], 1, Scalar(0,255,0), -1);
			for (unsigned int j = 0; j < rectangle2D3.size(); j++)
				circle(imageCopy, rectangle2D3[j], 1 , Scalar(0,0,255), -1);
		//}
	}

	frame_id += 1;

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));
	
	cap.write(imageCopy);

        imshow("out", imageCopy);


        char key = (char)waitKey(waitTime);
        if(key == 27) break;

        
    }
    inputVideo.release();
    cap.release();
    
    resultfile.close();

    return 0;
}
