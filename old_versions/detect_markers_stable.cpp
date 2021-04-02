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

#include </home/nico/packages/eigen/Eigen/SVD>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

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
	"{o        | 0.01  | Offset between markers (in meters) }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
	"{f        |       | Use stabilization filtering}"
	"{s        |       | Save results}";
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

/// Compute average of two quaternions using 
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

// Method to find the average of a set of rotation quaternions using Singular Value Decomposition
/*
 * The algorithm used is described here:
 * https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
 */
Eigen::Vector4f quaternionAverage(std::vector<Eigen::Vector4f> quaternions)
{
	if (quaternions.size() == 0)
	{
		std::cerr << "Error trying to calculate the average quaternion of an empty set!\n";
		return Eigen::Vector4f::Zero();
	}

	// first build a 4x4 matrix which is the elementwise sum of the product of each quaternion with itself
	Eigen::Matrix4f A = Eigen::Matrix4f::Zero();

	for (unsigned long int q=0; q<quaternions.size(); ++q)
		A += quaternions[q] * quaternions[q].transpose();

	// normalise with the number of quaternions
	A /= quaternions.size();

	// Compute the SVD of this 4x4 matrix
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

	Eigen::VectorXf singularValues = svd.singularValues();
	Eigen::MatrixXf U = svd.matrixU();

	// find the eigen vector corresponding to the largest eigen value
	int largestEigenValueIndex;
	float largestEigenValue;
	bool first = true;

	for (int i=0; i<singularValues.rows(); ++i)
	{
		if (first)
		{
			largestEigenValue = singularValues(i);
			largestEigenValueIndex = i;
			first = false;
		}
		else if (singularValues(i) > largestEigenValue)
		{
			largestEigenValue = singularValues(i);
			largestEigenValueIndex = i;
		}
	}

	Eigen::Vector4f average;
	average(0) = U(0, largestEigenValueIndex);
	average(1) = U(1, largestEigenValueIndex);
	average(2) = U(2, largestEigenValueIndex);
	average(3) = U(3, largestEigenValueIndex);

	return average;
}

// Transform a relative translation into absolute
// (useful for augmented reality when we have offset wrt marker frame)
Vec3d transformVec(Vec3d vec, Vec3d rotvec, Vec3d tvec) {
	Mat rotationMat = Mat::zeros(3, 3, CV_64F);
	Mat transformMat = Mat::eye(4, 4, CV_64F);
	Rodrigues(rotvec, rotationMat); //convert rodrigues angles into rotation matrix
	
	//build transformation matrix
	for (int i=0; i<3; i++) {
		transformMat.at<double>(i,3) = tvec[i];
		for (int j=0; j<3; j++) {
			transformMat.at<double>(i,j) = rotationMat.at<double>(i,j);
		}
	}
	
	Vec4d vechomo; //vec in homogeneous coordinates, i.e. <x,y,z,1>
	vechomo[0] = vec[0];
	vechomo[1] = vec[1];
	vechomo[2] = vec[2];
	vechomo[3] = 1.0;

	Vec3d vectrans;	//output, vector transformed
       	Mat vectransMat = transformMat*Mat(vechomo);
	vectrans[0] = vectransMat.at<double>(0);
	vectrans[1] = vectransMat.at<double>(1);
	vectrans[2] = vectransMat.at<double>(2);

	return vectrans;
}

// Generate markers used
void generateMarkers() {
    Mat marker1, marker2, marker3, marker4, marker5, marker6, marker7, marker8, marker9, marker10, marker11, marker12, marker13, marker14, marker15, marker16;
    Ptr<aruco::Dictionary> dict1 = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::drawMarker(dict1, 1, 200, marker1, 1);
    aruco::drawMarker(dict1, 2, 200, marker2, 1);
    aruco::drawMarker(dict1, 3, 200, marker3, 1);
    aruco::drawMarker(dict1, 4, 200, marker4, 1);
    aruco::drawMarker(dict1, 5, 200, marker5, 1);
    aruco::drawMarker(dict1, 6, 200, marker6, 1);
    aruco::drawMarker(dict1, 7, 200, marker7, 1);
    aruco::drawMarker(dict1, 8, 200, marker8, 1);
    aruco::drawMarker(dict1, 9, 200, marker9, 1);
    aruco::drawMarker(dict1, 10, 200, marker10, 1);
    aruco::drawMarker(dict1, 11, 200, marker11, 1);
    aruco::drawMarker(dict1, 12, 200, marker12, 1);	
    aruco::drawMarker(dict1, 13, 200, marker13, 1);
    aruco::drawMarker(dict1, 14, 200, marker14, 1);
    aruco::drawMarker(dict1, 15, 200, marker15, 1);
    aruco::drawMarker(dict1, 16, 200, marker16, 1);	

    imwrite("marker1.png", marker1);
    imwrite("marker2.png", marker2);
    imwrite("marker3.png", marker3);
    imwrite("marker4.png", marker4);
    imwrite("marker5.png", marker5);
    imwrite("marker6.png", marker6);
    imwrite("marker7.png", marker7);
    imwrite("marker8.png", marker8);
    imwrite("marker9.png", marker9);
    imwrite("marker10.png", marker10);
    imwrite("marker11.png", marker11);
    imwrite("marker12.png", marker12);
    imwrite("marker13.png", marker13);
    imwrite("marker14.png", marker14);
    imwrite("marker15.png", marker15);
    imwrite("marker16.png", marker16);
}

/**
 */
int main(int argc, char *argv[]) {
    generateMarkers();
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    // Messing with time a bit
    time_t now = time(0);
    cout << "Now we are at: " << now << endl;
    
    Mat bunny_cloud = cvcloud_load();
   
    //bunny_cloud = 0.5 * bunny_cloud;

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");
    float markerOffset = parser.get<float>("o");
    bool stabilFilt = parser.has("f");
    bool saveResults = parser.has("s");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

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
   
/* 
    Ptr<aruco::GridBoard> gridboard = aruco::GridBoard::create(2, 2, float(400), float(10), dictionary);
    Mat boardImage;
    Size imageSize;
    imageSize.width = 1000;
    imageSize.height = 1000;
    gridboard->draw(imageSize, boardImage, 4, 1);
    imwrite("board.png", boardImage);
    Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();
*/

    Mat camMatrix, distCoeffs;
    
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }
    

    VideoCapture inputVideo;
    
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;// 1000 * 1.0 /inputVideo.get(CAP_PROP_FPS);
        cout << "success" << endl;
    } else {
        inputVideo.open(0);
        waitTime = 1;
        cout << "failed" << endl;
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
    vector<Point2d> rectangle2D, rectangle2Db, rectangle2Dc, rectangle2D2, rectangle2D3;

    rectangle3D.push_back(Point3d(-0.2, 0.1, 0));
    rectangle3D.push_back(Point3d(0.2, 0.1, 0));
    rectangle3D.push_back(Point3d(0.2, 0.3, 0));
    rectangle3D.push_back(Point3d(-0.2, 0.3, 0));

    Vec3d rvec_store, tvec_store, tvec2_store, tvec3_store;
    vector< Vec3d > rvecs_store(12);
    vector< Vec3d > tvecs_store(12);
    vector< Vec4d > quats_store(12);

    unsigned int frame_id = 0;
    unsigned int lost_since;
    std::vector< unsigned int > lost_id(12,0);

    VideoWriter cap;

    // save to video
    if (saveResults) cap.open("demo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
		    inputVideo.get(CAP_PROP_FPS), Size(frame_width, frame_height));

    // save results to file
    ofstream resultfile;
    if (stabilFilt && saveResults) {
	    resultfile.open("results_filt.txt");
	    if (resultfile.is_open()) {
		    cout << "Filtered resulting transformations" << endl;
	    }
	    else cout << "Unable to open result file" << endl;
    }
    else if (!stabilFilt && saveResults) {
	    resultfile.open("results_unfilt.txt");
	    if (resultfile.is_open()) {
		   cout << "Unfiltered resulting transformations" << endl;
	    }
	    else cout << "Unable to open result file" << endl;
    }

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);
	//cv::resize(image, image, Size(image.cols/2,image.rows/2));
        
	cout << "Frame " << frame_id << endl;

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;
	vector< Vec3d > rvecs_ord(12);
	vector< Vec3d > tvecs_ord(12);
	Vec3d tvecs2, tvecs2b, tvecs2c, tvecs3, tvecs4; 

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
	double alpha = 0.7;
	double alpha2 = 0.7;

        // draw results
        image.copyTo(imageCopy);

	cout << camMatrix.size() << endl;
	cout << distCoeffs.size() << endl;
	for(unsigned int i=0; i<ids.size(); i++) {
		cout << ids[i] << endl;
		cout << rvecs[i] << endl;
	}
	for(unsigned int i=0; i<rvecs.size(); i++) {
		/*if (ids[i]==1) {
		       	rvecs_ord[0] = rvecs[i];
			tvecs_ord[0] = tvecs[i];
		}
		if (ids[i]==2) { 
			rvecs_ord[1] = rvecs[i];
			tvecs_ord[1] = tvecs[i];
		}
		if (ids[i]==3) { 
			rvecs_ord[2] = rvecs[i];
			tvecs_ord[2] = tvecs[i];
		}
		if (ids[i]==4) { 
			rvecs_ord[3] = rvecs[i];
			tvecs_ord[3] = tvecs[i];
		}*/
		rvecs_ord[ids[i]-1] = rvecs[i];
		tvecs_ord[ids[i]-1] = tvecs[i];
	}
	cout << rvecs_ord[0] << "\n" << rvecs_ord[1] << "\n" << rvecs_ord[2] << "\n" << rvecs_ord[3] << "\n" 
		<< rvecs_ord[4] << "\n" << rvecs_ord[5] << "\n" << rvecs_ord[6] << "\n" << rvecs_ord[7] << "\n" 
		<< rvecs_ord[8] << "\n" << rvecs_ord[9] << "\n" << rvecs_ord[10] << "\n" << rvecs_ord[11] << "\n" << endl;
 
        if(ids.size() > 0) {

            aruco::drawDetectedMarkers(imageCopy, corners, ids);
	    lost_since = 0;

            if(estimatePose) {
                
                for(unsigned int i = 0; i < 12; i++)//ids.size(); i++)
                {
		    if (rvecs_ord[i][0] == 0.0) {
			    lost_id[i] +=1;
			    continue;
		    }
		    // If the video just started, no previous data...	
		    else if (frame_id == 0 || lost_id[i] !=0) {
			    rvecs_store[i] = rvecs_ord[i];
			    tvecs_store[i] = tvecs_ord[i];
			    quats_store[i] = vec2quat(rvecs_ord[i]);
			    lost_id[i] = 0;
		    }

		    // Display some infos
		    Vec4d q1 = vec2quat(rvecs_ord[i]);
		    Vec4d q2 = vec2quat(rvecs_store[i]);
		   		    
		    Vec3d diff = tvecs_store[i] - tvecs_ord[i];
		    Vec4d diff_rot = q2 - q1;
		    float diff_mag = sqrt( diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] );
		    float diff_rot_mag = sqrt( diff_rot[0]*diff_rot[0] + diff_rot[1]*diff_rot[1] + 
				    diff_rot[2]*diff_rot[2] + diff_rot[3]*diff_rot[3] );

		    if (!stabilFilt && resultfile.is_open()) {
			    resultfile << "Frame " << frame_id     << ", " <<
				          "Rot "   << diff_rot_mag << ", " << 
				          "Dist "  << diff_mag     << "; " << "\n";
		    }
		  
		    cout << "x(t-1) = " << tvecs_store[i] << endl;
		    cout << "x(t) = " << tvecs_ord[i] << endl;
		    cout << "r(t-1) = " << rvecs_store[i] << endl;
		    cout << "r(t) = " << rvecs_ord[i] << endl;
		    cout << "q(t-1) = " << q2 << endl;
		    cout << "q(t) = " << q1 << endl;
		    cout << "||x(t-1) - x(t)|| = " << diff_mag << endl;
		    cout << "||q(t-1) - q(t)|| = " << diff_rot_mag << endl;

		    //if (stabilFilt) {
			    
			    if (diff_mag > 1 || diff_rot_mag > 1) {
				    rvecs_ord[i][0] = rvecs_ord[i][1] = rvecs_ord[i][2] = 0.0;
				    cout << "Limit for" << i << endl;
				    continue;
				    //rvecs_ord[i] = rvecs_store[i];
				    //tvecs_ord[i] = tvecs_store[i];
				    //lost_id[i] += 1;
				    //if (frame_id == 0) lost_id[i] = 0;
			    }/*
			    else {
				    lost_id[i] = 0;
			    }
			    
			    // Use quaternions to make a weighted average of rotations
			    Vec4d quat1, quat2, quat_avg;
			    quat1 = vec2quat(rvecs_ord[i]);
			    quat2 = vec2quat(rvecs_store[i]);
			    quat_avg = avgQuat(quat1, quat2, alpha2, (1-alpha2));
			    quats_store[i] = quat_avg;
			    rvecs_ord[i] = quat2vec(quat_avg);
			    
			    //for (int j=0; j<3; j++) {
			//	    tvecs_ord[i][j] = (1.0f - alpha)*tvecs_ord[i][j] + alpha*tvecs_store[i][j];
			  //  } 
			    
			    cout << "(1-alpha)*x(t-1) + alpha*x(t) = " << tvecs_ord[i] << endl;
			    cout << "(1-alpha)*r(t-1) + alpha*r(t) = " << rvecs_ord[i] << endl;
			    cout << "quats_store" << quats_store[i] << endl;
			    */
		   // }

		    
		    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs_ord[i], tvecs_ord[i],
                                    markerLength * 0.5f);

                    cout << "R: " << rvecs_ord[i] << "; T: " << tvecs_ord[i] << "\n" << endl; // the translation is with respect to the principal point
/*		    if (stabilFilt) {
			    q1 = vec2quat(rvecs_ord[i]);
			    q2 = vec2quat(rvecs_store[i]);
			    diff = tvecs_store[i] - tvecs_ord[i];
			    diff_rot = q2 - q1;
			    diff_mag = sqrt( diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] );
			    diff_rot_mag = sqrt( diff_rot[0]*diff_rot[0] + diff_rot[1]*diff_rot[1] + 
					    diff_rot[2]*diff_rot[2] + diff_rot[3]*diff_rot[3] );
			    
			    if (resultfile.is_open()) {
				    resultfile << "Frame " << frame_id     << ", " <<
					          "Rot "   << diff_rot_mag << ", " << 
					          "Dist "  << diff_mag     << "; " << "\n";
			    }
		    }
*/
		    // Save the computed estimation for the next frame
		    rvecs_store[i] = rvecs_ord[i];
		    tvecs_store[i] = tvecs_ord[i];
		    quats_store[i] = vec2quat(rvecs_ord[i]);
                }
	    }
	    // Average markers' orientation using quaternions
	    std::vector< Eigen::Vector4f > quat_eig, quat_eigb, quat_eigc;
	    Eigen::Vector4f quat_eig_avg, quat_eig_avgb, quat_eig_avgc;
	    Vec4d quat_avg, quat_avgb, quat_avgc;

	    bool quats, quatsb, quatsc;
	    quats = quatsb = quatsc = false;

	    for (unsigned int i=0; i<4; i++) {
		    if (lost_id[i] == 0 && rvecs_ord[i][0] != 0.0) {
			    Eigen::Vector4f quat_row;
			    for (int j=0; j<4; j++) {
				    quat_row[j] = quats_store[i][j];
			    }
			    quat_eig.push_back(quat_row);
			    quats = true;
		    }
	    }
	    for (unsigned int i=4; i<8; i++) {
		    if (lost_id[i] == 0 && rvecs_ord[i][0] != 0.0) {
			    Eigen::Vector4f quat_row;
			    for (int j=0; j<4; j++) {
				    quat_row[j] = quats_store[i][j];
			    }
			    quat_eigb.push_back(quat_row);
			    quatsb = true;
		    }
	    }
	    for (unsigned int i=8; i<12; i++) {
		    if (lost_id[i] == 0 && rvecs_ord[i][0] != 0.0) {
			    Eigen::Vector4f quat_row;
			    for (int j=0; j<4; j++) {
				    quat_row[j] = quats_store[i][j];
			    }
			    quat_eigc.push_back(quat_row);
			    quatsc = true;
		    }
	    }
	    //avoid averaging empty sets of quaternions
	    if (quats) quat_eig_avg = quaternionAverage(quat_eig);
	    if (quatsb) quat_eig_avgb = quaternionAverage(quat_eigb);
	    if (quatsc) quat_eig_avgc = quaternionAverage(quat_eigc);
	    //Eigen -> OpenCV
	    for (int i=0; i<4; i++) {
		    quat_avg[i] = quat_eig_avg[i];
		    quat_avgb[i] = quat_eig_avgb[i];
		    quat_avgc[i] = quat_eig_avgc[i];
	    }


	    //Average translation of markers
	    vector< Vec3d > tvecs_store_centered(12);
	    
	    for (int i=0; i<12; i++) {
		    tvecs_store_centered[i] = 0.0;//tvecs_store[i];
	    }

	    // Markers in a square
	    tvecs_store_centered[1][0] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[1][1] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[0][0] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[0][1] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[3][0] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[3][1] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[2][0] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[2][1] = -1.0 * (markerLength / 2 + markerOffset / 2);
	  

            tvecs_store_centered[5][0] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[5][1] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[4][0] = markerLength / 2 + markerOffset / 2 ;
	    tvecs_store_centered[4][1] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[7][0] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[7][1] = markerLength / 2 + markerOffset / 2 + 0.5;
	    tvecs_store_centered[6][0] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[6][1] = -1.0 * (markerLength / 2 + markerOffset / 2);

	    // Markers in line
	    tvecs_store_centered[0+8][0] = 1.5*markerLength + 1.5*markerOffset;
	    tvecs_store_centered[1+8][0] = markerLength / 2 + markerOffset / 2;
	    tvecs_store_centered[2+8][0] = -1.0 * (markerLength / 2 + markerOffset / 2);
	    tvecs_store_centered[3+8][0] = -1.0 * (1.5*markerLength + 1.5*markerOffset);

	    
	    for (int i=0; i<4; i++) {
		    tvecs_store_centered[i] = transformVec(tvecs_store_centered[i], quat2vec(quats_store[i]), tvecs_store[i]);
	    }
	    for (int i=4; i<8; i++) {
		    tvecs_store_centered[i] = transformVec(tvecs_store_centered[i], quat2vec(quats_store[i]), tvecs_store[i]);
	    }
	    for (int i=8; i<12; i++) {
		    tvecs_store_centered[i] = transformVec(tvecs_store_centered[i], quat2vec(quats_store[i]), tvecs_store[i]);
	    }
	    

	    for (int i=0; i<3; i++) {
		    tvecs2[i] = tvecs2b[i] = tvecs2c[i] = 0.0;
		    for (int j=0; j<4; j++) {
			    if (lost_id[j] == 0 && rvecs_ord[j][0] != 0.0) {
				    tvecs2[i] += tvecs_store_centered[j][i];
			    }
		    }
		    for (int j=4; j<8; j++) {
			    if (lost_id[j] == 0 && rvecs_ord[j][0] != 0.0) {
				    tvecs2b[i] += tvecs_store_centered[j][i];
			    }
		    }
		    for (int j=8; j<12; j++) {
			    if (lost_id[j] == 0 && rvecs_ord[j][0] != 0.0) {
				    tvecs2c[i] += tvecs_store_centered[j][i];
			    }
		    }
		    tvecs2[i] /= quat_eig.size();
		    tvecs2b[i] /= quat_eigb.size();
		    tvecs2c[i] /= quat_eigc.size();
		    tvecs3[i] = 0.0;
		    tvecs4[i] = 0.0;
	    }
	    tvecs3[0] += 2;
	    tvecs4[0] += -2;

	    tvecs3 = transformVec(tvecs3, quat2vec(quat_avg), tvecs2);
	    tvecs4 = transformVec(tvecs4, quat2vec(quat_avg), tvecs2);

	    projectPoints(bunny_cloud, quat2vec(quat_avg), tvecs2, camMatrix, distCoeffs, rectangle2D);
	    projectPoints(bunny_cloud, quat2vec(quat_avg), tvecs3, camMatrix, distCoeffs, rectangle2D2);
	    projectPoints(bunny_cloud, quat2vec(quat_avg), tvecs4, camMatrix, distCoeffs, rectangle2D3);

	    projectPoints(bunny_cloud, quat2vec(quat_avgb), tvecs2b, camMatrix, distCoeffs, rectangle2Db);
	    projectPoints(bunny_cloud, quat2vec(quat_avgc), tvecs2c, camMatrix, distCoeffs, rectangle2Dc);

	    for (unsigned int j = 0; j < rectangle2D.size(); j++)
	    {
		    circle(imageCopy, rectangle2D[j], 1, Scalar(255,0,0), -1);
	            circle(imageCopy, rectangle2D2[j], 1, Scalar(0,255,0), -1);
		    circle(imageCopy, rectangle2D3[j], 1, Scalar(0,0,255), -1);
		    circle(imageCopy, rectangle2Db[j], 1, Scalar(255,0,0), -1);
		    circle(imageCopy, rectangle2Dc[j], 1, Scalar(255,0,0), -1);
	    }
	}

	frame_id += 1;

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));
	
	if (saveResults) cap.write(imageCopy);

        //imshow("out", imageCopy);
	Mat imageResize;
	cv::resize(imageCopy, imageResize, Size(imageCopy.cols/3,imageCopy.rows/3));
	imshow("resize", imageResize);


        char key = (char)waitKey(waitTime); 
        if(key == 27) break;

        
    }
    inputVideo.release();
    if (saveResults) cap.release();
    
    resultfile.close();

    return 0;
}
