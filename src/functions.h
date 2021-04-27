#include </home/nicola/Packages/eigen/Eigen/SVD> 
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/plot.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <ctime>
#include <cmath> 
#include <math.h> 
#include <limits>

using namespace std;
using namespace cv;

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) { 
	FileStorage fs(filename, FileStorage::READ);
	if(!fs.isOpened()) return false; 
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
       	return true; 
}


static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
       	FileStorage fs(filename, FileStorage::READ); 
	if(!fs.isOpened()) return false;
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
	fs["minOtsuStdDev"] >>params->minOtsuStdDev;
       	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true; 
}


static Mat cvcloud_load() { 
	Mat cloud(1, 7708, CV_64FC3); 
	ifstream ifs("arrow.ply");
	string str; for(size_t i = 0; i < 13; ++i) getline(ifs, str);
	
	Point3d* data = cloud.ptr<cv::Point3d>(); 
	float dummy1, dummy2, dummy3;
	for(size_t i = 0; i < 7708; ++i) ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2 >> dummy3;
    
	return cloud; 
}


Mat create_bbox(double x_scale, double y_scale, double z_scale) { 
	Mat cloud(1, 4, CV_64FC3); 
	Point3d* bbox = cloud.ptr<cv::Point3d>();
	
	bbox[0].x = - 1.0 * (x_scale / 2.0);
       	bbox[0].y = - 1.0 * (y_scale / 2.0);
	bbox[0].z = z_scale / 2.0; 
	bbox[1].x = x_scale / 2.0;
       	bbox[1].y = - 1.0 *(y_scale / 2.0); 
	bbox[1].z = z_scale / 2.0;
       	bbox[2].x = x_scale / 2.0; 
	bbox[2].y = y_scale / 2.0; 
	bbox[2].z = z_scale / 2.0;
	bbox[3].x = - 1.0 * (x_scale / 2.0); 
	bbox[3].y = y_scale / 2.0; 
	bbox[3].z = z_scale / 2.0;
       
	return cloud;
}


// Get angle from Rodrigues vector
double getAngle(Vec3d rvec) { 
	double theta = sqrt( rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2] );
    return theta; 
}

Vec3d rodrigues2euler(Vec3d rvec, bool degrees=false) {
       	Vec3d rvec_euler;
	double angle;
       	double x, y, z;

	angle = getAngle(rvec);
       	x = rvec[0] / angle;
       	y = rvec[1] / angle;
       	z = rvec[2] / angle; 
	
	double s=sin(angle); 
	double c=cos(angle); 
	double t=1-c; 
	double heading, attitude, bank;
	
	if ((x*y*t + z*s) > 0.998) { // north pole singularity detected 
		heading = 2*atan2(x*sin(angle/2), cos(angle/2)); 
		attitude = M_PI/2;
	       	bank = 0;
		rvec_euler[0] = heading;
	       	rvec_euler[1] = attitude; 
		rvec_euler[2] = bank;
		
		if(degrees) rvec_euler*=(180.0/M_PI);
		return rvec_euler; 
	}
       	if ((x*y*t + z*s) < -0.998) { // south pole singularity detected 
		heading = -2*atan2(x*sin(angle/2), cos(angle/2));
	       	attitude = -M_PI/2; 
		bank = 0;
		rvec_euler[0] = heading; 
		rvec_euler[1] = attitude;
		rvec_euler[2] = bank;
		
		if(degrees) rvec_euler*=(180.0/M_PI);
		return rvec_euler; 
	} 
	heading = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t); 
	attitude = asin(x * y * t + z * s) ;
       	bank = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t);
    	rvec_euler[0] = heading;
       	rvec_euler[1] = attitude; 
	rvec_euler[2] = bank;
	
	if(degrees) rvec_euler*=(180.0/M_PI);
    	return rvec_euler;
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

Eigen::Vector4f vec2quat_eigen(Vec3d vec) {
       	Eigen::Vector4f q;
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

Vec3d quat_eigen2vec(Eigen::Vector4f quat) { 
	Vec3d v;
       	double ang = 2*acos(quat[3]);
       	v[0] = quat[0] / sqrt(1 - quat[3]*quat[3]) * ang;
       	v[1] = quat[1] / sqrt(1 - quat[3]*quat[3]) * ang;
       	v[2] = quat[2] / sqrt(1 - quat[3]*quat[3]) * ang;
	return v;
}

/// Compute average of two quaternions using eq. (18) of
//https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf, that
//provides a closed-form solution for averaging two quaternions
Vec4d avgQuat(Vec4d q1, Vec4d q2, double w1 = 1, double w2 = 1) {
       	Vec4d q3;
	double zed = sqrt( (w1-w2)*(w1-w2) + 4*w1*w2*(q1.dot(q2))*(q1.dot(q2)));
       	q3 = ((w1-w2+zed)*q1 + 2*w2*(q1.dot(q2))*q2);
       	double norm = sqrt( q3[0]*q3[0] + q3[1]*q3[1] + q3[2]*q3[2] + q3[3]*q3[3] );
       	q3 = q3 / norm; 
	return q3; 
}


// Method to find the average of a set of rotation quaternions using Singular
// Value Decomposition
/*
 * The algorithm used is described here:
 * https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
 */
Eigen::Vector4f quaternionAverage(std::vector<Eigen::Vector4f> quaternions) {
	if (quaternions.size() == 0) { 
		std::cerr << "Error trying to calculate the average quaternion of an empty set!\n"; 
		return	Eigen::Vector4f::Zero();
       	}
    	// first build a 4x4 matrix which is the elementwise sum of the product of
	// each quaternion with itself
	Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    	for (unsigned long int q=0; q<quaternions.size(); ++q) A += quaternions[q] * quaternions[q].transpose();
    
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
	
	for (int i=0; i<singularValues.rows(); ++i) { 
	       	if (first) {
			largestEigenValue = singularValues(i);
			largestEigenValueIndex = i;
	   		first = false;
		}
		else if (singularValues(i) > largestEigenValue) {
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


// Transform a relative translation into absolute (useful for augmented reality
// when we have offset wrt marker frame)
Vec3d transformVec(Vec3d vec, Vec3d rotvec, Vec3d tvec) {
       	Mat rotationMat = Mat::zeros(3, 3, CV_64F);
       	Mat transformMat = Mat::eye(4, 4, CV_64F);
	Rodrigues(rotvec, rotationMat); //convert rodrigues angles into	rotation matrix
	
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
    
	Vec3d vectrans; //output, vector transformed
       	Mat vectransMat = transformMat*Mat(vechomo);
       	vectrans[0] = vectransMat.at<double>(0);
       	vectrans[1] = vectransMat.at<double>(1);
       	vectrans[2] = vectransMat.at<double>(2);
    
	return vectrans;
}


// Transform a relative point into absolute
Point3d transformPoint(Point3d vec, Vec3d rotvec, Vec3d tvec) { 
	Mat rotationMat	= Mat::zeros(3, 3, CV_64F);
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
	vechomo[0] = vec.x;
       	vechomo[1] = vec.y;
       	vechomo[2] = vec.z;
       	vechomo[3] = 1.0;
    
	Point3d vectrans; //output, vector transformed 
	Mat vectransMat = transformMat*Mat(vechomo);
       	vectrans.x = vectransMat.at<double>(0);
       	vectrans.y = vectransMat.at<double>(1);
       	vectrans.z = vectransMat.at<double>(2);
    	
	return vectrans;
}


// Avg two poses with weights associated

Vec3d avgRot(Vec3d rvec1, Vec3d rvec2, double weight1, double weight2) { 
	Vec4d quat1 = vec2quat(rvec1);
       	Vec4d quat2 = vec2quat(rvec2);
       	Vec4d quat_avg = avgQuat(quat1, quat2, weight1, weight2);
    	
	return quat2vec(quat_avg);
}

Vec3d avgTrasl(Vec3d tvec1, Vec3d tvec2, double weight1, double weight2) {
	Vec3d tvec_avg;
       	for (int i=0; i<3; i++) { 
		tvec_avg[i] = weight1*tvec1[i] + weight2*tvec2[i];
       	}
     	return tvec_avg;
}


// Check diff between two rotations in Euler notation
bool checkDiffRot(Vec3d rvec1, Vec3d rvec2, std::vector<double> thr) {
       	Vec3d rvec1_eul = rodrigues2euler(rvec1);
       	Vec3d rvec2_eul = rodrigues2euler(rvec2);
    	for(int i=0; i<3; i++) { 
		if(std::abs(sin(rvec1_eul[i])-sin(rvec2_eul[i])) > thr[i]) { 
			return false;
	       	}
       	}
       	return true; 
}


// Compute combination pose at center of marker group
Vec3d computeAvgRot(std::vector<Vec3d> rvecs_ord, std::vector<bool> detect_id, int group) { 
	std::vector<Eigen::Vector4f> quat_avg;
       	Vec3d rvec_avg;
       	for(unsigned int i=0; i<4; i++) {
	       	Eigen::Vector4f quat;
		if(detect_id[group*4+i]) { 
			quat = vec2quat_eigen(rvecs_ord[group*4+i]);
			quat_avg.push_back(quat);
	       	}
       	}
       	rvec_avg = quat_eigen2vec(quaternionAverage(quat_avg));
    	
	return rvec_avg;
}

Vec3d computeAvgTrasl(std::vector<Vec3d> tvecs_ord, std::vector<Vec3d> rvecs_ord, std::vector<bool> detect_id, 
		int group, float markerLength, float markerOffset) {
       	std::vector<Vec3d> tvecs_centered;
       	Vec3d tvec_avg;
    
	if(group==0 || group==1 || group==3) { // markers in a square 
		for(unsigned int i=0; i<4; i++) {
		       	Vec3d tvec;
		       	if(detect_id[group*4+i]) {
				if(i==0) {
				       	tvec[0] = markerLength / 2 + markerOffset / 2;
					tvec[1] = -1.0 * (markerLength / 2 + markerOffset / 2);
				       	tvec[2] = 0.0;
				       	tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
	    				tvecs_centered.push_back(tvec);
			       	} 
				else if(i==1) {
    					tvec[0] = markerLength / 2 + markerOffset / 2;
				       	tvec[1] = markerLength / 2 + markerOffset / 2; 
					tvec[2] = 0.0;
    					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
    					tvecs_centered.push_back(tvec);
			       	}
			       	else if(i==2) { 
					tvec[0] = -1.0 * (markerLength / 2 + markerOffset / 2); 
					tvec[1] = -1.0 * (markerLength / 2 + markerOffset / 2);
					tvec[2] = 0.0;
				       	tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				} 
				else if(i==3) { 
					tvec[0] = -1.0 * (markerLength / 2 + markerOffset / 2);
				       	tvec[1] = markerLength / 2 + markerOffset / 2;
					tvec[2] = 0.0; 
					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				}
		       	}
	       	}
       	}
       	else { // markers in line
		for(unsigned int i=0; i<4; i++) {
	    		Vec3d tvec;
			if(detect_id[group*4+i]) { 
				if(i==0) {
					tvec[0] = 1.5*markerLength + 1.5*markerOffset;
					tvec[1] = tvec[2] = 0.0;
					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				} 
				else if(i==1) {
					tvec[0] = markerLength/2 + markerOffset/2;
					tvec[1] = tvec[2] = 0.0;
					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				}
				else if(i==2) {
					tvec[0] = -1.0*(markerLength/2 + markerOffset/2);
					tvec[1] = tvec[2] = 0.0;
					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				}
				else if(i==3) {	
					tvec[0] = -1.0*(1.5*markerLength + 1.5*markerOffset);
				     	tvec[1] = tvec[2] = 0.0;
					tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
					tvecs_centered.push_back(tvec);
				}
		      	}
	       	} 
	}
    	for (int i=0; i<3; i++) { 
		tvec_avg[i] = 0.0; 
		for (unsigned int j=0; j<tvecs_centered.size(); j++) { 
			tvec_avg[i] += tvecs_centered[j][i];
	       	}
	       	tvec_avg[i] /= tvecs_centered.size(); 
	}
    	
	return tvec_avg; 
} 



// Check if num markers' poses are consistent
std::vector<bool> checkPoseConsistent(std::vector<Vec3d> rvecs_ord,
		std::vector<bool> detect_id, unsigned int num, int group,
		std::vector<double> thr) {
       	cout << "Checking markers' consistency for GROUP " << group << endl;
       	std::vector<bool> checkVec = detect_id;
       	std::vector<Vec3d> rvecs; 
	unsigned int items=0;

    	for(int i=0; i<4; i++) {
		rvecs.push_back(rodrigues2euler(rvecs_ord[group*4+i])); 
		Mat rotationMat = Mat::zeros(3, 3, CV_64F);  
		Rodrigues(rvecs_ord[group*4+i], rotationMat); //convert rodrigues angles into rotation matrix
	       	cout << "[" ; 
		for(int i = 0; i<3; i++) { 
			for(int j=0; j<3; j++) { 
				cout << rotationMat.at<double>(i,j) << " ";
		       	}
		       	cout << endl;
	       	}
	       	cout << "]" << endl;
		
		if(detect_id[group*4+i]) {
		       	items += 1;
	       	}
       	}
    
	if(items < num) { 
		for(int i=0; i<4; i++) {
		       	checkVec[group*4+i] = false;
	       	}

		return checkVec; 
	}
    
	cout << "Detected markers to compare: " << items << endl;
    	std::vector<std::vector<bool>> checker(rvecs.size(), std::vector<bool>(rvecs.size(), true));
	
	for(unsigned int i=0; i<rvecs.size(); i++) {
	       	if(!detect_id[group*4+i]) {
			checker[0][i] = checker[1][i] = checker[2][i] = checker[3][i] = false; 
			continue;
	       	}
	       	for(unsigned int j=0; j<rvecs.size(); j++) { 
			if(i==j) continue;
			if(!detect_id[group*4+j]) { 
				checker[i][j] = false;
				continue; 
			}
	    		for(int k=0; k<3; k++) { 
				cout << "Diff between angles: " << std::abs(sin(rvecs[i][k])-sin(rvecs[j][k]))
				       	<< " > " << thr[k] << "?" << endl;
				if(std::abs(sin(rvecs[i][k])-sin(rvecs[j][k])) > thr[k]) {
				       	cout << "YES!!" << endl;
				       	checker[i][j] = false;
	    				break;
			       	}
			       	else {
					cout << "No, OK. " << endl;
				       	checker[i][j] = true;
			       	}
		       	}
	       	}
       	}
    
	for(unsigned int i=0; i<rvecs.size(); i++) {
	       	unsigned int trues=0;
	       	unsigned int falses=0;
	
		// count how many markers are consistent with current one
		for(unsigned int j=0; j<rvecs.size(); j++) { 
			if(i==j) continue;
			if(!checker[i][j]) { 
				falses += 1;
		       	}
		       	else { 
				trues += 1;
		       	} 
		}
		// If it agrees with all markers, keep it
		if(trues >= (num-1)) { 
			checkVec[group*4+i] = true;
		       	continue;
	       	}
	       	else {
			checkVec[group*4+i] = false;
		       	continue;
	       	} 
	}
    	cout << "Checker: ";
       	for(unsigned int i=0; i<rvecs.size();i++) {      
		for(auto&& j:checker[i]) {
		       	cout << j << " "; } 
	}
       
	return checkVec;
}


// Given a vector of 2d points, draw a semi-transparent box
void DrawBox2D(Mat imageCopy, vector<Point2d> box1, int b_ch, int r_ch, int g_ch) {
	line(imageCopy, box1[0], box1[1], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    	line(imageCopy, box1[1], box1[2], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
	line(imageCopy, box1[2], box1[3], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
	line(imageCopy, box1[3], box1[0], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
     
	Point face1[1][4];
    	face1[0][0] = Point(box1[0].x, box1[0].y); face1[0][1] = Point(box1[1].x, box1[1].y);
       	face1[0][2] = Point(box1[2].x, box1[2].y);
	face1[0][3] = Point(box1[3].x, box1[3].y);
      
	const Point* boxppt1[1] = {face1[0]};
      	int npt[] = {4}; double alpha = 0.3;
      
	Mat overlay1;
       
	imageCopy.copyTo(overlay1);
       	fillPoly(overlay1, boxppt1, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
       	addWeighted(overlay1, alpha, imageCopy, 1-alpha, 0, imageCopy);
}

// Function to average boxes
vector<Point2d> avgBoxes(vector<vector<Point2d>> boxes, vector<double> weights) {
       	vector<Point2d> avg_box(8); 
	avg_box[0].x = 0.0;
       	avg_box[0].y = 0.0;
	avg_box[1].x = 0.0;
       	avg_box[1].y = 0.0; 
	avg_box[2].x = 0.0;
	avg_box[2].y = 0.0; 
	avg_box[3].x = 0.0;
       	avg_box[3].y = 0.0;
    
	for(unsigned int i=0; i<boxes.size(); i++) { 
		avg_box[0].x += boxes[i][0].x * weights[i]; 
		avg_box[0].y += boxes[i][0].y * weights[i];
    		avg_box[1].x += boxes[i][1].x * weights[i]; 
		avg_box[1].y += boxes[i][1].y * weights[i];
	       	avg_box[2].x += boxes[i][2].x * weights[i];
	       	avg_box[2].y += boxes[i][2].y * weights[i];
    		avg_box[3].x += boxes[i][3].x * weights[i]; 
		avg_box[3].y +=	boxes[i][3].y * weights[i];
       	} 
	for(int i=0; i<8; i++) {
	       	cout << avg_box[i].x << ", " << avg_box[i].y << endl;
       	}
    
	return avg_box;
}

// Function that returns the coordinates of boxes in the scene
std::vector<Vec3d> computeAvgBoxes(std::vector<Vec3d> rMaster, std::vector<Vec3d> tMaster, std::vector<bool> init_id, int scene) {
       	std::vector<Vec3d> avg_points;
       	Vec3d a0, b0, c0, d0, a1, b1, c1, d1, a3, b3, c3, d3, a_avg, b_avg, c_avg, d_avg;
	Vec3d e0, f0, g0, h0, e1, f1, g1, h1, e3, f3, g3, h3, e_avg, f_avg, g_avg, h_avg;
	std::vector<Vec3d> a_sum, b_sum, c_sum, d_sum, e_sum, f_sum, g_sum, h_sum;

    	if(scene == 3) { 
	a0[0] = -1.5 - 0.1; 
	a0[1] = -(-0.5 + 10.7); 
	a0[2] = -3;
	b0[0] = -1.5 - 0.1;
       	b0[1] = -(-0.5 + 10.7);
       	b0[2] = -43; 
	c0[0] = -1.5 - 0.1;
       	c0[1] = -(-20 - 0.5 + 10.7); 
	c0[2] = -23;
       	d0[0] = -1.5 - 0.1;
       	d0[1] = -(20 - 0.5 + 10.7); 
	d0[2] = -23;
       	e0[0] = -1.5 - 0.1;
	e0[1] = -(-10.5 - 0.5 + 10.7);
	e0[2] = -3 - 3;
	f0[0] = 1.5 - 0.1;
	f0[1] = -(-10.5 - 0.5 + 10.7); 
	f0[2] = -3 - 3 - 2*17.0;
	g0[0] = 1.5 - 0.1;
	g0[1] = -(10.5 - 0.5 + 10.7);
	g0[2] = -3 - 3 - 2*17.0;
	h0[0] = -1.5 - 0.1;
	h0[1] = -(10.5 - 0.5 + 10.7);
	h0[2] = -3 - 3;
	a1[0] = -0.5;
	a1[1] = -1.5;
	a1[2] = -3;
	b1[0] = -0.5;
	b1[1] = -1.5;
	b1[2] = -43;
	c1[0] = -20 - 0.5;
	c1[1] = -1.5;
	c1[2] = -23;
	d1[0] = 20 - 0.5;
	d1[1] = -1.5;
	d1[2] = -23;
	e1[0] = -10.5 - 0.5;
	e1[1] = -1.5;
	e1[2] = -3 - 3;
	f1[0] = -10.5 - 0.5;
	f1[1] = -1.5;
	f1[2] = -3 - 3 - 2*17.0;
	g1[0] = 10.5 - 0.5;
	g1[1] = -1.5;
	g1[2] = -3 - 3 - 2*17.0;
	h1[0] = 10.5 - 0.5;
	h1[1] = -1.5;
	h1[2] = -3 - 3;

            
	a0 = transformVec(a0, rMaster[0], tMaster[0]);
	b0 = transformVec(b0, rMaster[0], tMaster[0]);
	c0 = transformVec(c0, rMaster[0], tMaster[0]);
	d0 = transformVec(d0, rMaster[0], tMaster[0]);
	e0 = transformVec(e0, rMaster[0], tMaster[0]);
	f0 = transformVec(f0, rMaster[0], tMaster[0]);
	g0 = transformVec(g0, rMaster[0], tMaster[0]);
	h0 = transformVec(h0, rMaster[0], tMaster[0]);   
	a1 = transformVec(a1, rMaster[1], tMaster[1]);
	b1 = transformVec(b1, rMaster[1], tMaster[1]);
	c1 = transformVec(c1, rMaster[1], tMaster[1]);
	d1 = transformVec(d1, rMaster[1], tMaster[1]);  
	e1 = transformVec(e1, rMaster[1], tMaster[1]);
	f1 = transformVec(f1, rMaster[1], tMaster[1]);
	g1 = transformVec(g1, rMaster[1], tMaster[1]);
	h1 = transformVec(h1, rMaster[1], tMaster[1]);


	if(init_id[0]) {
            a_sum.push_back(a0);
            b_sum.push_back(b0);
            c_sum.push_back(c0);
            d_sum.push_back(d0);
            e_sum.push_back(e0);
            f_sum.push_back(f0);
            g_sum.push_back(g0);
            h_sum.push_back(h0);
        }
        if(init_id[4]) {
            a_sum.push_back(a1);
            b_sum.push_back(b1);
            c_sum.push_back(c1);
            d_sum.push_back(d1);
            e_sum.push_back(e1);
            f_sum.push_back(f1);
            g_sum.push_back(g1);
            h_sum.push_back(h1);
        }
        if(init_id[0]||init_id[4]){
            for (int i=0; i<3; i++) {
                a_avg[i] = 0.0;
                b_avg[i] = 0.0;
                c_avg[i] = 0.0;
                d_avg[i] = 0.0;
                e_avg[i] = 0.0;
                f_avg[i] = 0.0;
                g_avg[i] = 0.0;
                h_avg[i] = 0.0;

                for (unsigned int j=0; j<a_sum.size(); j++) {
                    a_avg[i] += a_sum[j][i];
                    b_avg[i] += b_sum[j][i];
                    c_avg[i] += c_sum[j][i];
                    d_avg[i] += d_sum[j][i];
                    e_avg[i] += e_sum[j][i];
                    f_avg[i] += f_sum[j][i];
                    g_avg[i] += g_sum[j][i];
                    h_avg[i] += h_sum[j][i];

                }
                a_avg[i] /= a_sum.size();
                b_avg[i] /= b_sum.size();
                c_avg[i] /= c_sum.size();
                d_avg[i] /= d_sum.size();
                e_avg[i] /= e_sum.size();
                f_avg[i] /= f_sum.size();
                g_avg[i] /= g_sum.size();
                h_avg[i] /= h_sum.size();

            }
        }
        else {
            return avg_points;
        }
        avg_points.push_back(a_avg);
        avg_points.push_back(b_avg);
        avg_points.push_back(c_avg);
        avg_points.push_back(d_avg);
        avg_points.push_back(e_avg);
        avg_points.push_back(f_avg);
        avg_points.push_back(g_avg);
        avg_points.push_back(h_avg);

    }
    else if(scene==1) {
        a0[0] = 0.0;
	a0[1] = 6.0;
	a0[2] = -1;
	b0[0] = 5.0;
	b0[1] = 6.0;
	b0[2] = -1;
	c0[0] = -5.5;
	c0[1] = 7.0;
	c0[2] = -6;
        
        a0 = transformVec(a0, rMaster[1], tMaster[1]);
	b0 = transformVec(b0, rMaster[1], tMaster[1]);
	c0 = transformVec(c0, rMaster[1], tMaster[1]);

        if(init_id[4]) {
            avg_points.push_back(a0);
            avg_points.push_back(b0);
            avg_points.push_back(c0);
        }
        else {
            return avg_points;
        }
    }
    else if(scene==5) {
        a0[0] = 0.0;
	a0[1] = 2.0;
	a0[2] = 0.0;        
        a0 = transformVec(a0, rMaster[2], tMaster[2]);
        if(init_id[8]) {
            avg_points.push_back(a0);
        }
        else {
            return avg_points;
        }
    }
    return avg_points;
} 

void drawToImg(Mat img, vector<vector<Point2d>>& boxes, vector<bool>& init_id, int scene) {
	if (init_id[0] || init_id[4] || init_id[8]) {
		if (scene==3) {
			DrawBox2D(img, boxes[0], 60, 20, 220);
			DrawBox2D(img, boxes[1], 60, 20, 220);
			DrawBox2D(img, boxes[2], 60, 20, 220);
			DrawBox2D(img, boxes[3], 60, 20, 220);
			DrawBox2D(img, boxes[4], 60, 20, 220);
			DrawBox2D(img, boxes[5], 60, 20, 220);
			DrawBox2D(img, boxes[6], 60, 20, 220);
			DrawBox2D(img, boxes[7], 60, 20, 220);
		}
		else if (scene==1) {
			DrawBox2D(img, boxes[0], 60, 20, 220);
			DrawBox2D(img, boxes[1], 60, 20, 220);
			DrawBox2D(img, boxes[2], 60, 20, 220);
		}
		else if (scene==5) {
			DrawBox2D(img, boxes[0], 60, 20, 220);
		}
		else {
			cout << "No known scenario...maybe wrong scene?" << endl;
		}
	}
}

void combineBoxes(Mat camMatrix, Mat distCoeffs, Mat box_cloud, vector<vector<Point2d>>& boxes, vector<bool>& init_id, vector<Vec3d>& avg_points, vector<double>& weights, bool average, int scene) { 
	vector<vector<Point2d>> boxes1, boxes2, boxes3, boxes4, boxes5, boxes6, boxes7, boxes8;

	if(average==true) {
            if (scene==3) {
                boxes1.push_back(boxes[0]);
                boxes2.push_back(boxes[1]);
                boxes3.push_back(boxes[2]);
                boxes4.push_back(boxes[3]);
		boxes5.push_back(boxes[4]);
		boxes6.push_back(boxes[5]);
		boxes7.push_back(boxes[6]);
		boxes8.push_back(boxes[7]);
            }
            else if (scene==1) {
                boxes1.push_back(boxes[0]);
                boxes2.push_back(boxes[1]);
                boxes3.push_back(boxes[2]);
            }
            else if (scene==5) {
                boxes1.push_back(boxes[0]);
            }
	    }
	    else {
		cout << "EMPTY!!!" << endl;
	    }
	    if (init_id[0] || init_id[4] || init_id[8]) {
            	average = true;
           	if (scene==3) {
               		projectPoints(box_cloud, Vec3d::zeros(), avg_points[0], camMatrix, distCoeffs, boxes[0]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[1], camMatrix, distCoeffs, boxes[1]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[2], camMatrix, distCoeffs, boxes[2]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[3], camMatrix, distCoeffs, boxes[3]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[4], camMatrix, distCoeffs, boxes[4]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[5], camMatrix, distCoeffs, boxes[5]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[6], camMatrix, distCoeffs, boxes[6]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[7], camMatrix, distCoeffs, boxes[7]);
            	}
            	else if (scene==1) {
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[0], camMatrix, distCoeffs, boxes[0]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[1], camMatrix, distCoeffs, boxes[1]);
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[2], camMatrix, distCoeffs, boxes[2]);
           	 }
            	else if (scene==5) {
                	projectPoints(box_cloud, Vec3d::zeros(), avg_points[0], camMatrix, distCoeffs, boxes[0]);
            	}
	   }
	   if(!boxes1.empty()) {
           if (scene==3) {
		   boxes1.push_back(boxes[0]);
		   boxes2.push_back(boxes[1]);
		   boxes3.push_back(boxes[2]);
		   boxes4.push_back(boxes[3]); 
		   boxes5.push_back(boxes[4]);
		   boxes6.push_back(boxes[5]);
		   boxes7.push_back(boxes[6]);
		   boxes8.push_back(boxes[7]); 
		   boxes[0] = avgBoxes(boxes1, weights);
		   boxes[1] = avgBoxes(boxes2, weights);
		   boxes[2] = avgBoxes(boxes3, weights);
		   boxes[3] = avgBoxes(boxes4, weights); 
		   boxes[4] = avgBoxes(boxes5, weights);
		   boxes[5] = avgBoxes(boxes6, weights);
		   boxes[6] = avgBoxes(boxes7, weights);
		   boxes[7] = avgBoxes(boxes8, weights);
            }
            else if (scene==1) {
		boxes1.push_back(boxes[0]);
		boxes2.push_back(boxes[1]);
		boxes3.push_back(boxes[2]); 
		boxes[0] = avgBoxes(boxes1, weights);
		boxes[1] = avgBoxes(boxes2, weights);
		boxes[2] = avgBoxes(boxes3, weights);
            }
            else if (scene==5) {
                boxes1.push_back(boxes[0]);
                boxes[0] = avgBoxes(boxes1, weights);
            }
	    }	
}

/* Functions for monocular visual odometry */
// Credits: Avi Singh 2015
//

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for(unsigned int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}


void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}
