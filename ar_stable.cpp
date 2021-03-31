#include </home/nico/packages/eigen/Eigen/SVD>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <math.h>
#include <limits>

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
        "{l        | 0.54  | Marker side lenght (in meters). Needed for correct scale in camera pose }"
        "{o        | 0.04  | Offset between markers (in meters) }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
        "{n        | false | Naive mode (no stabilization)}"
        "{s        |       | Save results}";
}


static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


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

    return cloud;
}


Mat create_bbox(double x_scale, double y_scale, double z_scale) 
{
    Mat cloud(1, 8, CV_64FC3);
    Point3d* bbox = cloud.ptr<cv::Point3d>();

    bbox[0].x = - 1.0 * (x_scale / 2.0);
    bbox[0].y = - 1.0 * (y_scale / 2.0);
    bbox[0].z = z_scale / 2.0;
    bbox[1].x = x_scale / 2.0;
    bbox[1].y = - 1.0 * (y_scale / 2.0);
    bbox[1].z = z_scale / 2.0;
    bbox[2].x = x_scale / 2.0;
    bbox[2].y = y_scale / 2.0;
    bbox[2].z = z_scale / 2.0;
    bbox[3].x = - 1.0 * (x_scale / 2.0);
    bbox[3].y = y_scale / 2.0;
    bbox[3].z = z_scale / 2.0;
    bbox[4].x = x_scale / 2.0;
    bbox[4].y = y_scale / 2.0;
    bbox[4].z = - 1.0 * (z_scale / 2.0);
    bbox[5].x = x_scale / 2.0;
    bbox[5].y = - 1.0 * (y_scale / 2.0);
    bbox[5].z = - 1.0 * (z_scale / 2.0);
    bbox[6].x = - 1.0 * (x_scale / 2.0);
    bbox[6].y = y_scale / 2.0;
    bbox[6].z = - 1.0 * (z_scale / 2.0);
    bbox[7].x = - 1.0 * (x_scale / 2.0);
    bbox[7].y = - 1.0 * (y_scale / 2.0);
    bbox[7].z = - 1.0 * (z_scale / 2.0);

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

    Vec3d vectrans; //output, vector transformed
        Mat vectransMat = transformMat*Mat(vechomo);
    vectrans[0] = vectransMat.at<double>(0);
    vectrans[1] = vectransMat.at<double>(1);
    vectrans[2] = vectransMat.at<double>(2);

    return vectrans;
}
// Transform a relative point into absolute
Point3d transformPoint(Point3d vec, Vec3d rotvec, Vec3d tvec) {
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

Vec3d computeAvgTrasl(std::vector<Vec3d> tvecs_ord, std::vector<Vec3d> rvecs_ord, 
                      std::vector<bool> detect_id, int group, float markerLength, float markerOffset) {
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
                    tvec[0] = markerLength / 2 + markerOffset / 2;
                    tvec[1] = tvec[2] = 0.0;
                    tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
                    tvecs_centered.push_back(tvec);
                }
                else if(i==2) {
                    tvec[0] = -1.0 * (markerLength / 2 + markerOffset / 2);
                    tvec[1] = tvec[2] = 0.0;
                    tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
                    tvecs_centered.push_back(tvec);
                }
                else if(i==3) {
                    tvec[0] = -1.0 * (1.5*markerLength + 1.5*markerOffset);
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


// Compute pose for the whole scene
Vec3d computeSceneRot(std::vector<Vec3d> rMaster, std::vector<bool> detect_id, std::vector<bool> init_id, int scene) {
    std::vector<Eigen::Vector4f> quat_avg;
    Vec3d rvec_avg;
    for(unsigned int i=0; i<4; i++) {
        Eigen::Vector4f quat;
        if(init_id[i*4]) {
            quat = vec2quat_eigen(rMaster[i]);
            quat_avg.push_back(quat);
        }
    }
    cout << "Size" << quat_avg.size() << endl;
    if (quat_avg.size()==0) return rMaster[0];
    if (quat_avg.size()==1) return quat_eigen2vec(quat_avg[0]);
    rvec_avg = quat_eigen2vec(quaternionAverage(quat_avg));

    return rvec_avg;
}

Vec3d computeSceneTrasl(std::vector<Vec3d> tvecs_ord, std::vector<Vec3d> rvecs_ord, 
                      std::vector<bool> detect_id, int group, float markerLength, float markerOffset) {
    std::vector<Vec3d> tvecs_centered;
    Vec3d tvec_avg;

    Vec3d tvec1, tvec2;
    tvec1[0] = -2.05;
    tvec1[1] = -1.1;
    tvec1[2] = 0.0;
    tvec2[0] = 1.0;
    tvec2[1] = 1.0;
    tvec2[2] = 0.0;


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
                    tvec[0] = markerLength / 2 + markerOffset / 2;
                    tvec[1] = tvec[2] = 0.0;
                    tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
                    tvecs_centered.push_back(tvec);
                }
                else if(i==2) {
                    tvec[0] = -1.0 * (markerLength / 2 + markerOffset / 2);
                    tvec[1] = tvec[2] = 0.0;
                    tvec = transformVec(tvec, rvecs_ord[group*4+i], tvecs_ord[group*4+i]);
                    tvecs_centered.push_back(tvec);
                }
                else if(i==3) {
                    tvec[0] = -1.0 * (1.5*markerLength + 1.5*markerOffset);
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
std::vector<bool> checkPoseConsistent(std::vector<Vec3d> rvecs_ord, std::vector<bool> detect_id, unsigned int num, 
                         int group, std::vector<double> thr) {
    std::vector<bool> checkVec = detect_id;
    std::vector<Vec3d> rvecs;
    unsigned int items=0;

    
    for(int i=0; i<4; i++) {
        rvecs.push_back(rodrigues2euler(rvecs_ord[group*4+i]));

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

                if(std::abs(sin(rvecs[i][k])-sin(rvecs[j][k])) > thr[k]) {
                    checker[i][j] = false;
                    break;
                }
                else {
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
   
    return checkVec;
}


void DrawBox2D(Mat imageCopy, vector<Point2d> box1, int b_ch, int r_ch, int g_ch) {

    line(imageCopy, box1[0], box1[1], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[1], box1[2], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[2], box1[3], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[3], box1[0], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);

    line(imageCopy, box1[4], box1[6], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[6], box1[7], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[7], box1[5], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[5], box1[4], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);

    line(imageCopy, box1[6], box1[3], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[7], box1[0], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[2], box1[4], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);
    line(imageCopy, box1[1], box1[5], Scalar(b_ch,r_ch,g_ch), 2, LINE_8);

    Point face1[1][4];
    Point face2[1][4];
    Point face3[1][4];
    Point face4[1][4];
    Point face5[1][4];
    Point face6[1][4];

    face1[0][0] = Point(box1[0].x, box1[0].y);
    face1[0][1] = Point(box1[1].x, box1[1].y);
    face1[0][2] = Point(box1[2].x, box1[2].y);
    face1[0][3] = Point(box1[3].x, box1[3].y);

    face2[0][0] = Point(box1[1].x, box1[1].y);
    face2[0][1] = Point(box1[5].x, box1[5].y);
    face2[0][2] = Point(box1[4].x, box1[4].y);
    face2[0][3] = Point(box1[2].x, box1[2].y);

    face3[0][0] = Point(box1[5].x, box1[5].y);
    face3[0][1] = Point(box1[7].x, box1[7].y);
    face3[0][2] = Point(box1[6].x, box1[6].y);
    face3[0][3] = Point(box1[4].x, box1[4].y);

    face4[0][0] = Point(box1[7].x, box1[7].y);
    face4[0][1] = Point(box1[0].x, box1[0].y);
    face4[0][2] = Point(box1[3].x, box1[3].y);
    face4[0][3] = Point(box1[6].x, box1[6].y);

    face5[0][0] = Point(box1[3].x, box1[3].y);
    face5[0][1] = Point(box1[2].x, box1[2].y);
    face5[0][2] = Point(box1[4].x, box1[4].y);
    face5[0][3] = Point(box1[6].x, box1[6].y);

    face6[0][0] = Point(box1[0].x, box1[0].y);
    face6[0][1] = Point(box1[1].x, box1[1].y);
    face6[0][2] = Point(box1[5].x, box1[5].y);
    face6[0][3] = Point(box1[7].x, box1[7].y);


    const Point* boxppt1[1] = {face1[0]};
    const Point* boxppt2[1] = {face2[0]};
    const Point* boxppt3[1] = {face3[0]};
    const Point* boxppt4[1] = {face4[0]};
    const Point* boxppt5[1] = {face5[0]};
    const Point* boxppt6[1] = {face6[0]};

    int npt[] = {4};
    double alpha = 0.3;

    Mat overlay1, overlay2, overlay3, overlay4, overlay5, overlay6;
    imageCopy.copyTo(overlay1);
    fillPoly(overlay1, boxppt1, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay1, alpha, imageCopy, 1-alpha, 0, imageCopy);

    imageCopy.copyTo(overlay2);
    fillPoly(overlay2, boxppt2, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay2, alpha, imageCopy, 1-alpha, 0, imageCopy);

    imageCopy.copyTo(overlay3);
    fillPoly(overlay3, boxppt3, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay3, alpha, imageCopy, 1-alpha, 0, imageCopy);

    imageCopy.copyTo(overlay4);
    fillPoly(overlay4, boxppt4, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay4, alpha, imageCopy, 1-alpha, 0, imageCopy);

    imageCopy.copyTo(overlay5);
    fillPoly(overlay5, boxppt5, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay5, alpha, imageCopy, 1-alpha, 0, imageCopy);

    imageCopy.copyTo(overlay6);
    fillPoly(overlay6, boxppt6, npt, 1, Scalar(b_ch,r_ch,g_ch), LINE_8);
    addWeighted(overlay6, alpha, imageCopy, 1-alpha, 0, imageCopy);
}

vector<Point2d> avgBoxes(vector<vector<Point2d>> boxes, vector<bool> init_id) {
    vector<Point2d> avg_box(8);
    avg_box[0].x = 0.0;
    avg_box[0].y = 0.0;
    avg_box[1].x = 0.0;
    avg_box[1].y = 0.0;
    avg_box[2].x = 0.0;
    avg_box[2].y = 0.0;
    avg_box[3].x = 0.0;
    avg_box[3].y = 0.0;
    avg_box[4].x = 0.0;
    avg_box[4].y = 0.0;
    avg_box[5].x = 0.0;
    avg_box[5].y = 0.0;
    avg_box[6].x = 0.0;
    avg_box[6].y = 0.0;
    avg_box[7].x = 0.0;
    avg_box[7].y = 0.0;

    int initialized = 0;
    for(int i=0;i<4;i++) {
        if(init_id[i*4]) {
            initialized += 1;
            avg_box[0].x += boxes[i][0].x;
            avg_box[0].y += boxes[i][0].y;
            avg_box[1].x += boxes[i][1].x;
            avg_box[1].y += boxes[i][1].y;
            avg_box[2].x += boxes[i][2].x;
            avg_box[2].y += boxes[i][2].y;
            avg_box[3].x += boxes[i][3].x;
            avg_box[3].y += boxes[i][3].y;
            avg_box[4].x += boxes[i][4].x;
            avg_box[4].y += boxes[i][4].y;
            avg_box[5].x += boxes[i][5].x;
            avg_box[5].y += boxes[i][5].y;
            avg_box[6].x += boxes[i][6].x;
            avg_box[6].y += boxes[i][6].y;
            avg_box[7].x += boxes[i][7].x;
            avg_box[7].y += boxes[i][7].y;
        }
    }
    if(initialized==0) return boxes[0];
    if(initialized==1) return avg_box;
    for(int i=0;i<8;i++) {
        avg_box[i].x /= initialized;
        avg_box[i].y /= initialized;
    }
    cout << "initialized" << initialized << endl;
    for(int i=0; i<8; i++) {
        cout << avg_box[i].x << ", " << avg_box[i].y << endl;
    }

    return avg_box;
}

Vec3d rotateAxis(Vec3d rvec) {
    Vec3d rvecTrans;
    Mat rMatr, rMatrTrans1, rMatrTrans2, rMatrTrans3;
    rMatr = rMatrTrans1 = rMatrTrans2 = rMatrTrans3 = Mat::zeros(3,3,CV_64F);

    Rodrigues(rvec, rMatr);
/*
    // + 90 x 
    rMatrTrans.at<double>(0,0) = rMatr.at<double>(0,0);
    rMatrTrans.at<double>(1,0) = - rMatr.at<double>(2,0);
    rMatrTrans.at<double>(2,0) = rMatr.at<double>(1,0);
    rMatrTrans.at<double>(0,1) = rMatr.at<double>(0,1);
    rMatrTrans.at<double>(1,1) = - rMatr.at<double>(2,1);
    rMatrTrans.at<double>(2,1) = rMasterMatr.at<double>(1,1);
    rMatrTrans.at<double>(0,2) = rMasterMatr.at<double>(0,2);
    rMatrTrans.at<double>(1,2) = - rMasterMatr.at<double>(2,2);
    rMatrTrans.at<double>(2,2) = rMasterMatr.at<double>(1,2);

    // - 90 x 
    rMatrTrans.at<double>(0,0) = rMatr.at<double>(0,0);
    rMatrTrans.at<double>(1,0) = rMatr.at<double>(2,0);
    rMatrTrans.at<double>(2,0) = - rMatr.at<double>(1,0);
    rMatrTrans.at<double>(0,1) = rMatr.at<double>(0,1);
    rMatrTrans.at<double>(1,1) = rMatr.at<double>(2,1);
    rMatrTrans.at<double>(2,1) = - rMasterMatr.at<double>(1,1);
    rMatrTrans.at<double>(0,2) = rMasterMatr.at<double>(0,2);
    rMatrTrans.at<double>(1,2) = rMasterMatr.at<double>(2,2);
    rMatrTrans.at<double>(2,2) = - rMasterMatr.at<double>(1,2);

    // + 90 y 
    rMatrTrans.at<double>(0,0) = rMatr.at<double>(2,0);
    rMatrTrans.at<double>(1,0) = rMatr.at<double>(1,0);
    rMatrTrans.at<double>(2,0) = - rMatr.at<double>(0,0);
    rMatrTrans.at<double>(0,1) = rMatr.at<double>(2,1);
    rMatrTrans.at<double>(1,1) = rMatr.at<double>(1,1);
    rMatrTrans.at<double>(2,1) = - rMasterMatr.at<double>(0,1);
    rMatrTrans.at<double>(0,2) = rMasterMatr.at<double>(2,2);
    rMatrTrans.at<double>(1,2) = rMasterMatr.at<double>(1,2);
    rMatrTrans.at<double>(2,2) = - rMasterMatr.at<double>(0,2);

    // - 90 y 
    rMatrTrans.at<double>(0,0) = - rMatr.at<double>(2,0);
    rMatrTrans.at<double>(1,0) = rMatr.at<double>(1,0);
    rMatrTrans.at<double>(2,0) = rMatr.at<double>(0,0);
    rMatrTrans.at<double>(0,1) = - rMatr.at<double>(2,1);
    rMatrTrans.at<double>(1,1) = rMatr.at<double>(1,1);
    rMatrTrans.at<double>(2,1) = rMasterMatr.at<double>(0,1);
    rMatrTrans.at<double>(0,2) = - rMasterMatr.at<double>(2,2);
    rMatrTrans.at<double>(1,2) = rMasterMatr.at<double>(1,2);
    rMatrTrans.at<double>(2,2) = rMasterMatr.at<double>(0,2);

    // + 90 z 
    rMatrTrans.at<double>(0,0) = - rMatr.at<double>(1,0);
    rMatrTrans.at<double>(1,0) = rMatr.at<double>(0,0);
    rMatrTrans.at<double>(2,0) = rMatr.at<double>(2,0);
    rMatrTrans.at<double>(0,1) = - rMatr.at<double>(1,1);
    rMatrTrans.at<double>(1,1) = rMatr.at<double>(0,1);
    rMatrTrans.at<double>(2,1) = rMasterMatr.at<double>(2,1);
    rMatrTrans.at<double>(0,2) = - rMasterMatr.at<double>(1,2);
    rMatrTrans.at<double>(1,2) = rMasterMatr.at<double>(0,2);
    rMatrTrans.at<double>(2,2) = rMasterMatr.at<double>(2,2);

    // - 90 z 
    rMatrTrans.at<double>(0,0) = rMatr.at<double>(1,0);
    rMatrTrans.at<double>(1,0) = - rMatr.at<double>(0,0);
    rMatrTrans.at<double>(2,0) = rMatr.at<double>(2,0);
    rMatrTrans.at<double>(0,1) = rMatr.at<double>(1,1);
    rMatrTrans.at<double>(1,1) = - rMatr.at<double>(0,1);
    rMatrTrans.at<double>(2,1) = rMasterMatr.at<double>(2,1);
    rMatrTrans.at<double>(0,2) = rMasterMatr.at<double>(1,2);
    rMatrTrans.at<double>(1,2) = - rMasterMatr.at<double>(0,2);
    rMatrTrans.at<double>(2,2) = rMasterMatr.at<double>(2,2);
*/
    // - 90 y 
    rMatrTrans1.at<double>(0,0) = - rMatr.at<double>(2,0);
    rMatrTrans1.at<double>(1,0) = rMatr.at<double>(1,0);
    rMatrTrans1.at<double>(2,0) = rMatr.at<double>(0,0);
    rMatrTrans1.at<double>(0,1) = - rMatr.at<double>(2,1);
    rMatrTrans1.at<double>(1,1) = rMatr.at<double>(1,1);
    rMatrTrans1.at<double>(2,1) = rMatr.at<double>(0,1);
    rMatrTrans1.at<double>(0,2) = - rMatr.at<double>(2,2);
    rMatrTrans1.at<double>(1,2) = rMatr.at<double>(1,2);
    rMatrTrans1.at<double>(2,2) = rMatr.at<double>(0,2);

    // - 90 z 
    rMatrTrans2.at<double>(0,0) = rMatrTrans1.at<double>(1,0);
    rMatrTrans2.at<double>(1,0) = - rMatrTrans1.at<double>(0,0);
    rMatrTrans2.at<double>(2,0) = rMatrTrans1.at<double>(2,0);
    rMatrTrans2.at<double>(0,1) = rMatrTrans1.at<double>(1,1);
    rMatrTrans2.at<double>(1,1) = - rMatrTrans1.at<double>(0,1);
    rMatrTrans2.at<double>(2,1) = rMatrTrans1.at<double>(2,1);
    rMatrTrans2.at<double>(0,2) = rMatrTrans1.at<double>(1,2);
    rMatrTrans2.at<double>(1,2) = - rMatrTrans1.at<double>(0,2);
    rMatrTrans2.at<double>(2,2) = rMatrTrans1.at<double>(2,2);

    // - 90 z 
    rMatrTrans3.at<double>(0,0) = rMatrTrans2.at<double>(1,0);
    rMatrTrans3.at<double>(1,0) = - rMatrTrans2.at<double>(0,0);
    rMatrTrans3.at<double>(2,0) = rMatrTrans2.at<double>(2,0);
    rMatrTrans3.at<double>(0,1) = rMatrTrans2.at<double>(1,1);
    rMatrTrans3.at<double>(1,1) = - rMatrTrans2.at<double>(0,1);
    rMatrTrans3.at<double>(2,1) = rMatrTrans2.at<double>(2,1);
    rMatrTrans3.at<double>(0,2) = rMatrTrans2.at<double>(1,2);
    rMatrTrans3.at<double>(1,2) = - rMatrTrans2.at<double>(0,2);
    rMatrTrans3.at<double>(2,2) = rMatrTrans2.at<double>(2,2);

    Rodrigues(rMatrTrans3, rvec);

    return rvecTrans;
}



//////
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);


    if (argc < 2) {
        parser.printMessage();
        return 0;
    }


    // Parser
    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");
    float markerOffset = parser.get<float>("o");
    bool naiveMode = parser.get<bool>("n");
    bool saveResults = parser.has("s");


    // Detector parameters
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;


    // Load video 
    String video;
    VideoCapture inputVideo;

    int waitTime;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }
    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 1000 * 1.0 /inputVideo.get(CAP_PROP_FPS);
        // waitTime = 0; // wait for user for next frame
        cout << "Success: video loaded" << endl;
    } 
    else {
        inputVideo.open(0);
        waitTime = 1;
        cout << "Fail: video not found" << endl;
    }
    if(!inputVideo.isOpened()) {
        cout << "Video could not be opened..." << endl;
        return -1;
    }


    //Select dictionary for markers detection
    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));


    // Load camera parameters
    Mat camMatrix, distCoeffs;

    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }
    
    // Get frame width and height
    int frame_width = inputVideo.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = inputVideo.get(CAP_PROP_FRAME_HEIGHT);
    
    // Adapt camera intrinsic parameters to current resolution
    // fx' = (width'/width) * fx
    // fy' = (height'/height) * fy
    // cx' = (width'/width) * cx
    // cy' = (height'/height) * cy
    if(frame_width < frame_height) {  // TODO if video is vertical
        camMatrix.at<double>(0,0) *= (frame_width/1080);
        camMatrix.at<double>(1,1) *= (frame_height/1920);
        camMatrix.at<double>(0,2) *= (frame_width/1080);
        camMatrix.at<double>(1,2) *= (frame_height/1920);
    }
    else {
        camMatrix.at<double>(0,0) *= (frame_width/1920);
        camMatrix.at<double>(1,1) *= (frame_height/1080);
        camMatrix.at<double>(0,2) *= (frame_width/1920);
        camMatrix.at<double>(1,2) *= (frame_height/1080);
    }

    // Save results to video
    VideoWriter cap;
    if (saveResults) cap.open("demo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
    inputVideo.get(CAP_PROP_FPS), Size(frame_width, frame_height));


    // Save results to file
    ofstream resultfile;

    if (!naiveMode && saveResults) {
        resultfile.open("results_filt.txt");
        if (resultfile.is_open()) {
            cout << "Filtered resulting transformations" << endl;
        }
        else cout << "Unable to open result file" << endl;
    }
    else if (naiveMode && saveResults) {
        resultfile.open("results_unfilt.txt");
        if (resultfile.is_open()) {
           cout << "Unfiltered resulting transformations" << endl;
        }
        else cout << "Unable to open result file" << endl;
    }


    // Load arrow point cloud
    Mat arrow_cloud = cvcloud_load();

    Mat box_cloud = create_bbox(1.0, 1.0, 1.0);



    // Define variables
    double totalTime = 0;
    int totalIterations = 0;

    double abs_tick = (double)getTickCount();
    double delta_t = 0;

    vector<Point2d> arrow1, arrow2, arrow3, arrow4, box1, box2, box3, box4; // vec to print arrow on image plane

    // We have four big markers
    std::vector<double>  t_lost(4, 0); // count seconds from last time marker was seen
    std::vector<double>  t_stable(4, 0); // count seconds from moment markers are consistent
    double thr_lost = 0.1; // TODO threshold in seconds for going into init
    double thr_stable = 0.5; // TODO threshold in seconds for acquiring master pose
    int consist_markers = 3;

    // Weights for averaging final poses
    double alpha_rot = 0.5;
    double alpha_trasl = 0.5;
    std::vector<double> thr_init(3); // TODO angle threshold for markers consistency in INIT
    std::vector<double> thr_noinit(3); // TODO angle threshold for markers consistency AFTER INIT
    thr_init[0] = (sin(M_PI/2.0));
    thr_init[1] = (sin(M_PI/2.0));
    thr_init[2] = (sin(M_PI/3.0));
    thr_noinit[0] = (sin(M_PI/3.0));
    thr_noinit[1] = (sin(M_PI/3.0));
    thr_noinit[2] = (sin(M_PI/4.0));

    
    if(naiveMode) {
        thr_init[0] = thr_init[1] = thr_init[2] = thr_noinit[0] = thr_noinit[1] = thr_noinit[2] = 2.0;
        thr_lost = std::numeric_limits<double>::max();
        thr_stable = 0.0;
    consist_markers = 1.0;
    }

    // One master pose for each group
    vector<Vec3d> rMaster(4);
    vector<Vec3d> tMaster(4);
    // One pose for the whole scene
    Vec3d rScene;
    Vec3d tScene;

    std::vector<bool> init_id(16, false); // check if marker has been seen before


    ////// ---KEY PART--- //////
    while(inputVideo.grab()) {

        double tickk = (double)getTickCount();

        Mat image, imageCopy;
        inputVideo.retrieve(image);
        //cv::resize(image, image, Size(image.cols/2, image.rows/2)); // lower video resolution
    
        // We have 16 markers
        vector<Vec3d> rvecs_ord(16); // store markers' Euler rotation vectors
        vector<Vec3d> tvecs_ord(16); // store markers' translation vectors
        std::vector<bool> detect_id(16, true); // check if marker was detected or not



        cout << "Frame " << totalIterations << endl;
        cout << "abs_tick" << ((double)getTickCount() - abs_tick) / getTickFrequency() << endl;

        double tick = (double)getTickCount();
        double delta = 0;

        vector<int> ids; // markers identified
        vector<vector<Point2f>> corners, rejected;
        vector<Vec3d> rvecs, tvecs; 

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);


        // Compute detection time
        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);

        // reorder rvecs and tvecs into rvecs_ord and tvecs_ord
        for(unsigned int i=0; i<rvecs.size(); i++) {
            rvecs_ord[ids[i]-1] = rvecs[i];
            tvecs_ord[ids[i]-1] = tvecs[i];
        }


        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            // Loop over markers
            for(unsigned int i=0; i<16; i++) {

                // check if marker was detected
                if(rvecs_ord[i][0] == 0.0) { 
                    detect_id[i] = false;
                    continue;
                }

                // if not initialized, go on with other markers
                if(!init_id[i]) {
                    continue;
                }

                else if(!checkDiffRot(rvecs_ord[i], rMaster[ceil(i/4)], thr_init)) {
                    detect_id[i] = false;
                    continue;
                }
                
                aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs_ord[i], tvecs_ord[i], markerLength * 0.5f);
            }


            // Loop over groups
            for(unsigned int i=0; i<4; i++) {

                if(!init_id[i*4]) { // if group needs init

                    cout << "GROUP " << i << endl;
                    cout << "INIT" << endl;

                    std::vector<bool> detect_id_check = checkPoseConsistent(rvecs_ord, detect_id, 3, i, thr_init);

                    for(int j=0; j<16; j++) {
                        detect_id[j] = detect_id_check[j];
                    }

                    int counter=0;

                    for(int j=0; j<4; j++) {
                        if(detect_id[i*4+j]) {
                            counter += 1;
                        } 
                    }

                    cout << "Counter " << counter << endl;

                    if(counter >= consist_markers) { // if n markers are consistent
                        t_stable[i] += delta_t;
                        if(t_stable[i] >= thr_stable) {
                            init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = true;
                            rMaster[i] = computeAvgRot( rvecs_ord, detect_id, i);
                            tMaster[i] = computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset);
                            t_stable[i] = 0;
                        }
                        else {
                            init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                        }
                    }
                    else {
                        t_stable[i] = 0;
                    }
                } // if already init
                else {
                    cout << "GROUP " << i << endl;
                    cout << "NOT INIT" << endl;
                    if(!detect_id[i*4] && !detect_id[i*4+1] && !detect_id[i*4+2] && !detect_id[i*4+3]) {
                        t_lost[i] += delta_t;
                        if(t_lost[i] >= thr_lost) {
                            init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                            t_lost[i] = 0;
                        }
                    }
                    else{
                        rMaster[i] = avgRot(computeAvgRot(rvecs_ord, detect_id, i), rMaster[i], alpha_rot, (1 - alpha_rot));
                        tMaster[i] = avgTrasl(computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset), tMaster[i], alpha_trasl, (1 - alpha_trasl));
                        //rMaster[i] = computeAvgRot( rvecs_ord, detect_id, i);
                        //tMaster[i] = computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset);
                    }
                }
            }

            /*
            projectPoints(arrow_cloud, rMaster[0], tMaster[0], camMatrix, distCoeffs, arrow1);
            projectPoints(arrow_cloud, rMaster[1], tMaster[1], camMatrix, distCoeffs, arrow2);
            projectPoints(arrow_cloud, rMaster[2], tMaster[2], camMatrix, distCoeffs, arrow3);
            projectPoints(arrow_cloud, rMaster[3], tMaster[3], camMatrix, distCoeffs, arrow4);

            for (unsigned int j = 0; j < arrow1.size(); j++)
            {
                if(init_id[0] && (detect_id[0] || detect_id[1] || detect_id[2] || detect_id[3])) {
                    circle(imageCopy, arrow1[j], 1, Scalar(255,0,0), -1);
                }
                if(init_id[4] && (detect_id[0+4] || detect_id[1+4] || detect_id[2+4] || detect_id[3+4])) {
                    circle(imageCopy, arrow2[j], 1, Scalar(0,255,0), -1);
                }
                if(init_id[8] && (detect_id[0+8] || detect_id[1+8] || detect_id[2+8] || detect_id[3+8])) {
                    circle(imageCopy, arrow3[j], 1, Scalar(0,0,255), -1);
                }
                if(init_id[12] && (detect_id[0+12] || detect_id[1+12] || detect_id[2+12] || detect_id[3+12])) {
                    circle(imageCopy, arrow4[j], 1, Scalar(0,0,255), -1);
                }
            }
            */

            tScene = tMaster[0];
            rScene = computeSceneRot(rMaster, detect_id, init_id, 1);

            Vec3d tvec1, tvec2;
            tvec1[0] = 2.2;
            tvec1[1] = 0.0;//1.1;
            tvec1[2] = 0.0;
            tvec2[0] = - 1.3;
            tvec2[1] = 1.2;
            tvec2[2] = 0.0;

            rMaster[0] = rotateAxis(rMaster[0]);

            tvec1 = transformVec(tvec1, rMaster[0], tMaster[0]);
            tvec2 = transformVec(tvec2, rMaster[1], tMaster[1]);

/*
            projectPoints(box_cloud, rMaster[0], tMaster[0], camMatrix, distCoeffs, box1);
            projectPoints(box_cloud, rMaster[1], tMaster[1], camMatrix, distCoeffs, box2);
            projectPoints(box_cloud, rMaster[2], tMaster[2], camMatrix, distCoeffs, box3);
            projectPoints(box_cloud, rMaster[3], tMaster[3], camMatrix, distCoeffs, box4);

*/

            //projectPoints(box_cloud, rScene, tScene, camMatrix, distCoeffs, box1);
            projectPoints(box_cloud, rMaster[0], tvec1, camMatrix, distCoeffs, box1);
            projectPoints(box_cloud, rMaster[1], tvec2, camMatrix, distCoeffs, box2);
            projectPoints(box_cloud, rMaster[2], tMaster[2], camMatrix, distCoeffs, box3);
            projectPoints(box_cloud, rMaster[3], tMaster[3], camMatrix, distCoeffs, box4);

            vector<vector<Point2d>> boxes(4);
            vector<Point2d> avg_box; 

            boxes[0] = box1;
            boxes[1] = box2;
            boxes[2] = box3;
            boxes[3] = box4;

            avg_box = avgBoxes(boxes, init_id);

            
            if(init_id[0] && (detect_id[0] || detect_id[1] || detect_id[2] || detect_id[3])) {
                DrawBox2D(imageCopy, box1, 60, 20, 220);
                DrawBox2D(imageCopy, avg_box, 0, 0, 0);
            }
            if(init_id[4] && (detect_id[0+4] || detect_id[1+4] || detect_id[2+4] || detect_id[3+4])) {
                DrawBox2D(imageCopy, box2, 0, 255, 0);
                DrawBox2D(imageCopy, avg_box, 0, 0, 0);
                }
/*            if(init_id[8] && (detect_id[0+8] || detect_id[1+8] || detect_id[2+8] || detect_id[3+8])) {
                DrawBox2D(imageCopy, box3, 60, 20, 220);
            }
            if(init_id[12] && (detect_id[0+12] || detect_id[1+12] || detect_id[2+12] || detect_id[3+12])) {
                DrawBox2D(imageCopy, box4, 60, 20, 220);
            }
*/
        }

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        if (saveResults) cap.write(imageCopy);

        Mat imageResize;

        cv::resize(imageCopy, imageResize, Size(imageCopy.cols/3,imageCopy.rows/3));
        imshow("resize", imageResize);

        delta = ((double)getTickCount() - tickk) / getTickFrequency();
        delta_t = delta;


        cout << "Stable time " << t_stable[0] << endl;
        cout << t_stable[1] << endl;
        cout << t_stable[2] << endl;
        cout << t_stable[3] << endl;

        cout << "Lost time " << t_lost[0] << endl;
        cout << t_lost[1] << endl;
        cout << t_lost[2] << endl;
        cout << t_lost[3] << endl;


        char key = (char)waitKey(waitTime); 
        if(key == 27) break;
    }

    inputVideo.release();
    if (saveResults) cap.release();
    
    resultfile.close();

    return 0;
}
