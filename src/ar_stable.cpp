#include "functions.h"

using namespace std;
using namespace cv;

#define MAX_FRAME 450
#define MIN_NUM_FEAT 1000

namespace { const char* about = "Basic marker detection";
            const char* keys  =
	     "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
	     "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, " 
	     "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
	     "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}" 
	     "{c        |	    | Camera intrinsic parameters. Needed for camera pose }"
	     "{l        |    	| Marker side lenght (in meters). Needed for correct scale in camera pose }"
	     "{o        |       | Offset between markers (in meters)}"
	     "{dp       |       | File of marker detector parameters }"
	     "{r        | false	| show rejected	candidates too}"
	     "{n        | false | Naive mode (no stabilization)}"
	     "{u        |       | Use-case / scenario (0, 1, 2, 3, 4, 5)}";
}


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
    int scene = parser.get<int>("u");


    // Detector parameters
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    
    //detectorParams->adaptiveThreshWinSizeMin=3;
    //detectorParams->adaptiveThreshWinSizeMin=23;
    //detectorParams->adaptiveThreshWinSizeStep=10;
    detectorParams->adaptiveThreshConstant=0;
    //detectorParams->minMarkerPerimeterRate=0.03;
    //detectorParams->maxMarkerPerimeterRate=1.0;
    //detectorParams->polygonalApproxAccuracyRate=0.05;
    //detectorParams->minCornerDistanceRate=0.05;
    //detectorParams->minMarkerDistanceRate=0.05;
    //detectorParams->minDistanceToBorder=3;
    //detectorParams->markerBorderBits=1;
    //detectorParams->minOtsuStdDev=5.0;
    //detectorParams->perspectiveRemovePixelPerCell=4;
    //detectorParams->perspectiveRemoveIgnoredMarginPerCell=0.33;
    //detectorParams->maxErroneousBitsInBorderRate=0.35;
    //detectorParams->errorCorrectionRate=0.6;
    detectorParams->cornerRefinementMethod=aruco::CORNER_REFINE_CONTOUR;
    detectorParams->cornerRefinementWinSize=5;
    detectorParams->cornerRefinementMaxIterations=30;
    detectorParams->cornerRefinementMinAccuracy=0.1;


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
    

    // Generate box point cloud
    Mat box_cloud;
    if(scene==3) box_cloud = create_bbox(3.0, 2.0, 1.0);
    else if(scene==1 || scene==5) box_cloud = create_bbox(1.0,1.0,1.0);


    // Define variables
    double totalTime = 0;
    int totalIterations = 0;
    double abs_tick = (double)getTickCount();
    double delta_t = 0;

    vector<vector<Point2d>> boxes(8);

    // We have four big markers
    std::vector<double>  t_lost(4, 0); // count seconds from last time marker was seen
    std::vector<double>  t_stable(4, 0); // count seconds from moment markers are consistent
    double thr_lost = 60; // TODO threshold in seconds for going into init
    double thr_stable = 0.5; // TODO threshold in seconds for acquiring master pose
    int consist_markers = 3;

    // Weights for averaging final poses
    double alpha_rot = 0.3;
    double alpha_trasl = 0.3;
    std::vector<double> thr_init(3); // TODO angle threshold for markers consistency in INIT
    std::vector<double> thr_noinit(3); // TODO angle threshold for markers consistency AFTER INIT
    thr_init[0] = (sin(M_PI/12.0));
    thr_init[1] = (sin(M_PI/12.0));
    thr_init[2] = (sin(M_PI/12.0));
    thr_noinit[0] = (sin(M_PI/12.0));
    thr_noinit[1] = (sin(M_PI/12.0));
    thr_noinit[2] = (sin(M_PI/12.0));
    if(naiveMode) {
        thr_init[0] = thr_init[1] = thr_init[2] = thr_noinit[0] = thr_noinit[1] = thr_noinit[2] = 2.0;
        thr_lost = std::numeric_limits<double>::max();
        thr_stable = 0.0;
        consist_markers = 1.0;
    }

    // One master pose for each group
    vector<Vec3d> rMaster(4);
    vector<Vec3d> tMaster(4);
    vector<Vec3d> tMaster_old(4);

    std::vector<bool> init_id(16, false); // check if marker has been seen before
    
    bool average = false; //flag to decide whether to average or not
    char filename[100];
    char resultname[100];
    
    Mat prevImage, currImage, imageCopy;

    // Visual odometry
    double scale = 1.0;
    Mat E, R, t, mask; // estimated camera motion
    Mat H; // homography to project box in next frame
    vector<Point2f> prevFeatures, currFeatures; // features used for tracking
    std::vector<Point2f> group_corners(4);


    ////// ---KEY PART--- //////
    for(int numFrame = 1; numFrame < MAX_FRAME; numFrame++) {
	sprintf(filename, "/home/nicola/pama_marker/videos/frames/pama5_bis/%06d.jpg", numFrame);
	sprintf(resultname, "/home/nicola/pama_marker/videos/out_frames/pama5_bis/%06d.jpg", numFrame);
    
    double tickk = (double)getTickCount();
       
	Mat currImage_c  = imread(filename);	
	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY); // we work with grayscale images
	vector<uchar> status;

	// Visula odometry: detect features and track
	if(numFrame==1) {
		featureDetection(currImage, currFeatures);
	}
	else {
		cout << "# of features detected = " << prevFeatures.size() << endl;
		featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
		if (prevFeatures.size() < MIN_NUM_FEAT) {
			cout << "Too few features remained...redetecting" << endl;
			featureDetection(prevImage, prevFeatures);
			featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
		}
	}

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
        aruco::detectMarkers(currImage, dictionary, corners, ids, detectorParams, rejected);

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
        currImage_c.copyTo(imageCopy);

        // reorder rvecs and tvecs into rvecs_ord and tvecs_ord
        for(unsigned int i=0; i<rvecs.size(); i++) {
            rvecs_ord[ids[i]-1] = rvecs[i];
            tvecs_ord[ids[i]-1] = tvecs[i];
        }


        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            // Loop over markers
            for(unsigned int i=0; i<16; i++) {
                cout << "Group " << ceil(i/4) << endl;

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

/*
            // If at least one of the markers' group has been initialized, 
            // start computing visual odometry
            if(init_id[0]||init_id[4]||init_id[8]) {
                cout << "Computing the essential matrix E (for now up to a scale...)" << endl; //TODO
                E = findEssentialMat(currFeatures, prevFeatures, camMatrix, RANSAC, 0.999, 1.0, mask);
                recoverPose(E, currFeatures, prevFeatures, camMatrix, R, t, mask);
                H = findHomography(prevFeatures, currFeatures, RANSAC);
            }
*/
            // Loop over groups
            for(unsigned int i=0; i<4; i++) {
                if(!init_id[i*4]) { // if group needs init
                    cout << "GROUP " << i << ": initializing..." << endl;

                    std::vector<bool> detect_id_check = checkPoseConsistent(rvecs_ord, detect_id, 4, i, thr_init);

                    for(int j=0; j<16; j++) {
                        detect_id[j] = detect_id_check[j];
                    }

                    int counter=0;

                    for(int j=0; j<4; j++) {
                        if(detect_id[i*4+j]) {
                            counter += 1;
                        } 
                    }

                    cout << "In group " << i << " there are " << counter << " consistent markers" << endl;

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
                    cout << "GROUP " << i << " is already initialized." << endl;

                    if(!detect_id[i*4] && !detect_id[i*4+1] && !detect_id[i*4+2] && !detect_id[i*4+3]) {
                        t_lost[i] += delta_t;
                        if(t_lost[i] >= thr_lost) {
                            init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                            t_lost[i] = 0;
                            average=false;    
                        }
                    }
                    else{
                        //rMaster[i] = avgRot(computeAvgRot(rvecs_ord, detect_id, i), rMaster[i], alpha_rot, (1 - alpha_rot));
                        //tMaster[i] = avgTrasl(computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset), tMaster[i], alpha_trasl, (1 - alpha_trasl));
                        //combineTransVO(rMaster[i], tMaster[i], T_homo);
                        rMaster[i] = computeAvgRot( rvecs_ord, detect_id, i);
                        tMaster[i] = computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset);                    }
                }
            }

            
	    vector<Vec3d> avg_points = computeAvgBoxes(rMaster, tMaster, init_id, scene);
	    vector<double> weights={0.5,0.5}; //weights for past and current frame

	    combineBoxes(camMatrix, distCoeffs, box_cloud, boxes, init_id, avg_points, weights, average, scene);
	    
	    drawToImg(imageCopy, boxes, init_id, scene);
        if(init_id[4]) group_corners = drawGroupBorders(imageCopy, tMaster[1], rMaster[1], camMatrix, distCoeffs, markerLength, markerOffset);
	}
        else {
           	if(numFrame==1) {
		        featureDetection(currImage, currFeatures);
    	    }
            else {
                cout << "# of features detected = " << prevFeatures.size() << endl;
                featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
                if (prevFeatures.size() < MIN_NUM_FEAT) {
                    cout << "Too few features remained...redetecting" << endl;
                    featureDetection(prevImage, prevFeatures);
                    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
                }
                // If at least one of the markers' group has been initialized, 
                // start computing visual odometry
                if(init_id[0]||init_id[4]||init_id[8]) {
                    cout << "Computing the essential matrix E (for now up to a scale...)" << endl; //TODO
                    E = findEssentialMat(currFeatures, prevFeatures, camMatrix, RANSAC, 0.999, 1.0, mask);
                    recoverPose(E, currFeatures, prevFeatures, camMatrix, R, t, mask);
                    H = findHomography(prevFeatures, currFeatures, RANSAC);
                }
            }
            for(unsigned int i=0; i<4; i++) {
                if(init_id[i*4]) {
                    t_lost[i] += delta_t;
                    vector<Point2f> group_corners_new(4);
                    vector<Point2f> corners0_undist(4);
                    vector<Point2f> corners1_undist(4); 
                    perspectiveTransform(group_corners, group_corners_new, H);

                    // In OpenCV, no homogeneous coordinates but directly [R|t], hence 3*4
                    Mat cam0 = Mat::eye(3, 4, CV_64F);
                    Mat cam1 = Mat::eye(3, 4, CV_64F);
                    createHomoTransformVec(cam0, rMaster[i], tMaster[i]);
                    createHomoTransform(cam1, R, t);
                    Mat group_corners_3d(4, group_corners.size(), CV_64F);

                    // We need to undistort points to triangulate
                    undistortPoints(group_corners, corners0_undist, camMatrix, distCoeffs);
                    undistortPoints(group_corners_new, corners1_undist, camMatrix, distCoeffs);

                    triangulatePoints(cam0, cam1, corners0_undist, corners1_undist, group_corners_3d);
                    for(size_t i=0; i < 4; i++) {
                        cout << "[" << endl;
                        cout << group_corners[i].x << ", " << group_corners[i].y  << endl;
                        cout << "]" << endl;
                    }
                    for(size_t i=0; i < 4; i++) {
                        cout << "[" << endl;
                        cout << group_corners_new[i].x << ", " << group_corners_new[i].y << endl;
                        cout << "]" << endl;
                    }
                    // Get euclidean coordinates <x, y, z, w> -> <x/w, y/w, z/w, w/w> 
                    for(size_t i=0; i < group_corners_3d.rows; i++) {
                        cout << "[" << endl;
                        for(size_t j=0; j<group_corners_3d.cols; j++) {
                            group_corners_3d.at<float>(i,j) /= group_corners_3d.at<float>(i,3); 
                            cout << group_corners_3d.at<float>(i,j) << endl;
                        }
                        cout << "]" << endl;
                    }

                    group_corners = group_corners_new;
                    line(imageCopy, group_corners[0], group_corners[1], Scalar(255, 0, 0), 4);
                    line(imageCopy, group_corners[1], group_corners[2], Scalar(255, 0, 0), 4);
                    line(imageCopy, group_corners[2], group_corners[3], Scalar(255, 0, 0), 4);
                    line(imageCopy, group_corners[3], group_corners[0], Scalar(255, 0, 0), 4);                     

                    if(t_lost[i] >= thr_lost) {
                        init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                        t_lost[i] = 0;
                        average=false;
                    }
                }
            }	      
            drawToImg(imageCopy, boxes, init_id, scene); 
        }   

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        Mat imageResize;

        cv::resize(imageCopy, imageResize, Size(imageCopy.cols/4,imageCopy.rows/4));
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
        cout << "///////////////////////////////////" << endl;
        
        imwrite(resultname, imageCopy);
        
        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        tMaster_old = tMaster;

        char key = (char)waitKey(1); 
        if(key == 27) break;
    }

    return 0;
}
