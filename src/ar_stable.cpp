#include "functions.h"

using namespace std;
using namespace cv;

#define MAX_FRAME 973
#define MIN_NUM_FEAT 1000

namespace { const char* about = "Basic marker detection";
            const char* keys  =
                "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
	            "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, " 
	            "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
	            "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}" 
	            "{c        |	    | Camera intrinsic parameters. Needed for camera pose }"
	            "{l        |    	| Marker side lenght (in meters). Needed for correct scale in camera pose }"
	            "{o        |        | Offset between markers (in meters)}"
	            "{dp       |        | File of marker detector parameters }"
	            "{u        |        | Use-case / scenario (0, 1, 2, 3, 4, 5)}";
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
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");
    float markerOffset = parser.get<float>("o");
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
    
    detectorParams->adaptiveThreshConstant=0;
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
    box_cloud = create_bbox(1.0,1.0,1.0);

    // Define variables
    double totalTime = 0;
    int totalIterations = 0;
    double abs_tick = (double)getTickCount();
    double delta_t = 0;

    vector<vector<Point2d>> boxes(8);

    // We have four big markers
    std::vector<double>  t_lost(4, 0); // count seconds from last time marker was seen
    std::vector<double>  t_stable(4, 0); // count seconds from moment markers are consistent
    double thr_stable = 0.5; // TODO threshold in seconds for acquiring master pose
    int consist_markers = 3;

    std::vector<double> thr_init(3); // TODO angle threshold for markers consistency in INIT
    std::vector<double> thr_noinit(3); // TODO angle threshold for markers consistency AFTER INIT
    thr_init[0] = (sin(M_PI/12.0));
    thr_init[1] = (sin(M_PI/12.0));
    thr_init[2] = (sin(M_PI/12.0));
    thr_noinit[0] = (sin(M_PI/12.0));
    thr_noinit[1] = (sin(M_PI/12.0));
    thr_noinit[2] = (sin(M_PI/12.0));

    // One master pose for each group
    vector<Vec3d> rMaster(4);
    vector<Vec3d> tMaster(4);

    std::vector<bool> init_id(16, false); // check if marker has been seen before
    
    bool average = false; //flag to decide whether to average or not
    char filename[100];
    char resultname[100];
    
    Mat prevImage, currImage, imageCopy;

    // Visual odometry
    vector<bool> noo(4,false);
    Mat H; // homography to project box in next frame
    vector<Point2f> prevFeatures, currFeatures; // features used for tracking
    vector<vector<Point2f>> group_corners(4, vector<Point2f>(4)); // corners of markers' group in 2D


    ////// ---KEY PART--- //////
    for(int numFrame = 1; numFrame < MAX_FRAME; numFrame++) {
        sprintf(filename, "/home/nicola/pama_marker/videos/frames/vid9/%06d.jpg", numFrame);
        sprintf(resultname, "/home/nicola/pama_marker/videos/out_frames/vid9/%06d.jpg", numFrame);
    
        double tickk = (double)getTickCount();
    
        //Mat currImage_dist  = imread(filename);
        Mat currImage_c = imread(filename);
        //undistort(currImage_dist, currImage_c, camMatrix, distCoeffs); 
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
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, Mat::zeros(8, 1, CV_64F), rvecs, tvecs);

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
                aruco::drawAxis(imageCopy, camMatrix, Mat::zeros(8, 1, CV_64F), rvecs_ord[i], tvecs_ord[i], markerLength * 0.5f);
            }

            // If at least one of the markers' group has been initialized, 
            // start computing visual odometry
            if(init_id[0]||init_id[4]||init_id[8]||init_id[12]) {
                cout << "Computing the homography H..." << endl;
                H = findHomography(prevFeatures, currFeatures, RANSAC);
            }

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
                } 
                else { // if already init
                    cout << "GROUP " << i << " is already initialized." << endl;
    
                    if(!detect_id[i*4] && !detect_id[i*4+1] && !detect_id[i*4+2] && !detect_id[i*4+3]) {
                        cout << "After check, none of the detected markers was found consistent!" << endl;
                        t_lost[i] += delta_t;

                        noo[i] = true;
                        group_corners[i] = getNewGroupCorners(imageCopy, group_corners[i], H);    
                        
                        getNewBoxes(boxes, H, scene);

                       if(markersOutsideFrame(imageCopy, group_corners[i])) {
                            init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                            t_lost[i] = 0; 
                       }
                       average = false;
                    }
                    else{
                        rMaster[i] = computeAvgRot( rvecs_ord, detect_id, i);
                        tMaster[i] = computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset);        
                    }
                }
            }
            
            if(!noo[0] && !noo[1] && !noo[2] && !noo[3]) {
                vector<Vec3d> avg_points = computeAvgBoxes(rMaster, tMaster, init_id, scene);
                vector<double> weights={0.2,0.8}; //weights for past and current frame
                combineBoxes(camMatrix, Mat::zeros(8, 1, CV_64F), box_cloud, boxes, init_id, avg_points, weights, average, scene); //TODO
            }

	        drawToImg(imageCopy, boxes, init_id, scene);
            for (int i=0; i<4; i++) {
                if(init_id[i*4] && !noo[i]) { // TODO
                    group_corners[i] = drawGroupBorders(imageCopy, tMaster[i], rMaster[i], camMatrix, Mat::zeros(8, 1, CV_64F), markerLength, markerOffset, scene);
                }
            }
            noo[0] = noo[1] = noo[2] = noo[3] = false;
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
                if(init_id[0]||init_id[4]||init_id[8]||init_id[12]) {
                    cout << "Computing the homography H..." << endl;
                    H = findHomography(prevFeatures, currFeatures, RANSAC);
                }
            }
            for(unsigned int i=0; i<4; i++) {
                if(init_id[i*4]) {
                    cout << "No marker was detected, using homography!" << endl;
                    t_lost[i] += delta_t;

                    group_corners[i] = getNewGroupCorners(imageCopy, group_corners[i], H); 

                    getNewBoxes(boxes, H, scene);

                    if(markersOutsideFrame(imageCopy, group_corners[i])) {
                        init_id[i*4] = init_id[i*4+1] = init_id[i*4+2] = init_id[i*4+3] = false;
                        t_lost[i] = 0;
                    }
                    average = false;
                }
            }	      
            drawToImg(imageCopy, boxes, init_id, scene); 
        }   

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

        char key = (char)waitKey(1); 
        if(key == 27) break;
    }
    return 0;
}
