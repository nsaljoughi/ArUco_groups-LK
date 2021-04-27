#include "functions.h"

using namespace std; using namespace cv;

namespace { const char* about = "Basic marker detection";
            const char* keys  =
	     "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
	     "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, " 
	     "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
	     "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}" 
	     "{v|       | Input from video file, if ommited, input comes from camera }"
	     "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
	     "{c        |	| Camera intrinsic parameters. Needed for camera pose }"
	     "{l        |	| Marker side lenght (in meters). Needed for correct scale in camera pose }"
	     "{o        |       | Offset between markers (in meters)}"
	     "{dp       |       | File of marker detector parameters }"
	     "{r        | false	| show rejected	candidates too}"
	     "{n        | false | Naive mode (no stabilization)}"
	     "{s        |       | Save results}"
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
    bool saveResults = parser.has("s");
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
    cout << "Frame size: " << frame_width << "x" << frame_height << endl;
    
    // Save results to video
    VideoWriter cap;
    if (saveResults) {
        cap.open("demo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                inputVideo.get(CAP_PROP_FPS), Size(frame_width, frame_height));
    }


    // Save results to file
    ofstream resultfile;

    if (!naiveMode && saveResults) {
        resultfile.open("results_filt.txt");
        if (resultfile.is_open()) {
            cout << "Filtered resulting transformations" << endl;
        }
        else {
            cout << "Unable to open result file" << endl;
        }
    }
    else if (naiveMode && saveResults) {
        resultfile.open("results_unfilt.txt");
        if (resultfile.is_open()) {
           cout << "Unfiltered resulting transformations" << endl;
        }
        else {
            cout << "Unable to open result file" << endl;
        }
    }


    // Load arrow point cloud
    Mat arrow_cloud = cvcloud_load();

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
    double thr_lost = 2; // TODO threshold in seconds for going into init
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

    std::vector<bool> init_id(16, false); // check if marker has been seen before
    Vec3d a_avg, b_avg, c_avg, d_avg;
    
    bool average = false; //flag to decide whether to average or not


    ////// ---KEY PART--- //////
    while(inputVideo.grab()) {

        double tickk = (double)getTickCount();

        Mat image, imageGray, imageCopy;
        inputVideo.retrieve(image);
	cvtColor(image, imageGray, COLOR_BGR2GRAY); // we work with grayscale images
    
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
        aruco::detectMarkers(imageGray, dictionary, corners, ids, detectorParams, rejected);

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
                    Mat rotationMat = Mat::zeros(3, 3, CV_64F);
                    Rodrigues(rvecs_ord[i], rotationMat); //convert rodrigues angles into rotation matrix
                    cout << "Marker is wrong! " << endl;
                    cout <<  "Rvec matrix: " << endl;
                    cout << "[" ;
                    for(int i = 0; i<3; i++) {
                        for(int j=0; j<3; j++) {
                            cout << rotationMat.at<double>(i,j) << " ";
                        }
                        cout << endl;
                    }
                    cout << "]" << endl;
                    cout << "rMaster matrix: " << endl; 
                    Mat rotationMatM = Mat::zeros(3, 3, CV_64F);
                    Rodrigues(rMaster[ceil(i/4)], rotationMatM); //convert rodrigues angles into rotation matrix
                    cout << "[" ;
                    for(int i = 0; i<3; i++) {
                        for(int j=0; j<3; j++) {
                            cout << rotationMatM.at<double>(i,j) << " ";
                        }
                        cout << endl;
                    }
                    cout << "]" << endl;

                    detect_id[i] = false;
                    continue;
                }
                
                aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs_ord[i], tvecs_ord[i], markerLength * 0.5f);
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
                        rMaster[i] = avgRot(computeAvgRot(rvecs_ord, detect_id, i), rMaster[i], alpha_rot, (1 - alpha_rot));
                        tMaster[i] = avgTrasl(computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset), tMaster[i], alpha_trasl, (1 - alpha_trasl));
                        //rMaster[i] = computeAvgRot( rvecs_ord, detect_id, i);
                        //tMaster[i] = computeAvgTrasl(tvecs_ord, rvecs_ord, detect_id, i, markerLength, markerOffset);
                    }
                }
            }

            
	    vector<Vec3d> avg_points = computeAvgBoxes(rMaster, tMaster, init_id, scene);
	    vector<double> weights={0.5,0.5}; //weights for past and current frame

	    combineBoxes(camMatrix, distCoeffs, box_cloud, boxes, init_id, avg_points, weights, average, scene);
	    
	    drawToImg(imageCopy, boxes, init_id, scene);
	}
        else {
            for(unsigned int i=0; i<4; i++) {
                if(init_id[i*4]) {
                    t_lost[i] += delta_t;
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

        if (saveResults) cap.write(imageCopy);

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

	Mat rMat1 = Mat::zeros(3, 3, CV_64F);
	Rodrigues(rMaster[1], rMat1);

	resultfile << rMat1.at<double>(0,0) << " " << rMat1.at<double>(0,1) << " " << rMat1.at<double>(0,2) << " " << tMaster[1][0] << " " << rMat1.at<double>(1,0) << " " << rMat1.at<double>(1,1) << " " << rMat1.at<double>(1,2) << " " << tMaster[1][1] << " "  << rMat1.at<double>(2,0) << " " << rMat1.at<double>(2,1) << " " << rMat1.at<double>(2,2) << " " << tMaster[1][2] << endl;

        cout << "///////////////////////////////////" << endl;


        char key = (char)waitKey(waitTime); 
        if(key == 27) break;
    }
    
    inputVideo.release();
    if (saveResults) cap.release();
    
    resultfile.close();

    return 0;
}
