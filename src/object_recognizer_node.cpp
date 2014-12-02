#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

float h_range[] = {0,256};
float s_range[] = {0,256};
float v_range[] = {0,256};
const float* ranges[] = {h_range, s_range}; //, v_range};
int channels[] = {0,1};

const int THRESH_TYPE = 0; // 0 -> Binary Threshold
const int THRESH = 0;

// Stage of object detection which examines coor
class ColorStage {
private:
    const static double back_weight = 0.82;
    cv::Mat obj_hist; // Object histogram
    cv::Mat back_hist; // background histogram

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Mat obj_descriptors;
    cv::Mat obj_img;
    std::vector<cv::KeyPoint> obj_keypoints;

    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
public:
    ColorStage(ros::NodeHandle nh, std::string calib_image, cv::Mat hist, cv::Mat back_hist) : it(nh), obj_hist(hist), back_hist(back_hist) {
        image_sub = it.subscribe("/softkinetic_camera/color", 1, &ColorStage::process_frame, this);

        detector = cv::FeatureDetector::create("ORB");
        extractor = cv::DescriptorExtractor::create("FREAK");
        obj_img = cv::imread(calib_image, CV_LOAD_IMAGE_GRAYSCALE);
        detector->detect(obj_img, obj_keypoints);
        extractor->compute(obj_img, obj_keypoints, obj_descriptors);
    }

    void process_frame(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr img_ptr;

        try {
            img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch(cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat img;
        cv::Mat color;
        cv::cvtColor(img_ptr->image, img, CV_BGR2GRAY);
        color = img_ptr->image;
        std::vector<cv::KeyPoint> img_keypoints;
        cv::Mat img_descriptors;

        detector->detect(img, img_keypoints);
        extractor->compute(img, img_keypoints, img_descriptors);

        //cv::FlannBasedMatcher matcher;
        //cv::BruteForceMatcher<cv::Hamming> matcher;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<cv::DMatch> matches;
        matcher->match(obj_descriptors, img_descriptors, matches);

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < obj_descriptors.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector<cv::DMatch> good_matches;

        for( int i = 0; i < obj_descriptors.rows; i++ )
        { 
            if( matches[i].distance < 3*min_dist ) { 
                good_matches.push_back( matches[i]); }
        }

        cv::Mat img_matches;
        cv::drawMatches( obj_img, obj_keypoints, img, img_keypoints,
                     good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Localize the object
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( obj_keypoints[ good_matches[i].queryIdx ].pt );
            scene.push_back( img_keypoints[ good_matches[i].trainIdx ].pt );
        }

        cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC );

        if(!H.empty()) {

            //-- Get the corners from the image_1 ( the object to be "detected" )
            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); 
            obj_corners[1] = cvPoint( obj_img.cols, 0 );
            obj_corners[2] = cvPoint( obj_img.cols, obj_img.rows ); 
            obj_corners[3] = cvPoint( 0, obj_img.rows );

            std::vector<cv::Point2f> scene_corners(4);

            perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            cv::line( color, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
            cv::line( color, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
            /*
            cv::line( color, scene_corners[0] + cv::Point2f( obj_img.cols, 0), scene_corners[1] + cv::Point2f( obj_img.cols, 0), cv::Scalar(0, 255, 0), 4 );
            cv::line( color, scene_corners[1] + cv::Point2f( obj_img.cols, 0), scene_corners[2] + cv::Point2f( obj_img.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[2] + cv::Point2f( obj_img.cols, 0), scene_corners[3] + cv::Point2f( obj_img.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[3] + cv::Point2f( obj_img.cols, 0), scene_corners[0] + cv::Point2f( obj_img.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            */
        }
        
        //-- Show detected matches

        cv::imshow("OUT", color);
        cv::waitKey(3);

        /*
        // Save image size
        cv::Size img_size = img_ptr->image.size();

        // Grayscale images where we will store the histogram backprojections.
        cv::Mat_<unsigned char> obj_backhist(img_size);
        cv::Mat_<unsigned char> back_backhist(img_size);

        // Convert image to HSV format, needed for histogram
        cv::Mat_<cv::Vec3b> hsv(img_size);
        cv::cvtColor(img_ptr->image, hsv, CV_RGB2HSV_FULL);

        // Binary output image
        cv::Mat_<unsigned char> binary(img_size);

        // Perform histogram backprojections.
        cv::calcBackProject(&hsv, 1, channels, obj_hist, obj_backhist, ranges);
        cv::calcBackProject(&hsv, 1, channels, back_hist, back_backhist, ranges);

        // 
        binary = obj_backhist > back_weight*back_backhist;

        //cv::imshow("Output", binary);
        //cv::imshow("HIST", obj_backhist);
        //cv::imshow("BAKC HIST", back_backhist);
        cv::waitKey(3);
        */
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_recognizer");
    ros::NodeHandle nh;

    if(argc < 1) {
        std::cout << "Missing argument\n";
        return -1;
    }

    cv::Mat hist, backHist;
    //cv::FileStorage fs(argv[1], cv::FileStorage::READ);

    /*
    if(!fs.isOpened()) {
        std::cout << "Failed to open histogram file\n";
        return -1;
    }

    fs["hist"] >> hist;
    fs["back_hist"] >> backHist;
    */


    ColorStage cs(nh, argv[1], hist, backHist);
    ros::spin();
}
