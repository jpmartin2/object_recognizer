#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class ObjDetector {
private:
    // Feature detector
    cv::Ptr<cv::FeatureDetector> detector;
    // Feature descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // Feature matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Object calibration image
    cv::Mat obj_img;
    // Object feature keypoints
    std::vector<cv::KeyPoint> obj_keypoints;
    // Object feature descriptors
    cv::Mat obj_descriptors;

    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
public:
    ObjDetector(ros::NodeHandle nh, std::string camera_topic, std::string calib_image_name) : it(nh) {
        // Setup feature detection/extraction/matching objects
        detector = cv::FeatureDetector::create("ORB");
        extractor = cv::DescriptorExtractor::create("FREAK");
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        // Read calibration image
        obj_img = cv::imread(calib_image_name, CV_LOAD_IMAGE_GRAYSCALE);
        detector->detect(obj_img, obj_keypoints);
        extractor->compute(obj_img, obj_keypoints, obj_descriptors);

        // Subscribe to image messages from the camera
        image_sub = it.subscribe(camera_topic, 1, &ObjDetector::process_frame, this);
    }

    void process_frame(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr img_ptr;

        try {
            img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch(cv_bridge::Exception& e) {
            ROS_ERROR("Failed to extract opencv image; cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat img;
        cv::Mat color;
        cv::cvtColor(img_ptr->image, img, CV_BGR2GRAY);
        color = img_ptr->image.clone();
        std::vector<cv::KeyPoint> img_keypoints;
        cv::Mat img_descriptors;

        detector->detect(img, img_keypoints);
        extractor->compute(img, img_keypoints, img_descriptors);

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
        }
        
        //-- Show detected matches

        cv::imshow("OUT", color);
        cv::waitKey(3);

    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_recognizer");
    ros::NodeHandle nh;

    if(argc < 3) {
        std::cout << "Missing "<< (3-argc) << " arguments!" << std::endl;
        return -1;
    }

    ObjDetector detector(nh, argv[1], argv[2]);
    ros::spin();
}
