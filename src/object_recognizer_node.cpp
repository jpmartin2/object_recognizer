#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <math.h>

static const std::string camera_topic = "/softkinetic_camera/color";
static const std::string map_topic = "/softkinetic_camera/registered_depth";
static const std::string output_topic = "/object_recognizer/detected";
static const std::string offset_topic = "/object_recognizer/object_offset";
//static const std::string pose_topic = "/object_recognizer/object_loc";
static const std::string pose_topic = "/simple_traj/target";

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
    image_transport::Subscriber map_sub;
    image_transport::Publisher image_pub;
    ros::Publisher offset_pub;
    ros::Publisher pose_pub;
    tf::TransformListener tf_listener;

    bool haveImageFrame, haveMapFrame;
    sensor_msgs::ImageConstPtr lastImageFrame, lastMapFrame;
public:
    ObjDetector(ros::NodeHandle nh, std::string calib_image_name) :
        it(nh), lastImageFrame(nullptr), lastMapFrame(nullptr),
        haveImageFrame(false), haveMapFrame(false)
    {
        // Setup feature detection/extraction/matching objects
        detector = cv::FeatureDetector::create("ORB");
        extractor = cv::DescriptorExtractor::create("ORB");
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        // Read calibration image
        obj_img = cv::imread(calib_image_name, CV_LOAD_IMAGE_COLOR);
        cv::Mat_<unsigned char> obj_gray(obj_img.size());
        cv::cvtColor(obj_img, obj_gray, CV_BGR2GRAY);

        detector->detect(obj_gray, obj_keypoints);
        extractor->compute(obj_gray, obj_keypoints, obj_descriptors);

        // Subscribe to image/map messages from the camera
        image_sub = it.subscribe(camera_topic, 1, &ObjDetector::process_image_frame, this);
        map_sub   = it.subscribe(map_topic, 1, &ObjDetector::process_map_frame, this);
        image_pub = it.advertise(output_topic, 1);
        pose_pub  = nh.advertise<geometry_msgs::PoseStamped>(pose_topic, 1);
        offset_pub = nh.advertise<geometry_msgs::Point>(offset_topic, 1);
    }

    /**
     * find_object
     */
    tf::Vector3 find_object(const sensor_msgs::ImageConstPtr msg, const sensor_msgs::ImageConstPtr mapMsg) {
        cv_bridge::CvImagePtr img_ptr;

        tf::Vector3 ret(0.0, 0.0, 0.0);

        try {
            img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            if(img_ptr == nullptr) return ret;
        } catch(cv_bridge::Exception& e) {
            ROS_ERROR("Failed to extract opencv image; cv_bridge exception: %s", e.what());
            exit(-1);
        }

        // Camera image converted to grayscale
        cv::Mat img;
        // Color camera image
        cv::Mat& color = img_ptr->image;

        // Perform the Color->Gray conversion
        cv::cvtColor(img_ptr->image, img, CV_BGR2GRAY);

        // Compute keypoints and descriptors
        std::vector<cv::KeyPoint> img_keypoints;
        cv::Mat img_descriptors;

        std::vector<cv::DMatch> matches;

        try {
            detector->detect(img, img_keypoints);
            extractor->compute(img, img_keypoints, img_descriptors);
            // Match keypoints from current frame to calibration image
            matcher->match(obj_descriptors, img_descriptors, matches);
        } catch(...) {

        }
        if(!matches.size()) {
            image_pub.publish(img_ptr->toImageMsg());

            // Show image (TODO: perhaps republish it instead?)
            cv::imshow("OUT", color);
            cv::waitKey(1);
            return ret;
        }

        double max_dist = 0; double min_dist = 100;

        // Calculate min and max distances between keypoints
        //for( int i = 0; i < obj_descriptors.rows; i++ )
        for( int i = 0; i < matches.size(); i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // The determination of what a "good" match is should probably
        // be revised.
        std::vector<cv::DMatch> good_matches;

        for( int i = 0; i < matches.size(); i++ )
        {
            if( matches[i].distance < 55)
                good_matches.push_back( matches[i]);
        }

        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( obj_keypoints[ good_matches[i].queryIdx ].pt );
            scene.push_back( img_keypoints[ good_matches[i].trainIdx ].pt );
        }

        float x_avg = 0;
        float y_avg = 0;

        for(int i = 0; i < scene.size(); i++) {
            x_avg += scene[i].x;
            y_avg += scene[i].y;
        }

        x_avg /= scene.size();
        y_avg /= scene.size();


        if(good_matches.size() >= 30) {
            // Find homography from the calibration image to the current frame.
            cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC );
            // Get the corners from the image_1 ( the object to be "detected" )
            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0);
            obj_corners[1] = cvPoint( obj_img.cols, 0 );
            obj_corners[2] = cvPoint( obj_img.cols, obj_img.rows );
            obj_corners[3] = cvPoint( 0, obj_img.rows );

            std::vector<cv::Point2f> scene_corners(4);

            // Transform object corners from calibration image to current frame
            perspectiveTransform( obj_corners, scene_corners, H);

            // Draw lines around object
            cv::circle(color, cv::Point(x_avg, y_avg), 10, cv::Scalar(0,0,255), -1);
            cv::line( color, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
            cv::line( color, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
            cv::line( color, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );

            cv_bridge::CvImagePtr map_ptr;

            try {
                map_ptr = cv_bridge::toCvCopy(mapMsg, sensor_msgs::image_encodings::TYPE_32FC3);
                if(map_ptr == nullptr) return ret;
            } catch(cv_bridge::Exception& e) {
                ROS_ERROR("Failed to extract opencv image; cv_bridge exception: %s", e.what());
                exit(-1);
            }

            cv::Vec3f out = cv::Vec3f(0.0, 0.0, 0.0);
            int n = 0;

            float min_x = scene_corners[0].x;
            float min_y = scene_corners[0].y;
            float max_x = scene_corners[0].x;
            float max_y = scene_corners[0].y;

            for(cv::Point2f p : scene_corners) {
                if(p.x < min_x) min_x = p.x;
                if(p.x > max_x) max_x = p.x;
                if(p.y < min_y) min_y = p.y;
                if(p.y > max_y) max_y = p.y;
            }


            min_x += 20;
            min_y += 20;
            max_x -= 20;
            max_y -= 20;
            //min_x = std::max(min_x+10, 0.0f);
            //min_y = std::max(min_y+10, 0.0f);
            //max_x = std::min(max_x-10, (float)map_ptr->image.size().width-1);
            //max_y = std::min(max_y-10, (float)map_ptr->image.size().height-1);

            //std::cout <<"x_avg: "<<x_avg<<" y_avg: "<<y_avg<< " minx: "<<min_x<<" maxx: "<<max_x<<" miny: "<<min_y<<" maxy: "<<max_y<<std::endl;

            std::vector<cv::Point2f> test_corners(4);
            test_corners[0] = cvPoint(min_x,min_y);
            test_corners[1] = cvPoint(min_x,max_y);
            test_corners[2] = cvPoint(max_x,max_y);
            test_corners[3] = cvPoint(max_x,min_y);
            cv::line( color, test_corners[0], test_corners[1], cv::Scalar(0, 0, 255), 4 );
            cv::line( color, test_corners[1], test_corners[2], cv::Scalar( 0, 0, 255), 4 );
            cv::line( color, test_corners[2], test_corners[3], cv::Scalar( 0, 0, 255), 4 );
            cv::line( color, test_corners[3], test_corners[0], cv::Scalar( 0, 0, 255), 4 );


            float x_ratio = (float)map_ptr->image.cols/color.cols;
            float y_ratio = (float)map_ptr->image.rows/color.rows;

            for(int i = std::min(map_ptr->image.cols- 1, std::max((int)(min_x*x_ratio), 0)); i < std::max(0, std::min((int)(max_x*x_ratio), map_ptr->image.cols- 1)); i++) {
                for(int j = std::min(map_ptr->image.rows -1, std::max((int)(min_y*y_ratio), 0)); j < std::max(0, std::min((int)(max_y*y_ratio),map_ptr->image.rows -1 )); j++) {
                    cv::Vec3f point = map_ptr->image.at<cv::Vec3f>((int)(j),(int)(i));
                    if(point[0] != point[0] || point[1] != point[1] || point[2] != point[2] || (point[0] == 0 && point[1] == 0 && point[2] == 0)) continue;
                    out += point;
                    n++;
                }
            }
            //std::cout <<"N: "<< n << std::endl;

            out /= n;

            // Camera frame and forearm frame have different axes + an offset
            ret = tf::Vector3(out[1]-0.071/*-0.071*/, -out[0]-0.007/*+.015*//*-0.007*/, out[2]-.005);
            //ret = tf::Vector3(out[2]+0.07, out[0]-0.09, -out[1]);

            geometry_msgs::Point cam_frame_loc;
            cam_frame_loc.x = ret[0];
            cam_frame_loc.y = ret[1];
            cam_frame_loc.z = ret[2];
            offset_pub.publish(cam_frame_loc);
        }

        image_pub.publish(img_ptr->toImageMsg());

        // Show image (TODO: perhaps republish it instead?)
        cv::imshow("OUT", color);
        cv::waitKey(1);


        return ret;
    }

    void detect() {
        if(haveImageFrame && haveMapFrame) {
            tf::Vector3 object_loc = find_object(lastImageFrame, lastMapFrame);

            ros::Time now = ros::Time::now();
            tf::StampedTransform transform;
            bool success = tf_listener.waitForTransform("/base", "/left_hand", now, ros::Duration(2.0));
            if(!success) return;
            tf_listener.lookupTransform("/base", "/left_hand", now, transform);
            // Transform into /base frame
            tf::Vector3 obj_base = transform(object_loc);
            geometry_msgs::Point obj;
            obj.x = obj_base.getX() - 0.08;
            obj.y = obj_base.getY();// + 0.009; // Offset for non-symetric grippers
            obj.z = obj_base.getZ();
            geometry_msgs::Quaternion rot;
            rot.x = 0.0;
            rot.y = 0.7;
            rot.z = 0.0;
            rot.w = 0.7;
            //rot.x = 0.0;
            //rot.y = 0.0;
            //rot.z = 0.0;
            //rot.w = 1.0;
            geometry_msgs::Pose pose;
            pose.position = obj;
            pose.orientation = rot;
            geometry_msgs::PoseStamped stamped;
            std_msgs::Header header;
            header.stamp = now;
            header.frame_id = "/base";
            stamped.pose = pose;
            stamped.header = header;

            pose_pub.publish(stamped);

            // Need to publish/send out object location here.
            haveImageFrame = false;
            haveMapFrame = false;
        }
    }

    void process_image_frame(const sensor_msgs::ImageConstPtr& msg) {
        lastImageFrame = msg;
        haveImageFrame = true;
    }

    void process_map_frame(const sensor_msgs::ImageConstPtr& msg) {
        lastMapFrame = msg;
        haveMapFrame = true;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_recognizer");
    ros::NodeHandle nh;

    if(argc < 2) {
        std::cout << "Missing "<< (2-argc) << " arguments!" << std::endl;
        return -1;
    }

    ObjDetector detector(nh, argv[1]);

    // Rate to check for new messages from the camera
    ros::Rate sampleRate(30); // 30HZ

    while(ros::ok()) {
        ros::spinOnce();
        detector.detect();
        sampleRate.sleep();
    }
}
