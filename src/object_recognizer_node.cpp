#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
public:
    ColorStage(ros::NodeHandle nh, cv::Mat hist, cv::Mat back_hist) : it(nh), obj_hist(hist), back_hist(back_hist) {
        image_sub = it.subscribe("/softkinetic_camera/color", 1, &ColorStage::process_frame, this);
    }

    void process_frame(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr img_ptr;

        try {
            img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch(cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::imshow("Camera", img_ptr->image);
        cv::waitKey(3);

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

        cv::imshow("Output", binary);
        cv::imshow("HIST", obj_backhist);
        cv::imshow("BAKC HIST", back_backhist);
        cv::waitKey(3);
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
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);

    if(!fs.isOpened()) {
        std::cout << "Failed to open histogram file\n";
        return -1;
    }

    fs["hist"] >> hist;
    fs["back_hist"] >> backHist;


    ColorStage cs(nh, hist, backHist);
    ros::spin();
}
