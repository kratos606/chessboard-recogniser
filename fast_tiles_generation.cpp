#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace fs = std::filesystem; // For filesystem operations

// Function to compute gradient in x-direction using larger Sobel kernel
cv::Mat gradientx(const cv::Mat &img)
{
    cv::Mat grad_x;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 31);
    return grad_x;
}

// Function to compute gradient in y-direction using larger Sobel kernel
cv::Mat gradienty(const cv::Mat &img)
{
    cv::Mat grad_y;
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 31);
    return grad_y;
}

// Function to check if line differences match expected pattern
bool checkMatch(const std::vector<int> &lineset)
{
    std::vector<int> linediff(lineset.size());
    std::adjacent_difference(lineset.begin(), lineset.end(), linediff.begin());
    linediff.erase(linediff.begin()); // Remove the first element which is not valid

    int x = 0;
    int cnt = 0;
    for (const auto &line : linediff)
    {
        if (std::abs(line - x) < 5)
        {
            cnt++;
        }
        else
        {
            cnt = 0;
            x = line;
        }
    }
    return cnt >= 5;
}

// Function to prune lines near the margins and match expected patterns
std::vector<int> pruneLines(std::vector<int> lineset, int image_dim, int margin = 20)
{
    // Remove lines near the margins
    lineset.erase(
        std::remove_if(lineset.begin(), lineset.end(), [margin, image_dim](int x)
                       { return x <= margin || x >= image_dim - margin; }),
        lineset.end());

    if (lineset.empty())
    {
        return lineset;
    }

    std::vector<int> linediff(lineset.size());
    std::adjacent_difference(lineset.begin(), lineset.end(), linediff.begin());
    linediff.erase(linediff.begin()); // Remove the first element which is not valid

    int x = 0;
    int cnt = 0;
    int start_pos = 0;
    for (size_t i = 0; i < linediff.size(); ++i)
    {
        int line = linediff[i];
        if (std::abs(line - x) < 5)
        {
            cnt++;
            if (cnt >= 5)
            {
                size_t end_pos = i + 2;
                return std::vector<int>(lineset.begin() + start_pos, lineset.begin() + end_pos);
            }
        }
        else
        {
            cnt = 0;
            x = line;
            start_pos = i;
        }
    }
    return lineset;
}

// Function to skeletonize a 1D signal
std::vector<double> skeletonize_1d(const std::vector<double> &arr)
{
    std::vector<double> _arr = arr;

    for (size_t i = 0; i < _arr.size() - 1; ++i)
    {
        if (_arr[i] <= _arr[i + 1])
        {
            _arr[i] = 0;
        }
    }
    for (size_t i = _arr.size() - 1; i > 0; --i)
    {
        if (_arr[i - 1] > _arr[i])
        {
            _arr[i] = 0;
        }
    }
    return _arr;
}

// Function to perform 1D convolution
std::vector<double> convolve1D(const std::vector<double> &signal, const std::vector<double> &kernel)
{
    int signal_size = signal.size();
    int kernel_size = kernel.size();
    int conv_size = signal_size + kernel_size - 1;
    std::vector<double> conv(conv_size, 0.0);

    for (int i = 0; i < conv_size; ++i)
    {
        int kmin = (i >= kernel_size - 1) ? i - (kernel_size - 1) : 0;
        int kmax = (i < signal_size - 1) ? i : signal_size - 1;
        for (int j = kmin; j <= kmax; ++j)
        {
            conv[i] += signal[j] * kernel[i - j];
        }
    }
    // Trim to original signal size
    int trim = (kernel_size - 1) / 2;
    std::vector<double> result(signal_size);
    for (int i = 0; i < signal_size; ++i)
    {
        result[i] = conv[i + trim];
    }
    return result;
}

// Function to get chessboard lines from Hough transform signals
std::tuple<std::vector<int>, std::vector<int>, bool> getChessLines(const std::vector<double> &hdx, const std::vector<double> &hdy,
                                                                   double hdx_thresh, double hdy_thresh, const cv::Size &image_shape)
{
    // Generate Gaussian window
    int window_size = 21;
    double sigma = 8.0;
    cv::Mat gaussMat = cv::getGaussianKernel(window_size, sigma, CV_64F);
    std::vector<double> gausswin(gaussMat.rows);
    for (int i = 0; i < gaussMat.rows; ++i)
    {
        gausswin[i] = gaussMat.at<double>(i, 0);
    }

    // Threshold signals
    std::vector<double> hdx_thresh_binary(hdx.size());
    std::vector<double> hdy_thresh_binary(hdy.size());

    std::transform(hdx.begin(), hdx.end(), hdx_thresh_binary.begin(),
                   [hdx_thresh](double val)
                   { return val > hdx_thresh ? 1.0 : 0.0; });
    std::transform(hdy.begin(), hdy.end(), hdy_thresh_binary.begin(),
                   [hdy_thresh](double val)
                   { return val > hdy_thresh ? 1.0 : 0.0; });

    // Blur signals using convolution with Gaussian window
    std::vector<double> blur_x = convolve1D(hdx_thresh_binary, gausswin);
    std::vector<double> blur_y = convolve1D(hdy_thresh_binary, gausswin);

    // Skeletonize signals
    std::vector<double> skel_x = skeletonize_1d(blur_x);
    std::vector<double> skel_y = skeletonize_1d(blur_y);

    // Find line positions
    std::vector<int> lines_x;
    std::vector<int> lines_y;

    for (size_t i = 0; i < skel_x.size(); ++i)
    {
        if (skel_x[i] > 0)
            lines_x.push_back(static_cast<int>(i));
    }
    for (size_t i = 0; i < skel_y.size(); ++i)
    {
        if (skel_y[i] > 0)
            lines_y.push_back(static_cast<int>(i));
    }

    // Prune lines
    lines_x = pruneLines(lines_x, image_shape.width);
    lines_y = pruneLines(lines_y, image_shape.height);

    // Check if lines match expected pattern
    bool is_match = (lines_x.size() == 7) && (lines_y.size() == 7) &&
                    checkMatch(lines_x) && checkMatch(lines_y);

    return {lines_x, lines_y, is_match};
}

// Function to extract chessboard tiles from the image
std::vector<cv::Mat> getChessTiles(const cv::Mat &img, const std::vector<int> &lines_x, const std::vector<int> &lines_y)
{
    int stepx = static_cast<int>(std::round(static_cast<double>(lines_x[1] - lines_x[0])));
    int stepy = static_cast<int>(std::round(static_cast<double>(lines_y[1] - lines_y[0])));

    // Pad the image if necessary
    int padl_x = 0, padr_x = 0, padl_y = 0, padr_y = 0;
    if (lines_x.front() - stepx < 0)
    {
        padl_x = std::abs(lines_x.front() - stepx);
    }
    if (lines_x.back() + stepx > img.cols - 1)
    {
        padr_x = lines_x.back() + stepx - img.cols + 1;
    }
    if (lines_y.front() - stepy < 0)
    {
        padl_y = std::abs(lines_y.front() - stepy);
    }
    if (lines_y.back() + stepy > img.rows - 1)
    {
        padr_y = lines_y.back() + stepy - img.rows + 1;
    }
    cv::Mat img_padded;
    cv::copyMakeBorder(img, img_padded, padl_y, padr_y, padl_x, padr_x, cv::BORDER_REPLICATE);

    std::vector<int> setsx{lines_x.front() - stepx + padl_x};
    std::vector<int> setsy{lines_y.front() - stepy + padl_y};
    for (const auto &x : lines_x)
        setsx.push_back(x + padl_x);
    setsx.push_back(lines_x.back() + stepx + padl_x);
    for (const auto &y : lines_y)
        setsy.push_back(y + padl_y);
    setsy.push_back(lines_y.back() + stepy + padl_y);

    std::vector<cv::Mat> squares;
    for (int j = 0; j < 8; ++j)
    {
        for (int i = 0; i < 8; ++i)
        {
            int x1 = setsx[i];
            int x2 = setsx[i + 1];
            int y1 = setsy[j];
            int y2 = setsy[j + 1];
            // Adjust sizes to ensure squares are of equal size
            if ((x2 - x1) != stepx)
            {
                x2 = x1 + stepx;
            }
            if ((y2 - y1) != stepy)
            {
                y2 = y1 + stepy;
            }
            cv::Mat square = img_padded(cv::Range(y1, y2), cv::Range(x1, x2));
            squares.push_back(square);
        }
    }
    return squares;
}

// Function to process an individual image
void process_image(const std::string &image_path, const std::string &output_dir)
{
    // Load the image
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cout << "Failed to load image: " << image_path << std::endl;
        return;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Preprocessing
    cv::Mat equ;
    cv::equalizeHist(gray, equ);
    cv::Mat norm_image;
    equ.convertTo(norm_image, CV_32F, 1.0 / 255.0);

    // Compute the gradients
    cv::Mat grad_x = gradientx(norm_image);
    cv::Mat grad_y = gradienty(norm_image);

    // Clip the gradients
    cv::Mat Dx_pos = cv::max(grad_x, 0);
    cv::Mat Dx_neg = cv::max(-grad_x, 0);
    cv::Mat Dy_pos = cv::max(grad_y, 0);
    cv::Mat Dy_neg = cv::max(-grad_y, 0);

    // Compute the Hough transform
    cv::Mat sum_Dx_pos, sum_Dx_neg, sum_Dy_pos, sum_Dy_neg;
    cv::reduce(Dx_pos, sum_Dx_pos, 0, cv::REDUCE_SUM, CV_64F);
    cv::reduce(Dx_neg, sum_Dx_neg, 0, cv::REDUCE_SUM, CV_64F);
    cv::reduce(Dy_pos, sum_Dy_pos, 1, cv::REDUCE_SUM, CV_64F);
    cv::reduce(Dy_neg, sum_Dy_neg, 1, cv::REDUCE_SUM, CV_64F);

    double norm_factor_x = static_cast<double>(norm_image.rows * norm_image.rows);
    double norm_factor_y = static_cast<double>(norm_image.cols * norm_image.cols);

    std::vector<double> hough_Dx;
    for (int i = 0; i < sum_Dx_pos.cols; ++i)
    {
        double pos_val = sum_Dx_pos.at<double>(0, i);
        double neg_val = sum_Dx_neg.at<double>(0, i);
        double val = (pos_val * neg_val) / norm_factor_x;
        hough_Dx.push_back(val);
    }
    std::vector<double> hough_Dy;
    for (int i = 0; i < sum_Dy_pos.rows; ++i)
    {
        double pos_val = sum_Dy_pos.at<double>(i, 0);
        double neg_val = sum_Dy_neg.at<double>(i, 0);
        double val = (pos_val * neg_val) / norm_factor_y;
        hough_Dy.push_back(val);
    }

    // Adaptive thresholding
    int a = 1;
    bool is_match = false;
    std::vector<int> lines_x;
    std::vector<int> lines_y;

    while (a < 5)
    {
        double threshold_x = *std::max_element(hough_Dx.begin(), hough_Dx.end()) * (static_cast<double>(a) / 5.0);
        double threshold_y = *std::max_element(hough_Dy.begin(), hough_Dy.end()) * (static_cast<double>(a) / 5.0);

        std::tie(lines_x, lines_y, is_match) = getChessLines(hough_Dx, hough_Dy, threshold_x, threshold_y, norm_image.size());

        if (is_match)
        {
            break;
        }
        else
        {
            a++;
        }
    }

    if (is_match)
    {
        std::cout << "7 horizontal and vertical lines found, slicing up squares" << std::endl;
        std::vector<cv::Mat> squares = getChessTiles(gray, lines_x, lines_y);
        std::cout << "Tiles generated: (" << squares[0].rows << "x" << squares[0].cols << ") * " << squares.size() << std::endl;

        // Extract filename and FEN (assuming filename is FEN)
        std::string img_filename = fs::path(image_path).filename().string();
        std::string fen = fs::path(img_filename).stem().string();
        std::string img_save_dir = (fs::path(output_dir) / fen).string();

        if (!fs::exists(img_save_dir))
        {
            fs::create_directories(img_save_dir);
            std::cout << "Created dir " << img_save_dir << std::endl;
        }

        std::string letters = "ABCDEFGH";
        for (size_t i = 0; i < squares.size(); ++i)
        {
            std::string filename = fen + "_" + letters[i % 8] + std::to_string(8 - (i / 8)) + ".png";
            std::string save_path = (fs::path(img_save_dir) / filename).string();
            // Resize to 32x32 and save
            cv::Mat resized;
            cv::resize(squares[i], resized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
            cv::imwrite(save_path, resized);
        }
    }
    else
    {
        std::cout << "No squares to save for " << image_path << std::endl;
    }
}

// Main code
// Thread-safe queue for task distribution
template <typename T>
class SafeQueue
{
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;

public:
    void push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(item);
        lock.unlock();
        cond.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.empty())
        {
            return false;
        }
        item = queue.front();
        queue.pop();
        return true;
    }

    bool empty()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.empty();
    }

    void wait_and_pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty())
        {
            cond.wait(lock);
        }
        item = queue.front();
        queue.pop();
    }
};

// Worker function for thread pool
void worker(SafeQueue<std::pair<std::string, std::string>> &task_queue, std::mutex &cout_mutex)
{
    std::pair<std::string, std::string> task;
    while (task_queue.pop(task))
    {
        process_image(task.first, task.second);
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Processed: " << task.first << std::endl;
        }
    }
}

// Main function with threading
int main()
{
    try
    {
        std::string input_folder = "train_images";
        std::string output_folder = "tiles";

        fs::create_directories(output_folder);

        SafeQueue<std::pair<std::string, std::string>> task_queue;
        std::mutex cout_mutex;

        // Populate the task queue
        for (const auto &entry : fs::directory_iterator(input_folder))
        {
            if (entry.is_regular_file())
            {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
                {
                    std::string image_path = entry.path().string();
                    task_queue.push({image_path, output_folder});
                }
            }
        }

        // Create thread pool
        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i)
        {
            threads.emplace_back(worker, std::ref(task_queue), std::ref(cout_mutex));
        }

        // Wait for all threads to complete
        for (auto &thread : threads)
        {
            thread.join();
        }

        std::cout << "All images processed." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}