/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015-2016, Jiri Horner.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Jiri Horner nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

#ifndef MERGING_PIPELINE_H_
#define MERGING_PIPELINE_H_

#include <vector>

#include <geometry_msgs/Transform.h>
#include <nav_msgs/OccupancyGrid.h>

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc/imgproc.hpp"

namespace combine_grids
{
enum class FeatureType { AKAZE, ORB, SURF };

/**
 * @brief Pipeline for merging overlapping occupancy grids
 * @details Pipeline works on internally stored grids.
 */
class MergingPipeline
{
public:
  MergingPipeline()
    : main_map_width_(0.0), main_map_height_(0.0), main_map_id_(0) {
  }
  template <typename InputIt>
  void feed(InputIt grids_begin, InputIt grids_end);
  bool estimateTransforms(FeatureType feature = FeatureType::AKAZE,
                          double confidence = 1.0);
  nav_msgs::OccupancyGrid::Ptr composeGrids();

  std::vector<geometry_msgs::Transform> getTransforms() const;
  template <typename InputIt>
  bool setTransforms(InputIt transforms_begin, InputIt transforms_end);
  template <typename InputIt>
  bool setTransforms(InputIt transforms_begin, InputIt transforms_end,
                     double shift_range, double rotate_deg_range);
  void setMainMapSize(double width, double height) {
    main_map_width_ = width;
    main_map_height_ = height;
  }

private:
  std::vector<nav_msgs::OccupancyGrid::ConstPtr> grids_;
  std::vector<cv::Mat> images_;
  std::vector<cv::Mat> transforms_;
  double main_map_width_;
  double main_map_height_;
  int main_map_id_;
};

template <typename InputIt>
void MergingPipeline::feed(InputIt grids_begin, InputIt grids_end)
{
  static_assert(std::is_assignable<nav_msgs::OccupancyGrid::ConstPtr&,
                                   decltype(*grids_begin)>::value,
                "grids_begin must point to nav_msgs::OccupancyGrid::ConstPtr "
                "data");

  // we can't reserve anything, because we want to support just InputIt and
  // their guarantee validity for only single-pass algos
  images_.clear();
  grids_.clear();
  int main_finding_ind = 0;
  for (InputIt it = grids_begin; it != grids_end; ++it, main_finding_ind++) {
    if (*it && !(*it)->data.empty()) {
      if ((*it)->header.frame_id == "map") {
        main_map_id_ = main_finding_ind;
        printf("---------------------------main map id is %d\n", main_map_id_);
      }
      grids_.push_back(*it);
      /* convert to opencv images. it creates only a view for opencv and does
       * not copy or own actual data. */
      images_.emplace_back((*it)->info.height, (*it)->info.width, CV_8UC1,
                           const_cast<signed char*>((*it)->data.data()));
    } else {
      grids_.emplace_back();
      images_.emplace_back();
    }
  }
}

template <typename InputIt>
bool MergingPipeline::setTransforms(InputIt transforms_begin,
                                    InputIt transforms_end,
                                    double shift_range,
                                    double rotate_deg_range) {
  static_assert(std::is_assignable<geometry_msgs::Transform&,
                                   decltype(*transforms_begin)>::value,
                "transforms_begin must point to geometry_msgs::Transform "
                "data");

  decltype(transforms_) transforms_buf;
  int map_ind = 0;
  double global_half_width = main_map_width_ / 2;
  double global_half_height = main_map_height_ / 2;
  for (InputIt it = transforms_begin; it != transforms_end; ++it, map_ind++) {
    const geometry_msgs::Quaternion& q = it->rotation;
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) <
        std::numeric_limits<double>::epsilon()) {
      // represents invalid transform
      transforms_buf.emplace_back();
      continue;
    }
    double half_width = (double)grids_[map_ind]->info.width / 2;
    double half_height = (double)grids_[map_ind]->info.height / 2;
    double resolution = (double)grids_[map_ind]->info.resolution;
    double yaw = 2 * acos(q.w);
    cv::Mat RR = cv::getRotationMatrix2D(cv::Point2f(half_width, half_height), yaw * 180 / 3.1415926, 1.0);

    double tx = it->translation.x;
    double ty = it->translation.y;
    cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);

    double minVal = -999.0, maxVal = -999.0;
    cv::Point minLoc(shift_range, shift_range), maxLoc(shift_range, shift_range);
    double rotating_degree = 0.0;
    if (map_ind != main_map_id_ && half_width < global_half_width && half_height < global_half_height) {
      cv::Rect roi((tx / resolution + global_half_width - half_width - shift_range),
                   (ty / resolution + global_half_height - half_height - shift_range),
                   (half_width + shift_range) * 2,
                   (half_height + shift_range) * 2);

      cv::rectangle(images_[main_map_id_], roi, cv::Scalar(127));
      cv::MatIterator_<char> it, end;
      cv::Mat cropped_main_image = images_[main_map_id_](roi).clone();
      for( it = cropped_main_image.begin<char>(), end = cropped_main_image.end<char>(); it != end; ++it) {
        if (*it < 60) {
          *it = 0;
        } else {
          *it = 255;
        }
      }
      cv::Mat image_template = images_[map_ind].clone();
      for( it = image_template.begin<char>(), end = image_template.end<char>(); it != end; ++it) {
        if (*it < 60) {
          *it = 0;
        } else {
          *it = 255;
        }
      }
      cv::Mat image_matched;
      cv::Mat image_template_affined;
      for (float deg = -rotate_deg_range; deg < rotate_deg_range; deg = deg + 0.1) {
        double _minVal = 0.0, _maxVal = 0.0;
        cv::Point _minLoc(shift_range, shift_range), _maxLoc(shift_range, shift_range);
        cv::Mat _RR = cv::getRotationMatrix2D(cv::Point2f(half_width, half_height), -(yaw * 180 / 3.1415926 + deg), 1.0);
        cv::warpAffine(image_template, image_template_affined, _RR, image_template.size());
        cv::matchTemplate(cropped_main_image, image_template_affined, image_matched, cv::TM_CCORR_NORMED);
        cv::minMaxLoc(image_matched, &_minVal, &_maxVal, &_minLoc, &_maxLoc);
        if (_maxVal > maxVal) {
          maxVal = _maxVal;
          maxLoc = _maxLoc;
          rotating_degree = deg;
        }
      }
      RR = cv::getRotationMatrix2D(cv::Point2f(half_width, half_height), yaw * 180 / 3.1415926 + rotating_degree, 1.0);
      printf("maxVal ++ %f -- maxLoc ++ %f ++ %f -- deg ++ %f\n", maxVal, double(maxLoc.x), double(maxLoc.y), rotating_degree);
      printf("shift pixel ++ %f ++ %f\n", (double(maxLoc.x) - shift_range), (double(maxLoc.y) - shift_range));
    }
    transform.at<double>(0, 2) = -(tx / resolution - half_width + global_half_width + (double(maxLoc.x) - shift_range));
    transform.at<double>(1, 2) = -(ty / resolution - half_height + global_half_height + (double(maxLoc.y) - shift_range));
    cv::Mat RT = RR * transform;
    transforms_buf.emplace_back(std::move(RT));
  }
  if (transforms_buf.size() != images_.size()) {
    return false;
  }
  std::swap(transforms_, transforms_buf);

  return true;
}

template <typename InputIt>
bool MergingPipeline::setTransforms(InputIt transforms_begin,
                                    InputIt transforms_end)
{
  static_assert(std::is_assignable<geometry_msgs::Transform&,
                                   decltype(*transforms_begin)>::value,
                "transforms_begin must point to geometry_msgs::Transform "
                "data");

  decltype(transforms_) transforms_buf;
  for (InputIt it = transforms_begin; it != transforms_end; ++it) {
    const geometry_msgs::Quaternion& q = it->rotation;
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) <
        std::numeric_limits<double>::epsilon()) {
      // represents invalid transform
      transforms_buf.emplace_back();
      continue;
    }
    double s = 2.0 / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    double a = 1 - q.y * q.y * s - q.z * q.z * s;
    double b = q.x * q.y * s + q.z * q.w * s;
    double tx = it->translation.x;
    double ty = it->translation.y;
    cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);
    transform.at<double>(0, 0) = transform.at<double>(1, 1) = a;
    transform.at<double>(1, 0) = b;
    transform.at<double>(0, 1) = -b;
    transform.at<double>(0, 2) = tx;
    transform.at<double>(1, 2) = ty;

    transforms_buf.emplace_back(std::move(transform));
  }

  if (transforms_buf.size() != images_.size()) {
    return false;
  }
  std::swap(transforms_, transforms_buf);

  return true;
}

}  // namespace combine_grids

#endif  // MERGING_PIPELINE_H_

