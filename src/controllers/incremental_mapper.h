// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_CONTROLLERS_INCREMENTAL_MAPPER_H_
#define COLMAP_SRC_CONTROLLERS_INCREMENTAL_MAPPER_H_

#include "base/reconstruction_manager.h"
#include "sfm/incremental_mapper.h"
#include "util/threading.h"

namespace colmap {

struct IncrementalMapperOptions {
    /// @brief 增量式制图的一些选项
 public:
    /// 一个匹配对，必须要有>15个内点的feature才可以。
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

    ///@todo 水印图像对？
  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  /// @todo 是否要重建多个子模型？
  // Whether to reconstruct multiple sub-models.
  bool multiple_models = true;

  /// @todo sub model的个数？
  // The number of sub-models to reconstruct.
  int max_num_models = 50;

  /// @todo 子图Ａ与Ｂ共享了２５个图像，就会让其重建停止的逻辑。
  // The maximum number of overlapping images between sub-models. If the
  // current sub-models shares more than this number of images with another
  // model, then the reconstruction is stopped.
  int max_model_overlap = 20;

  /// 子图中至少要有１０幅图像。
  // The minimum number of registered images of a sub-model, otherwise the
  // sub-model is discarded.
  int min_model_size = 10;

  /// 初始化的图像id; 可以指定一个，或者两个都指定。
  // The image identifiers used to initialize the reconstruction. Note that
  // only one or both image identifiers can be specified. In the former case,
  // the second image is automatically determined.
  int init_image_id1 = -1;
  int init_image_id2 = -1;

  /// 尝试初始化的次数
  // The number of trials to initialize the reconstruction.
  int init_num_trials = 200;

  /// 是否从重建点云中提取颜色。
  // Whether to extract colors for reconstructed points.
  bool extract_colors = true;

  /// 使用的线程个数；
  // The number of threads to use during reconstruction.
  int num_threads = -1;

  /// 内参估计退化的判定阈值。
  // Thresholds for filtering images with degenerate intrinsics.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 1.0;

  /// 内参也并不是都进行优化，比如优化焦距、主线点，畸变(径向畸变，切向畸变)参数
  // Which intrinsic parameters to optimize during the reconstruction.
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = true;

  /// 使用多线程求解ｂａ的最小残差个数。
  // The minimum number of residuals per bundle adjustment problem to
  // enable multi-threading solving of the problems.
  int ba_min_num_residuals_for_multi_threading = 50000;

  /// 局部图像做BA的个数
  // The number of images to optimize in local bundle adjustment.
  int ba_local_num_images = 6;

  /// 局部mapper的BA最大迭代次数。
  // The maximum number of local bundle adjustment iterations.
  int ba_local_max_num_iterations = 25;

  /// 是否使用pba在全局优化中
  // Whether to use PBA in global bundle adjustment.
  bool ba_global_use_pba = false;

  /// 使用哪个GPU进行PBA
  // The GPU index for PBA bundle adjustment.
  int ba_global_pba_gpu_index = -1;

  /// 设置为全局图像
  // The growth rates after which to perform global bundle adjustment.
  double ba_global_images_ratio = 1.1;
  double ba_global_points_ratio = 1.1;
  int ba_global_images_freq = 500;
  int ba_global_points_freq = 250000;

  /// 全局ＢＡ的最大迭代次数
  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 50;

  /// BA停止条件
  // The thresholds for iterative bundle adjustment refinements.
  int ba_local_max_refinements = 2;
  double ba_local_max_refinement_change = 0.001;
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  /// 中间过程保存的路径
  // Path to a folder with reconstruction snapshots during incremental
  // reconstruction. Snapshots will be saved according to the specified
  // frequency of registered images.
  std::string snapshot_path = "";
  int snapshot_images_freq = 0;

  /// 指定哪些图像用来重建，如果没有指定，则所有的图像默认用来重建
  // Which images to reconstruct. If no images are specified, all images will
  // be reconstructed by default.
  std::unordered_set<std::string> image_names;

  /// @todo 重建结果作为输入？
  // If reconstruction is provided as input, fix the existing image poses.
  bool fix_existing_images = false;

  /// 返回IncrementalMapper::Options
  IncrementalMapper::Options Mapper() const;
  ///　返回IncrementalTriangulator::Options
  IncrementalTriangulator::Options Triangulation() const;
  /// 返回ＢＡ的参数
  BundleAdjustmentOptions LocalBundleAdjustment() const;
  BundleAdjustmentOptions GlobalBundleAdjustment() const;
  /// 返回PBA的参数
  ParallelBundleAdjuster::Options ParallelGlobalBundleAdjustment() const;

  bool Check() const;

 private:
  friend class OptionManager;
  friend class MapperGeneralOptionsWidget;
  friend class MapperTriangulationOptionsWidget;
  friend class MapperRegistrationOptionsWidget;
  friend class MapperInitializationOptionsWidget;
  friend class MapperBundleAdjustmentOptionsWidget;
  friend class MapperFilteringOptionsWidget;
  friend class ReconstructionOptionsWidget;
  IncrementalMapper::Options mapper;
  IncrementalTriangulator::Options triangulation;
};

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalMapperController : public Thread {
 public:
  enum {
    INITIAL_IMAGE_PAIR_REG_CALLBACK,
    NEXT_IMAGE_REG_CALLBACK,
    LAST_IMAGE_REG_CALLBACK,
  };

  IncrementalMapperController(const IncrementalMapperOptions* options,
                              const std::string& image_path,
                              const std::string& database_path,
                              ReconstructionManager* reconstruction_manager);

 private:
  void Run();
  bool LoadDatabase();
  void Reconstruct(const IncrementalMapper::Options& init_mapper_options);

  const IncrementalMapperOptions* options_;
  const std::string image_path_;
  const std::string database_path_;
  ReconstructionManager* reconstruction_manager_;
  DatabaseCache database_cache_;
};

// Globally filter points and images in mapper.
size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper);
size_t FilterImages(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper);

// Globally complete and merge tracks in mapper.
size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
                              IncrementalMapper* mapper);

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_INCREMENTAL_MAPPER_H_
