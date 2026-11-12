#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

namespace tatar_ocr {

struct PipelineSpec {
    // Model & alphabet
    std::vector<std::string> alphabet;

    // Canvas & ArUco
    int canvas_w = 2100;
    int canvas_h = 2970;
    std::array<cv::Point2f, 4> dst_corners = {
        cv::Point2f(60.0f, 60.0f),
        cv::Point2f(2040.0f, 60.0f),
        cv::Point2f(2040.0f, 2910.0f),
        cv::Point2f(60.0f, 2910.0f),
    };
    std::string aruco_dict = "DICT_4X4_50";
    std::vector<int> corner_ids = {0, 1, 2, 3};
    std::map<int, int> corner_marker_corner_idx = {{0, 0}, {1, 1}, {3, 2}, {2, 3}};
    int question_marker_base_id = 10;
    std::map<int, cv::Point2f> canonical_anchors = {
        {11, {130.0f, 494.0f}}, {12, {130.0f, 770.0f}},  {13, {128.0f, 1045.0f}}, {14, {125.0f, 1320.0f}},
        {15, {125.0f, 1595.0f}}, {16, {125.0f, 1872.0f}}, {17, {125.0f, 2150.0f}}, {18, {125.0f, 2428.0f}},
    };
    int rectify_pad = 60;
    int extract_pad = 40;
    int min_markers_fast_pass = 8;
    int win_min = 3;
    int win_max = 45;
    int win_step = 4;
    int thresh_primary = 7;
    int thresh_shadow = 12;
    double clahe_clip = 2.5;
    cv::Size clahe_grid = {8, 8};
    int min_anchors_ransac = 4;
    double ransac_reproj_thresh = 5.0;
    int min_quadrant_coverage = 3;
    cv::Size contour_blur_ksize = {5, 5};
    double contour_canny_low = 50.0;
    double contour_canny_high = 180.0;
    int contour_max_candidates = 5;
    double contour_poly_eps = 0.02;
    double contour_min_area_ratio = 0.4;

    // Grid snapping
    int adapt_block = 31;
    int adapt_c = 8;
    cv::Size bridge_h = {9, 1};
    cv::Size bridge_v = {1, 9};
    cv::Size long_h = {180, 1};
    cv::Size long_v = {1, 50};
    int peak_radius = 2;
    int cluster_max_gap = 10;

    std::array<int, 4> name_h_bounds = {200, 360, 450, 1800};
    std::array<int, 2> name_v_x_range = {420, 1850};
    int name_v_inset_y = 10;
    double name_peak_rel_thresh = 0.15;
    int name_default_top_y = 214;
    int name_default_bot_y = 306;
    std::array<int, 2> name_row_h_range = {75, 110};
    std::array<int, 2> name_pitch_range = {75, 98};
    int name_chain_tol = 12;
    int name_default_start_x = 465;
    double name_default_pitch = 88.0;
    int name_cell_count = 15;

    int min_questions = 4;
    int max_questions = 8;
    double q_fallback_base_y = 495.0;
    double q_fallback_step_y = 280.0;
    int q_h_search_up = 35;
    int q_h_search_down = 140;
    std::array<int, 2> q_h_x_range = {220, 1020};
    double q_peak_rel_thresh = 0.20;
    int q_default_top_offset = -5;
    int q_default_row_h = 100;
    std::array<int, 2> q_row_h_range = {85, 115};
    std::array<int, 2> q_v_x_range = {160, 1500};
    int q_v_min_x = 185;
    int q_v_inset_y = 10;
    std::array<int, 2> q_pitch_range = {85, 115};
    int q_chain_tol = 15;
    std::array<int, 2> q_cell_count_range = {8, 12};
    int q_default_start_x = 224;
    double q_default_pitch = 100.0;
    int q_default_cell_count = 8;

    // Cell segmentation, emptiness & normalization
    double default_margin_trim_pct = 4.0;
    int min_cell_dim = 10;
    int min_inner_dim = 8;
    int min_margin_px = 5;

    double contrast_high_thresh = 60.0;
    double ink_delta_high = 22.0;
    double contrast_mid_thresh = 35.0;
    double ink_delta_mid = 16.0;
    double ink_delta_low_min = 7.0;
    double ink_delta_low_ratio = 0.32;

    int hollow_box_edge_pad = 3;
    double hollow_box_max_fill = 0.12;
    double edge_strip_span_frac = 0.35;
    int edge_strip_thick = 8;
    int edge_strip_pad = 4;
    double edge_long_span_frac = 0.60;
    int edge_long_thick = 12;
    int edge_long_pad = 2;
    int edge_speck_pad = 3;
    int edge_speck_thick = 5;
    int edge_speck_max_area = 40;

    int min_main_comp_area = 20;
    int min_sub_comp_area = 8;
    int diacritic_bottom_overlap = 5;
    int diacritic_max_gap_y = 22;
    int diacritic_pad_x = 6;
    double diacritic_max_w_frac = 0.60;
    int sub_comp_border_pad = 2;
    double sub_comp_border_max_frac = 0.35;
    int sub_comp_max_dist = 10;
    double sub_comp_dist_max_frac = 0.25;

    double core_inset_frac = 0.15;
    double top_band_frac = 0.40;
    int blackhat_min_k = 9;
    double blackhat_k_frac = 0.25;
    int empty_min_ink_pixels = 35;
    double empty_min_core_frac = 0.005;
    double empty_bh_p98_thresh = 10.0;
    double empty_contrast_thresh = 25.0;
    double empty_intrusion_w_frac = 0.92;
    double empty_intrusion_low_core = 0.010;
    double empty_intrusion_top_ratio = 0.45;

    int glyph_crop_pad = 2;
    int canonical_canvas_size = 64;
    double canonical_target_dim = 48.0;
    int canonical_max_dim = 60;
    double norm_pct_low = 2.0;
    double norm_pct_high = 98.0;
    double norm_scale = 240.0;
    double norm_offset = 10.0;
    int canonical_bg_val = 250;

    std::vector<std::string> diacritic_guard_chars = {"Ё", "Й"};
    int diacritic_min_area = 10;
    int diacritic_guard_overlap_y = 2;

    static PipelineSpec load(const std::string& spec_path);
};

struct CellBox {
    std::string section;
    int q_num = -1;
    int cell_idx = 0;
    std::array<int, 4> bbox = {0, 0, 0, 0};
};

struct QuestionRow {
    int q_num = 0;
    int marker_id = 0;
    std::vector<CellBox> cells;
};

struct TopItem {
    std::string ch;
    std::string unicode;
    double confidence = 0.0;
};

struct CellResult {
    bool is_empty = true;
    std::string ch = " ";
    double confidence = 100.0;
    std::vector<std::pair<std::string, double>> top3;
    std::vector<TopItem> top_items;

    bool has_debug = false;
    bool has_top_diacritic = false;
    cv::Mat resized_glyph;
    cv::Mat inp_64;
    cv::Mat ink_mask;
    std::map<std::string, double> ink_stats;
    bool has_ink_stats = false;
};

struct ScannedCell {
    int cell_idx = 0;
    std::string ch = " ";
    double confidence = 100.0;
    bool is_empty = true;
    std::vector<std::pair<std::string, double>> top3;
    std::array<int, 4> bbox = {0, 0, 0, 0};
};

struct QuestionResult {
    int q_num = 0;
    std::string text;
    std::vector<ScannedCell> cells;
};

struct BlankStats {
    int total_letters = 0;
    double avg_confidence = 0.0;
    int questions_count = 0;
};

struct BlankResult {
    std::string status = "success";
    std::string rectification_method;
    std::string student_name;
    std::vector<ScannedCell> student_name_cells;
    std::vector<QuestionResult> questions;
    BlankStats stats;
    cv::Mat rectified_bgr;
};

class NativeBlankScanner {
public:
    explicit NativeBlankScanner(const std::string& model_path, const std::string& spec_path = "");

    std::map<int, cv::Mat> detect_markers_padded(const cv::Mat& img_bgr, int pad = -1);
    std::tuple<cv::Mat, bool, std::string> rectify_sheet(const cv::Mat& img_bgr);
    std::pair<std::vector<CellBox>, std::vector<QuestionRow>> extract_cells(const cv::Mat& rectified_bgr);
    CellResult classify_cell(const cv::Mat& cell_bgr,
                             double margin_trim_pct = -1.0,
                             int top_k = 3,
                             bool force_predict = false,
                             bool debug = false);
    BlankResult process_blank(const cv::Mat& img_bgr);

    const PipelineSpec& spec() const { return spec_; }

private:
    PipelineSpec spec_;
    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::aruco::Dictionary aruco_dict_;
    std::array<cv::aruco::ArucoDetector, 2> detectors_;

    std::vector<std::vector<float>> run_batch(const std::vector<float>& batch_data, int batch_size);
};

}  // namespace tatar_ocr
