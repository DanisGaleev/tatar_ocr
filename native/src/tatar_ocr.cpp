#include "tatar_ocr.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>

namespace tatar_ocr {
namespace {

constexpr const char* kDefaultAlphabet = "АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ";

int py_round(double x) {
    double r = std::round(x);
    if (std::abs(r - x) == 0.5) {
        auto ir = static_cast<long long>(r);
        if (ir % 2 != 0) r += (x > 0.0 ? -1.0 : 1.0);
    }
    return static_cast<int>(r);
}

double round_dec(double x, int decimals) {
    double s = (decimals == 1) ? 10.0 : 100.0;
    return std::round(x * s) / s;
}

std::filesystem::path to_fs_path(const std::string& p) {
    return std::filesystem::path(std::u8string(p.begin(), p.end()));
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        auto c = static_cast<unsigned char>(s[i]);
        size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        len = std::min(len, s.size() - i);
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::string utf8_to_unicode_hex(const std::string& ch) {
    if (ch.empty()) return "U+0000";
    auto b0 = static_cast<unsigned char>(ch[0]);
    uint32_t cp = 0;
    if (b0 < 0x80) {
        cp = b0;
    } else if ((b0 & 0xE0) == 0xC0 && ch.size() >= 2) {
        cp = ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(ch[1]) & 0x3F);
    } else if ((b0 & 0xF0) == 0xE0 && ch.size() >= 3) {
        cp = ((b0 & 0x0F) << 12) | ((static_cast<unsigned char>(ch[1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(ch[2]) & 0x3F);
    } else if (ch.size() >= 4) {
        cp = ((b0 & 0x07) << 18) | ((static_cast<unsigned char>(ch[1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(ch[2]) & 0x3F) << 6) | (static_cast<unsigned char>(ch[3]) & 0x3F);
    }
    char buf[16];
    std::snprintf(buf, sizeof(buf), "U+%04X", cp);
    return std::string(buf);
}

std::string trim_spaces(const std::string& s) {
    size_t b = s.find_first_not_of(' ');
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(' ');
    return s.substr(b, e - b + 1);
}

cv::aruco::PredefinedDictionaryType parse_aruco_dict(const std::string& name) {
    if (name == "DICT_4X4_50") return cv::aruco::DICT_4X4_50;
    if (name == "DICT_4X4_100") return cv::aruco::DICT_4X4_100;
    if (name == "DICT_5X5_50") return cv::aruco::DICT_5X5_50;
    if (name == "DICT_6X6_50") return cv::aruco::DICT_6X6_50;
    return cv::aruco::DICT_4X4_50;
}

cv::Mat ensure_bgr(const cv::Mat& img) {
    if (img.empty()) throw std::invalid_argument("Input image is empty");
    if (img.type() == CV_8UC3) return img;
    cv::Mat out;
    if (img.channels() == 1) {
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
    } else {
        img.convertTo(out, CV_8UC3);
    }
    return out;
}

std::pair<std::array<int, 256>, int> build_hist(const cv::Mat& m) {
    std::array<int, 256> hist{};
    for (int r = 0; r < m.rows; ++r) {
        const auto* row = m.ptr<uint8_t>(r);
        for (int c = 0; c < m.cols; ++c) ++hist[row[c]];
    }
    return {hist, m.rows * m.cols};
}

double hist_percentile(const std::array<int, 256>& hist, int count, double q) {
    if (count <= 0) return 0.0;
    double pos = (count - 1) * (q / 100.0);
    int lo = static_cast<int>(std::floor(pos));
    int hi = static_cast<int>(std::ceil(pos));
    auto at = [&](int idx) -> int {
        int cum = 0;
        for (int v = 0; v < 256; ++v) {
            cum += hist[v];
            if (cum > idx) return v;
        }
        return 255;
    };
    int vlo = at(lo);
    int vhi = (hi == lo) ? vlo : at(hi);
    return vlo + (pos - lo) * (vhi - vlo);
}

std::array<int, 4> clamp_bbox(int x, int y, int w, int h, int img_w, int img_h) {
    int x1 = std::max(0, std::min(img_w, x));
    int y1 = std::max(0, std::min(img_h, y));
    int x2 = std::max(x1, std::min(img_w, x + w));
    int y2 = std::max(y1, std::min(img_h, y + h));
    return {x1, y1, x2 - x1, y2 - y1};
}

struct Peak {
    int pos;
    double score;
};

std::vector<Peak> find_1d_peaks(const cv::Mat& prof_1d, int offset, double rel_thresh, int radius) {
    int len = static_cast<int>(prof_1d.total());
    if (len <= 2 * radius) return {};
    const auto* data = prof_1d.ptr<double>();
    double max_val = *std::max_element(data, data + len);
    if (max_val <= 0.0) return {};

    double thresh = rel_thresh * max_val;
    std::vector<Peak> peaks;
    int limit = len - radius;
    for (int d = radius; d < limit;) {
        double val = data[d];
        double win_max = *std::max_element(data + d - radius, data + d + radius + 1);
        if (val > thresh && val == win_max) {
            int end = d;
            while (end + 1 < limit && data[end + 1] == val) ++end;
            peaks.push_back({offset + (d + end) / 2, val});
            d = end + 1;
        } else {
            ++d;
        }
    }
    return peaks;
}

std::vector<int> cluster_peaks(const std::vector<int>& peaks, int max_gap) {
    if (peaks.empty()) return {};
    std::vector<int> out;
    long long sum = peaks[0];
    int cnt = 1;
    int prev = peaks[0];
    for (size_t i = 1; i < peaks.size(); ++i) {
        if (peaks[i] - prev <= max_gap) {
            sum += peaks[i];
            ++cnt;
        } else {
            out.push_back(py_round(static_cast<double>(sum) / cnt));
            sum = peaks[i];
            cnt = 1;
        }
        prev = peaks[i];
    }
    out.push_back(py_round(static_cast<double>(sum) / cnt));
    return out;
}

std::vector<int> extend_chain_forward(const std::vector<int>& peaks, std::vector<int> chain, int start_idx, int tol) {
    int last_idx = start_idx;
    int n = static_cast<int>(peaks.size());
    double pitch = (chain.back() - chain.front()) / static_cast<double>(chain.size() - 1);
    double expected = chain.back() + pitch;
    while (true) {
        int best_k = -1;
        double best_err = tol + 1.0;
        for (int k = last_idx + 1; k < n; ++k) {
            double err = std::abs(peaks[k] - expected);
            if (err <= tol && err < best_err) {
                best_err = err;
                best_k = k;
            } else if (peaks[k] > expected + tol) {
                break;
            }
        }
        if (best_k < 0) break;
        chain.push_back(peaks[best_k]);
        last_idx = best_k;
        pitch = (chain.back() - chain.front()) / static_cast<double>(chain.size() - 1);
        expected = chain.back() + pitch;
    }
    return chain;
}

std::vector<int> match_grid_chain(const std::vector<int>& peaks, int pitch_min, int pitch_max, int tol) {
    std::vector<int> best_chain;
    int n = static_cast<int>(peaks.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int cand = peaks[j] - peaks[i];
            if (cand < pitch_min || cand > pitch_max) continue;
            double expected_third = peaks[j] + cand;
            bool seeded = false;
            for (int k0 = j + 1; k0 < n; ++k0) {
                if (peaks[k0] > expected_third + tol) break;
                if (std::abs(peaks[k0] - expected_third) <= tol) {
                    seeded = true;
                    auto chain = extend_chain_forward(peaks, {peaks[i], peaks[j], peaks[k0]}, k0, tol);
                    if (chain.size() > best_chain.size()) best_chain = std::move(chain);
                }
            }
            if (!seeded && best_chain.size() < 2) best_chain = {peaks[i], peaks[j]};
        }
    }
    return best_chain;
}

std::pair<int, int> find_row_vertical_bounds(const cv::Mat& h_lines, int y_min, int y_max, int x_min, int x_max,
                                             double rel_thresh, int h_min, int h_max, int default_top, int default_h,
                                             int radius, bool sort_by_strength) {
    y_min = std::max(0, y_min);
    y_max = std::min(h_lines.rows, y_max);
    x_min = std::max(0, x_min);
    x_max = std::min(h_lines.cols, x_max);
    if (y_max <= y_min || x_max <= x_min) return {default_top, default_h};

    cv::Mat roi = h_lines(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
    cv::Mat prof;
    cv::reduce(roi, prof, 1, cv::REDUCE_SUM, CV_64F);
    auto peaks = find_1d_peaks(prof, y_min, rel_thresh, radius);
    if (sort_by_strength) {
        std::stable_sort(peaks.begin(), peaks.end(),
                         [](const Peak& a, const Peak& b) { return a.score > b.score; });
    }
    for (size_t i = 0; i < peaks.size(); ++i) {
        for (size_t j = i + 1; j < peaks.size(); ++j) {
            int dist = std::abs(peaks[i].pos - peaks[j].pos);
            if (h_min <= dist && dist <= h_max) return {std::min(peaks[i].pos, peaks[j].pos), dist};
        }
    }
    return {default_top, default_h};
}

std::vector<int> snap_row_grid(const cv::Mat& v_lines, int row_top, int row_h, int x_min, int x_max, int inset_y,
                               double rel_thresh, int pitch_min, int pitch_max, int chain_tol, int peak_radius,
                               int cluster_max_gap, int min_x = -1) {
    int y1 = std::max(0, row_top + inset_y);
    int y2 = std::min(v_lines.rows, row_top + row_h - inset_y);
    x_min = std::max(0, x_min);
    x_max = std::min(v_lines.cols, x_max);
    if (y2 <= y1 || x_max <= x_min) return {};

    cv::Mat roi = v_lines(cv::Range(y1, y2), cv::Range(x_min, x_max));
    cv::Mat prof;
    cv::reduce(roi, prof, 0, cv::REDUCE_SUM, CV_64F);
    auto raw = find_1d_peaks(prof, x_min, rel_thresh, peak_radius);
    std::vector<int> positions;
    positions.reserve(raw.size());
    for (const auto& p : raw) positions.push_back(p.pos);

    auto clustered = cluster_peaks(positions, cluster_max_gap);
    if (min_x >= 0) {
        clustered.erase(std::remove_if(clustered.begin(), clustered.end(), [min_x](int p) { return p < min_x; }),
                        clustered.end());
    }
    return match_grid_chain(clustered, pitch_min, pitch_max, chain_tol);
}

bool is_border_component(int bx, int by, int bw, int bh, int area, int iw, int ih, const PipelineSpec& s) {
    if (bx <= s.hollow_box_edge_pad && by <= s.hollow_box_edge_pad &&
        bx + bw >= iw - s.hollow_box_edge_pad && by + bh >= ih - s.hollow_box_edge_pad &&
        area < s.hollow_box_max_fill * (bw * bh)) {
        return true;
    }
    if (bw > s.edge_strip_span_frac * iw && bh <= s.edge_strip_thick &&
        (by <= s.edge_strip_pad || by + bh >= ih - s.edge_strip_pad)) {
        return true;
    }
    if (bh > s.edge_strip_span_frac * ih && bw <= s.edge_strip_thick &&
        (bx <= s.edge_strip_pad || bx + bw >= iw - s.edge_strip_pad)) {
        return true;
    }
    if (bh > s.edge_long_span_frac * ih && bw <= s.edge_long_thick &&
        (bx <= s.edge_long_pad || bx + bw >= iw - s.edge_long_pad)) {
        return true;
    }
    if (bw > s.edge_long_span_frac * iw && bh <= s.edge_long_thick &&
        (by <= s.edge_long_pad || by + bh >= ih - s.edge_long_pad)) {
        return true;
    }
    bool near_edge = (bx <= s.edge_speck_pad || by <= s.edge_speck_pad ||
                      bx + bw >= iw - s.edge_speck_pad || by + bh >= ih - s.edge_speck_pad);
    return near_edge && (bw <= s.edge_speck_thick || bh <= s.edge_speck_thick) && area < s.edge_speck_max_area;
}

struct PreparedCell {
    bool is_empty = true;
    bool has_top_diacritic = false;
    bool has_tensor = false;
    std::vector<float> tensor;
    cv::Mat resized_glyph, inp_64, valid_ink;
    std::map<std::string, double> ink_stats;
    bool has_ink_stats = false;
};

PreparedCell prepare_cell(const cv::Mat& cell_bgr, double margin_trim_pct, bool force_predict, bool debug,
                          const PipelineSpec& s) {
    PreparedCell out;
    int ch = cell_bgr.rows;
    int cw = cell_bgr.cols;
    if (ch < s.min_cell_dim || cw < s.min_cell_dim) return out;

    cv::Mat gray;
    if (cell_bgr.channels() == 3) {
        cv::cvtColor(cell_bgr, gray, cv::COLOR_BGR2GRAY);
    } else if (cell_bgr.channels() == 4) {
        cv::cvtColor(cell_bgr, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = cell_bgr;
    }

    int my = (margin_trim_pct > 0.0) ? std::max(s.min_margin_px, py_round(ch * (margin_trim_pct / 100.0))) : 0;
    int mx = (margin_trim_pct > 0.0) ? std::max(s.min_margin_px, py_round(cw * (margin_trim_pct / 100.0))) : 0;
    cv::Mat inner = (my > 0 && mx > 0 && (ch - 2 * my) >= s.min_inner_dim && (cw - 2 * mx) >= s.min_inner_dim)
                        ? gray(cv::Range(my, ch - my), cv::Range(mx, cw - mx))
                        : gray;

    int ih = inner.rows;
    int iw = inner.cols;
    if (ih < s.min_inner_dim || iw < s.min_inner_dim) return out;

    auto [inner_hist, inner_cnt] = build_hist(inner);
    double bg_val = hist_percentile(inner_hist, inner_cnt, 50.0);
    double p5_val = hist_percentile(inner_hist, inner_cnt, 5.0);
    double contrast_range = bg_val - p5_val;

    double ink_delta = (contrast_range > s.contrast_high_thresh)
                           ? s.ink_delta_high
                           : (contrast_range > s.contrast_mid_thresh
                                  ? s.ink_delta_mid
                                  : std::max(s.ink_delta_low_min, s.ink_delta_low_ratio * contrast_range));

    double thresh = bg_val - ink_delta;
    cv::Mat ink_mask(ih, iw, CV_8UC1);
    for (int r = 0; r < ih; ++r) {
        const auto* gp = inner.ptr<uint8_t>(r);
        auto* mp = ink_mask.ptr<uint8_t>(r);
        for (int c = 0; c < iw; ++c) mp[c] = (static_cast<double>(gp[c]) < thresh) ? 1 : 0;
    }

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(ink_mask, labels, stats, centroids, 8, CV_32S);
    cv::Mat valid_ink = cv::Mat::zeros(ih, iw, CV_8UC1);
    int total_ink_pixels = 0;

    if (num_labels <= 1) {
        if (!force_predict) return out;
    } else {
        std::vector<int> candidates;
        for (int i = 1; i < num_labels; ++i) {
            if (!is_border_component(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                                     stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT),
                                     stats.at<int>(i, cv::CC_STAT_AREA), iw, ih, s)) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            if (!force_predict) return out;
            for (int i = 1; i < num_labels; ++i) candidates.push_back(i);
        }

        int best_idx = candidates[0];
        int max_comp_area = stats.at<int>(best_idx, cv::CC_STAT_AREA);
        for (size_t idx = 1; idx < candidates.size(); ++idx) {
            int area = stats.at<int>(candidates[idx], cv::CC_STAT_AREA);
            if (area > max_comp_area) {
                max_comp_area = area;
                best_idx = candidates[idx];
            }
        }
        if (max_comp_area < s.min_main_comp_area && !force_predict) return out;

        int mbx = stats.at<int>(best_idx, cv::CC_STAT_LEFT);
        int mby = stats.at<int>(best_idx, cv::CC_STAT_TOP);
        int mbw = stats.at<int>(best_idx, cv::CC_STAT_WIDTH);
        int mbh = stats.at<int>(best_idx, cv::CC_STAT_HEIGHT);

        std::vector<uint8_t> keep(static_cast<size_t>(num_labels), 0);
        keep[best_idx] = 1;
        total_ink_pixels = max_comp_area;

        for (int i : candidates) {
            if (i == best_idx) continue;
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
            int by = stats.at<int>(i, cv::CC_STAT_TOP);
            int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            bool is_top_dia = (by + bh <= mby + s.diacritic_bottom_overlap &&
                               mby - (by + bh) <= s.diacritic_max_gap_y &&
                               bx + bw >= mbx - s.diacritic_pad_x && bx <= mbx + mbw + s.diacritic_pad_x &&
                               bw < s.diacritic_max_w_frac * iw);
            bool near_border = (bx <= s.sub_comp_border_pad || by <= s.sub_comp_border_pad ||
                                bx + bw >= iw - s.sub_comp_border_pad || by + bh >= ih - s.sub_comp_border_pad);
            if (near_border && !is_top_dia && area < s.sub_comp_border_max_frac * max_comp_area) continue;
            if (!is_top_dia) {
                int dist_x = std::max({0, bx - (mbx + mbw), mbx - (bx + bw)});
                int dist_y = std::max({0, by - (mby + mbh), mby - (by + bh)});
                if ((dist_x > s.sub_comp_max_dist || dist_y > s.sub_comp_max_dist) &&
                    area < s.sub_comp_dist_max_frac * max_comp_area) {
                    continue;
                }
                if (area < s.min_sub_comp_area) continue;
            }
            keep[i] = 1;
            total_ink_pixels += area;
            if (by + bh <= mby + s.diacritic_guard_overlap_y && area >= s.diacritic_min_area) {
                out.has_top_diacritic = true;
            }
        }

        for (int r = 0; r < ih; ++r) {
            const int* lp = labels.ptr<int>(r);
            auto* vp = valid_ink.ptr<uint8_t>(r);
            for (int c = 0; c < iw; ++c) vp[c] = keep[lp[c]];
        }
    }

    // Emptiness check
    int cy = py_round(s.core_inset_frac * ih);
    int cx = py_round(s.core_inset_frac * iw);
    int y_top_band = py_round(s.top_band_frac * ih);
    int core_sum = 0, low_core_sum = 0, top_sum = 0;
    int vx_min = iw, vx_max = -1, vy_min = ih, vy_max = -1;
    std::array<int, 256> ink_gray_hist{};
    int ink_gray_cnt = 0;

    for (int r = 0; r < ih; ++r) {
        const auto* vp = valid_ink.ptr<uint8_t>(r);
        const auto* gp = inner.ptr<uint8_t>(r);
        bool in_cx_y = (r >= cy && r < ih - cy);
        bool in_low_y = (r >= y_top_band && r < ih - cy);
        for (int c = 0; c < iw; ++c) {
            if (!vp[c]) continue;
            vx_min = std::min(vx_min, c);
            vx_max = std::max(vx_max, c);
            vy_min = std::min(vy_min, r);
            vy_max = std::max(vy_max, r);
            ++ink_gray_hist[gp[c]];
            ++ink_gray_cnt;
            if (r < y_top_band) ++top_sum;
            if (c >= cx && c < iw - cx) {
                if (in_cx_y) ++core_sum;
                if (in_low_y) ++low_core_sum;
            }
        }
    }

    double area_total = static_cast<double>(ih * iw);
    double core_frac = core_sum / area_total;
    double low_core_frac = low_core_sum / area_total;
    double ink_w = (vx_max >= vx_min) ? (vx_max - vx_min + 1) / static_cast<double>(iw) : 0.0;
    double ink_contrast = (ink_gray_cnt > 0) ? (bg_val - hist_percentile(ink_gray_hist, ink_gray_cnt, 50.0)) : 0.0;
    double top_ratio = top_sum / static_cast<double>(std::max(1, total_ink_pixels));

    auto compute_blackhat = [&]() {
        int bh_k = std::max(s.blackhat_min_k, py_round(s.blackhat_k_frac * std::min(ih, iw)) | 1);
        cv::Mat bh;
        cv::morphologyEx(inner, bh, cv::MORPH_BLACKHAT,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(bh_k, bh_k)));
        return bh;
    };

    if (!debug) {
        if (total_ink_pixels < s.empty_min_ink_pixels || core_frac < s.empty_min_core_frac ||
            (ink_w >= s.empty_intrusion_w_frac && low_core_frac < s.empty_intrusion_low_core &&
             top_ratio > s.empty_intrusion_top_ratio)) {
            out.is_empty = true;
        } else if (ink_contrast < s.empty_contrast_thresh) {
            cv::Mat bh = compute_blackhat();
            cv::Mat bh_core = bh(cv::Range(cy, std::max(cy, ih - cy)), cv::Range(cx, std::max(cx, iw - cx)));
            auto [bh_hist, bh_cnt] = build_hist(bh_core);
            out.is_empty = (hist_percentile(bh_hist, bh_cnt, 98.0) < s.empty_bh_p98_thresh);
        } else {
            out.is_empty = false;
        }
        if (out.is_empty && !force_predict) return out;
    } else {
        cv::Mat bh = compute_blackhat();
        cv::Mat bh_core = bh(cv::Range(cy, std::max(cy, ih - cy)), cv::Range(cx, std::max(cx, iw - cx)));
        auto [bh_hist, bh_cnt] = build_hist(bh_core);

        std::array<int, 256> core_bh_hist{}, stroke_bh_hist{};
        int core_bh_cnt = 0, stroke_bh_cnt = 0;
        for (int r = 0; r < ih; ++r) {
            const auto* vp = valid_ink.ptr<uint8_t>(r);
            const auto* bp = bh.ptr<uint8_t>(r);
            bool in_cy = (r >= cy && r < ih - cy);
            for (int c = 0; c < iw; ++c) {
                if (!vp[c]) continue;
                ++stroke_bh_hist[bp[c]];
                ++stroke_bh_cnt;
                if (in_cy && c >= cx && c < iw - cx) {
                    ++core_bh_hist[bp[c]];
                    ++core_bh_cnt;
                }
            }
        }
        double core_bh_p98 = hist_percentile(bh_hist, bh_cnt, 98.0);
        out.has_ink_stats = true;
        out.ink_stats = {
            {"ink_frac", total_ink_pixels / area_total},
            {"ink_h", (vy_max >= vy_min) ? (vy_max - vy_min + 1) / static_cast<double>(ih) : 0.0},
            {"ink_w", ink_w},
            {"ink_contrast", ink_contrast},
            {"core_frac", core_frac},
            {"low_core_frac", low_core_frac},
            {"top_ratio", top_ratio},
            {"core_contrast", core_bh_cnt > 0 ? hist_percentile(core_bh_hist, core_bh_cnt, 50.0) : 0.0},
            {"stroke_contrast", stroke_bh_cnt > 0 ? hist_percentile(stroke_bh_hist, stroke_bh_cnt, 50.0) : 0.0},
            {"core_bh_p90", hist_percentile(bh_hist, bh_cnt, 90.0)},
            {"core_bh_p95", hist_percentile(bh_hist, bh_cnt, 95.0)},
            {"core_bh_p98", core_bh_p98},
        };
        out.valid_ink = valid_ink;
        out.is_empty = (total_ink_pixels < s.empty_min_ink_pixels || core_frac < s.empty_min_core_frac ||
                        (core_bh_p98 < s.empty_bh_p98_thresh && ink_contrast < s.empty_contrast_thresh) ||
                        (ink_w >= s.empty_intrusion_w_frac && low_core_frac < s.empty_intrusion_low_core &&
                         top_ratio > s.empty_intrusion_top_ratio));
        if (out.is_empty && !force_predict) return out;
    }

    // Normalize glyph to 64x64 float tensor
    cv::Mat glyph_crop;
    if (vx_max >= vx_min && vy_max >= vy_min) {
        int x1 = std::max(0, vx_min - s.glyph_crop_pad);
        int y1 = std::max(0, vy_min - s.glyph_crop_pad);
        int x2 = std::min(iw - 1, vx_max + s.glyph_crop_pad);
        int y2 = std::min(ih - 1, vy_max + s.glyph_crop_pad);
        glyph_crop = inner(cv::Range(y1, y2 + 1), cv::Range(x1, x2 + 1));
    } else {
        glyph_crop = inner;
    }

    int gh = glyph_crop.rows;
    int gw = glyph_crop.cols;
    double scale = s.canonical_target_dim / (static_cast<double>(std::max(gh, gw)) + 1e-5);
    int new_w = std::max(1, std::min(s.canonical_max_dim, py_round(gw * scale)));
    int new_h = std::max(1, std::min(s.canonical_max_dim, py_round(gh * scale)));
    int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LANCZOS4;

    cv::Mat resized;
    cv::resize(glyph_crop, resized, cv::Size(new_w, new_h), 0.0, 0.0, interp);

    auto [r_hist, r_cnt] = build_hist(resized);
    double p_lo = hist_percentile(r_hist, r_cnt, s.norm_pct_low);
    double p_hi = hist_percentile(r_hist, r_cnt, s.norm_pct_high);
    double denom = p_hi - p_lo + 1e-5;

    int sz = s.canonical_canvas_size;
    cv::Mat inp_64(sz, sz, CV_8UC1, cv::Scalar(s.canonical_bg_val));
    int off_x = (sz - new_w) / 2;
    int off_y = (sz - new_h) / 2;
    for (int r = 0; r < new_h; ++r) {
        const auto* src = resized.ptr<uint8_t>(r);
        auto* dst = inp_64.ptr<uint8_t>(off_y + r) + off_x;
        for (int c = 0; c < new_w; ++c) {
            double v = ((static_cast<float>(src[c]) - p_lo) / denom) * s.norm_scale + s.norm_offset;
            dst[c] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
        }
    }

    out.has_tensor = true;
    out.tensor.resize(static_cast<size_t>(sz * sz));
    for (int r = 0; r < sz; ++r) {
        const auto* gp = inp_64.ptr<uint8_t>(r);
        float* tp = out.tensor.data() + r * sz;
        for (int c = 0; c < sz; ++c) tp[c] = ((static_cast<float>(gp[c]) / 255.0f) - 0.5f) / 0.5f;
    }

    if (debug) {
        out.resized_glyph = resized;
        out.inp_64 = inp_64;
    }
    return out;
}

CellResult build_prediction(const std::vector<float>& probs, bool has_top_diacritic, bool is_empty_cell, int top_k,
                            const PipelineSpec& s) {
    CellResult res;
    res.is_empty = is_empty_cell;
    int num_classes = static_cast<int>(std::min(probs.size(), s.alphabet.size()));
    if (num_classes <= 0) return res;

    int k = std::min(std::max(3, top_k), num_classes);
    std::vector<int> order(static_cast<size_t>(num_classes));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<std::string> top_chars;
    std::vector<double> top_confs;
    for (int i = 0; i < k; ++i) {
        top_chars.push_back(s.alphabet[order[i]]);
        top_confs.push_back(static_cast<double>(probs[order[i]]) * 100.0);
    }

    bool is_guard_char = !top_chars.empty() &&
                         std::find(s.diacritic_guard_chars.begin(), s.diacritic_guard_chars.end(), top_chars[0]) !=
                             s.diacritic_guard_chars.end();
    if (is_guard_char && !is_empty_cell && !has_top_diacritic) {
        std::string target_alt;
        if (top_chars[0] == "Ё") {
            target_alt = "Е";
        } else {
            bool has_o = std::find(top_chars.begin(), top_chars.end(), "О") != top_chars.end();
            target_alt = has_o ? "О" : "И";
        }
        auto it = std::find(top_chars.begin(), top_chars.end(), target_alt);
        if (it != top_chars.end()) {
            size_t idx = static_cast<size_t>(std::distance(top_chars.begin(), it));
            std::string ch = std::move(top_chars[idx]);
            double cf = top_confs[idx];
            top_chars.erase(top_chars.begin() + idx);
            top_confs.erase(top_confs.begin() + idx);
            top_chars.insert(top_chars.begin(), std::move(ch));
            top_confs.insert(top_confs.begin(), cf);
        } else {
            auto ait = std::find(s.alphabet.begin(), s.alphabet.end(), target_alt);
            if (ait != s.alphabet.end()) {
                size_t cidx = static_cast<size_t>(std::distance(s.alphabet.begin(), ait));
                if (cidx < probs.size()) {
                    top_chars.insert(top_chars.begin(), target_alt);
                    top_confs.insert(top_confs.begin(), static_cast<double>(probs[cidx]) * 100.0);
                    top_chars.resize(static_cast<size_t>(k));
                    top_confs.resize(static_cast<size_t>(k));
                }
            }
        }
    }

    int items_n = std::min(top_k, static_cast<int>(top_chars.size()));
    for (int i = 0; i < items_n; ++i) {
        res.top_items.push_back({top_chars[i], utf8_to_unicode_hex(top_chars[i]), round_dec(top_confs[i], 2)});
    }
    if (is_empty_cell) {
        res.ch = " ";
        res.confidence = 100.0;
        res.top3 = {{" ", 100.0}};
    } else {
        res.ch = top_chars[0];
        res.confidence = round_dec(top_confs[0], 1);
        int t3_n = std::min(3, static_cast<int>(top_chars.size()));
        for (int i = 0; i < t3_n; ++i) res.top3.emplace_back(top_chars[i], round_dec(top_confs[i], 1));
    }
    return res;
}

template <typename T>
void read_opt(const nlohmann::json& j, const char* key, T& dst) {
    if (j.contains(key) && !j[key].is_null()) dst = j[key].get<T>();
}

void read_size(const nlohmann::json& j, const char* key, cv::Size& dst) {
    if (j.contains(key) && j[key].is_array() && j[key].size() == 2) {
        dst = cv::Size(j[key][0].get<int>(), j[key][1].get<int>());
    }
}

}  // namespace

PipelineSpec PipelineSpec::load(const std::string& spec_path) {
    PipelineSpec s;
    s.alphabet = split_utf8(kDefaultAlphabet);
    if (spec_path.empty()) return s;

    std::ifstream ifs(to_fs_path(spec_path));
    if (!ifs.is_open()) throw std::runtime_error("Cannot open pipeline_spec.json: " + spec_path);
    nlohmann::json root = nlohmann::json::parse(ifs);

    if (root.contains("model") && root["model"].contains("alphabet")) {
        const auto& a = root["model"]["alphabet"];
        if (a.is_string()) {
            s.alphabet = split_utf8(a.get<std::string>());
        } else if (a.is_array()) {
            s.alphabet = a.get<std::vector<std::string>>();
        }
    }

    if (root.contains("canvas")) {
        const auto& c = root["canvas"];
        read_opt(c, "width", s.canvas_w);
        read_opt(c, "height", s.canvas_h);
        if (c.contains("dst_corners") && c["dst_corners"].size() == 4) {
            for (size_t i = 0; i < 4; ++i) {
                s.dst_corners[i] = cv::Point2f(c["dst_corners"][i][0].get<float>(), c["dst_corners"][i][1].get<float>());
            }
        }
    }

    if (root.contains("aruco")) {
        const auto& a = root["aruco"];
        read_opt(a, "dict", s.aruco_dict);
        read_opt(a, "corner_ids", s.corner_ids);
        if (a.contains("corner_marker_corner_idx")) {
            s.corner_marker_corner_idx.clear();
            for (const auto& [k, v] : a["corner_marker_corner_idx"].items()) {
                s.corner_marker_corner_idx[std::stoi(k)] = v.get<int>();
            }
        }
        read_opt(a, "question_marker_base_id", s.question_marker_base_id);
        if (a.contains("canonical_anchors")) {
            s.canonical_anchors.clear();
            for (const auto& [k, v] : a["canonical_anchors"].items()) {
                s.canonical_anchors[std::stoi(k)] = cv::Point2f(v[0].get<float>(), v[1].get<float>());
            }
        }
        read_opt(a, "rectify_pad", s.rectify_pad);
        read_opt(a, "extract_pad", s.extract_pad);
        read_opt(a, "min_markers_fast_pass", s.min_markers_fast_pass);
        read_opt(a, "win_min", s.win_min);
        read_opt(a, "win_max", s.win_max);
        read_opt(a, "win_step", s.win_step);
        read_opt(a, "thresh_primary", s.thresh_primary);
        read_opt(a, "thresh_shadow", s.thresh_shadow);
        read_opt(a, "clahe_clip", s.clahe_clip);
        read_size(a, "clahe_grid", s.clahe_grid);
        read_opt(a, "min_anchors_ransac", s.min_anchors_ransac);
        read_opt(a, "ransac_reproj_thresh", s.ransac_reproj_thresh);
        read_opt(a, "min_quadrant_coverage", s.min_quadrant_coverage);
        read_size(a, "contour_blur_ksize", s.contour_blur_ksize);
        if (a.contains("contour_canny") && a["contour_canny"].size() == 2) {
            s.contour_canny_low = a["contour_canny"][0].get<double>();
            s.contour_canny_high = a["contour_canny"][1].get<double>();
        }
        read_opt(a, "contour_max_candidates", s.contour_max_candidates);
        read_opt(a, "contour_poly_eps", s.contour_poly_eps);
        read_opt(a, "contour_min_area_ratio", s.contour_min_area_ratio);
    }

    if (root.contains("grid")) {
        const auto& g = root["grid"];
        read_opt(g, "adapt_block", s.adapt_block);
        read_opt(g, "adapt_c", s.adapt_c);
        read_size(g, "bridge_h", s.bridge_h);
        read_size(g, "bridge_v", s.bridge_v);
        read_size(g, "long_h", s.long_h);
        read_size(g, "long_v", s.long_v);
        read_opt(g, "peak_radius", s.peak_radius);
        read_opt(g, "cluster_max_gap", s.cluster_max_gap);
        if (g.contains("student_name")) {
            const auto& sn = g["student_name"];
            read_opt(sn, "h_bounds", s.name_h_bounds);
            read_opt(sn, "v_x_range", s.name_v_x_range);
            read_opt(sn, "v_inset_y", s.name_v_inset_y);
            read_opt(sn, "peak_rel_thresh", s.name_peak_rel_thresh);
            read_opt(sn, "default_top_y", s.name_default_top_y);
            read_opt(sn, "default_bot_y", s.name_default_bot_y);
            read_opt(sn, "row_h_range", s.name_row_h_range);
            read_opt(sn, "pitch_range", s.name_pitch_range);
            read_opt(sn, "chain_tol", s.name_chain_tol);
            read_opt(sn, "default_start_x", s.name_default_start_x);
            read_opt(sn, "default_pitch", s.name_default_pitch);
            read_opt(sn, "cell_count", s.name_cell_count);
        }
        if (g.contains("questions")) {
            const auto& q = g["questions"];
            read_opt(q, "min_questions", s.min_questions);
            read_opt(q, "max_questions", s.max_questions);
            read_opt(q, "fallback_base_y", s.q_fallback_base_y);
            read_opt(q, "fallback_step_y", s.q_fallback_step_y);
            read_opt(q, "h_search_up", s.q_h_search_up);
            read_opt(q, "h_search_down", s.q_h_search_down);
            read_opt(q, "h_x_range", s.q_h_x_range);
            read_opt(q, "peak_rel_thresh", s.q_peak_rel_thresh);
            read_opt(q, "default_top_offset", s.q_default_top_offset);
            read_opt(q, "default_row_h", s.q_default_row_h);
            read_opt(q, "row_h_range", s.q_row_h_range);
            read_opt(q, "v_x_range", s.q_v_x_range);
            read_opt(q, "v_min_x", s.q_v_min_x);
            read_opt(q, "v_inset_y", s.q_v_inset_y);
            read_opt(q, "pitch_range", s.q_pitch_range);
            read_opt(q, "chain_tol", s.q_chain_tol);
            read_opt(q, "cell_count_range", s.q_cell_count_range);
            read_opt(q, "default_start_x", s.q_default_start_x);
            read_opt(q, "default_pitch", s.q_default_pitch);
            read_opt(q, "default_cell_count", s.q_default_cell_count);
        }
    }

    if (root.contains("cell")) {
        const auto& cl = root["cell"];
        read_opt(cl, "default_margin_trim_pct", s.default_margin_trim_pct);
        read_opt(cl, "min_cell_dim", s.min_cell_dim);
        read_opt(cl, "min_inner_dim", s.min_inner_dim);
        read_opt(cl, "min_margin_px", s.min_margin_px);
        read_opt(cl, "contrast_high_thresh", s.contrast_high_thresh);
        read_opt(cl, "ink_delta_high", s.ink_delta_high);
        read_opt(cl, "contrast_mid_thresh", s.contrast_mid_thresh);
        read_opt(cl, "ink_delta_mid", s.ink_delta_mid);
        read_opt(cl, "ink_delta_low_min", s.ink_delta_low_min);
        read_opt(cl, "ink_delta_low_ratio", s.ink_delta_low_ratio);
        read_opt(cl, "hollow_box_edge_pad", s.hollow_box_edge_pad);
        read_opt(cl, "hollow_box_max_fill", s.hollow_box_max_fill);
        read_opt(cl, "edge_strip_span_frac", s.edge_strip_span_frac);
        read_opt(cl, "edge_strip_thick", s.edge_strip_thick);
        read_opt(cl, "edge_strip_pad", s.edge_strip_pad);
        read_opt(cl, "edge_long_span_frac", s.edge_long_span_frac);
        read_opt(cl, "edge_long_thick", s.edge_long_thick);
        read_opt(cl, "edge_long_pad", s.edge_long_pad);
        read_opt(cl, "edge_speck_pad", s.edge_speck_pad);
        read_opt(cl, "edge_speck_thick", s.edge_speck_thick);
        read_opt(cl, "edge_speck_max_area", s.edge_speck_max_area);
        read_opt(cl, "min_main_comp_area", s.min_main_comp_area);
        read_opt(cl, "min_sub_comp_area", s.min_sub_comp_area);
        read_opt(cl, "diacritic_bottom_overlap", s.diacritic_bottom_overlap);
        read_opt(cl, "diacritic_max_gap_y", s.diacritic_max_gap_y);
        read_opt(cl, "diacritic_pad_x", s.diacritic_pad_x);
        read_opt(cl, "diacritic_max_w_frac", s.diacritic_max_w_frac);
        read_opt(cl, "sub_comp_border_pad", s.sub_comp_border_pad);
        read_opt(cl, "sub_comp_border_max_frac", s.sub_comp_border_max_frac);
        read_opt(cl, "sub_comp_max_dist", s.sub_comp_max_dist);
        read_opt(cl, "sub_comp_dist_max_frac", s.sub_comp_dist_max_frac);
        read_opt(cl, "core_inset_frac", s.core_inset_frac);
        read_opt(cl, "top_band_frac", s.top_band_frac);
        read_opt(cl, "blackhat_min_k", s.blackhat_min_k);
        read_opt(cl, "blackhat_k_frac", s.blackhat_k_frac);
        read_opt(cl, "empty_min_ink_pixels", s.empty_min_ink_pixels);
        read_opt(cl, "empty_min_core_frac", s.empty_min_core_frac);
        read_opt(cl, "empty_bh_p98_thresh", s.empty_bh_p98_thresh);
        read_opt(cl, "empty_contrast_thresh", s.empty_contrast_thresh);
        read_opt(cl, "empty_intrusion_w_frac", s.empty_intrusion_w_frac);
        read_opt(cl, "empty_intrusion_low_core", s.empty_intrusion_low_core);
        read_opt(cl, "empty_intrusion_top_ratio", s.empty_intrusion_top_ratio);
        read_opt(cl, "glyph_crop_pad", s.glyph_crop_pad);
        read_opt(cl, "canonical_canvas_size", s.canonical_canvas_size);
        read_opt(cl, "canonical_target_dim", s.canonical_target_dim);
        read_opt(cl, "canonical_max_dim", s.canonical_max_dim);
        read_opt(cl, "norm_pct_low", s.norm_pct_low);
        read_opt(cl, "norm_pct_high", s.norm_pct_high);
        read_opt(cl, "norm_scale", s.norm_scale);
        read_opt(cl, "norm_offset", s.norm_offset);
        read_opt(cl, "canonical_bg_val", s.canonical_bg_val);
        read_opt(cl, "diacritic_guard_chars", s.diacritic_guard_chars);
        read_opt(cl, "diacritic_min_area", s.diacritic_min_area);
        read_opt(cl, "diacritic_guard_overlap_y", s.diacritic_guard_overlap_y);
    }
    return s;
}

NativeBlankScanner::NativeBlankScanner(const std::string& model_path, const std::string& spec_path)
    : spec_(PipelineSpec::load(spec_path)),
      env_(ORT_LOGGING_LEVEL_WARNING, "tatar_ocr"),
      aruco_dict_(cv::aruco::getPredefinedDictionary(parse_aruco_dict(spec_.aruco_dict))) {
    auto mpath = to_fs_path(model_path);
    if (!std::filesystem::exists(mpath)) {
        throw std::runtime_error("ONNX model not found: " + model_path);
    }
    session_ = std::make_unique<Ort::Session>(env_, mpath.c_str(), session_opts_);
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    clahe_ = cv::createCLAHE(spec_.clahe_clip, spec_.clahe_grid);
    auto make_det = [&](int thresh_const) {
        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        params.adaptiveThreshWinSizeMin = spec_.win_min;
        params.adaptiveThreshWinSizeMax = spec_.win_max;
        params.adaptiveThreshWinSizeStep = spec_.win_step;
        params.adaptiveThreshConstant = thresh_const;
        return cv::aruco::ArucoDetector(aruco_dict_, params);
    };
    detectors_[0] = make_det(spec_.thresh_primary);
    detectors_[1] = make_det(spec_.thresh_shadow);
}

std::map<int, cv::Mat> NativeBlankScanner::detect_markers_padded(const cv::Mat& img_bgr, int pad) {
    if (pad < 0) pad = spec_.rectify_pad;
    cv::Mat bgr = ensure_bgr(img_bgr);
    cv::Mat padded;
    cv::copyMakeBorder(bgr, padded, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    std::map<int, cv::Mat> marker_map;
    auto collect = [&](const cv::aruco::ArucoDetector& det, const cv::Mat& image) {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        det.detectMarkers(image, corners, ids);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (marker_map.count(ids[i])) continue;
            cv::Mat pts(4, 2, CV_32FC1);
            for (int k = 0; k < 4; ++k) {
                pts.at<float>(k, 0) = corners[i][k].x - static_cast<float>(pad);
                pts.at<float>(k, 1) = corners[i][k].y - static_cast<float>(pad);
            }
            marker_map.emplace(ids[i], std::move(pts));
        }
    };

    for (const auto& det : detectors_) {
        collect(det, padded);
        if (static_cast<int>(marker_map.size()) >= spec_.min_markers_fast_pass) return marker_map;
    }

    cv::Mat gray, enhanced;
    cv::cvtColor(padded, gray, cv::COLOR_BGR2GRAY);
    clahe_->apply(gray, enhanced);
    collect(detectors_[0], enhanced);
    return marker_map;
}

std::tuple<cv::Mat, bool, std::string> NativeBlankScanner::rectify_sheet(const cv::Mat& img_bgr) {
    cv::Mat bgr = ensure_bgr(img_bgr);
    int h_orig = bgr.rows;
    int w_orig = bgr.cols;
    cv::Size canvas_sz(spec_.canvas_w, spec_.canvas_h);
    auto marker_map = detect_markers_padded(bgr, spec_.rectify_pad);

    auto get_corner_pt = [&](int mid, int c_idx) {
        const cv::Mat& m = marker_map.at(mid);
        return cv::Point2f(m.at<float>(c_idx, 0), m.at<float>(c_idx, 1));
    };

    // 1. Exact 4-corner perspective transform
    bool all_corners = std::all_of(spec_.corner_ids.begin(), spec_.corner_ids.end(),
                                   [&](int id) { return marker_map.count(id) > 0; });
    if (all_corners) {
        const std::array<int, 4> order = {0, 1, 3, 2};
        std::array<cv::Point2f, 4> src_pts;
        for (size_t i = 0; i < 4; ++i) {
            int mid = order[i];
            int c_idx = spec_.corner_marker_corner_idx.at(mid);
            src_pts[i] = get_corner_pt(mid, c_idx);
        }
        cv::Mat M = cv::getPerspectiveTransform(src_pts.data(), spec_.dst_corners.data());
        if (!M.empty()) {
            cv::Mat warped;
            cv::warpPerspective(bgr, warped, M, canvas_sz, cv::INTER_LANCZOS4);
            return {warped, true, "ArUco 4-Point Geometry"};
        }
    }

    // 2. Multi-marker RANSAC similarity / homography
    std::vector<cv::Point2f> src_pts, dst_pts;
    auto dst_for_corner_id = [&](int mid) {
        for (size_t i = 0; i < 4; ++i) {
            static constexpr int kOrder[4] = {0, 1, 3, 2};
            if (kOrder[i] == mid) return spec_.dst_corners[i];
        }
        return spec_.dst_corners[0];
    };
    for (int mid : spec_.corner_ids) {
        if (marker_map.count(mid) && spec_.corner_marker_corner_idx.count(mid)) {
            src_pts.push_back(get_corner_pt(mid, spec_.corner_marker_corner_idx.at(mid)));
            dst_pts.push_back(dst_for_corner_id(mid));
        }
    }
    for (const auto& [mid, m] : marker_map) {
        auto it = spec_.canonical_anchors.find(mid);
        if (it == spec_.canonical_anchors.end()) continue;
        cv::Point2f mean_pt(0.0f, 0.0f);
        for (int k = 0; k < 4; ++k) {
            mean_pt.x += m.at<float>(k, 0);
            mean_pt.y += m.at<float>(k, 1);
        }
        src_pts.push_back(mean_pt * 0.25f);
        dst_pts.push_back(it->second);
    }

    if (static_cast<int>(src_pts.size()) >= spec_.min_anchors_ransac) {
        std::array<bool, 4> quads{};
        double mid_x = spec_.canvas_w / 2.0;
        double mid_y = spec_.canvas_h / 2.0;
        for (const auto& pt : dst_pts) {
            int q = (pt.x < mid_x ? 0 : 1) + (pt.y < mid_y ? 0 : 2);
            quads[q] = true;
        }
        int quad_cnt = static_cast<int>(std::count(quads.begin(), quads.end(), true));
        bool missing_top = !marker_map.count(0) && !marker_map.count(1);
        bool missing_bot = !marker_map.count(2) && !marker_map.count(3);
        bool unbalanced = (quad_cnt < spec_.min_quadrant_coverage) || missing_top || missing_bot;

        if (unbalanced) {
            cv::Mat M_aff = cv::estimateAffinePartial2D(src_pts, dst_pts, cv::noArray(), cv::RANSAC,
                                                        spec_.ransac_reproj_thresh);
            if (!M_aff.empty()) {
                cv::Mat warped;
                cv::warpAffine(bgr, warped, M_aff, canvas_sz, cv::INTER_LANCZOS4);
                return {warped, true,
                        "ArUco Similarity Transform (" + std::to_string(src_pts.size()) + " anchors, unbalanced)"};
            }
        } else {
            cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, spec_.ransac_reproj_thresh);
            if (!H.empty()) {
                cv::Mat warped;
                cv::warpPerspective(bgr, warped, H, canvas_sz, cv::INTER_LANCZOS4);
                return {warped, true,
                        "ArUco Multi-Marker Homography (" + std::to_string(src_pts.size()) + " anchors)"};
            }
        }
    }

    // 3. Document contour fallback
    cv::Mat gray, blurred, edged;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, spec_.contour_blur_ksize, 0);
    cv::Canny(blurred, edged, spec_.contour_canny_low, spec_.contour_canny_high);
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(edged, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(cnts.begin(), cnts.end(),
              [](const auto& a, const auto& b) { return cv::contourArea(a) > cv::contourArea(b); });
    size_t limit = std::min(cnts.size(), static_cast<size_t>(spec_.contour_max_candidates));

    std::array<cv::Point2f, 4> canvas_corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(spec_.canvas_w), 0.0f),
        cv::Point2f(static_cast<float>(spec_.canvas_w), static_cast<float>(spec_.canvas_h)),
        cv::Point2f(0.0f, static_cast<float>(spec_.canvas_h)),
    };

    for (size_t i = 0; i < limit; ++i) {
        double peri = cv::arcLength(cnts[i], true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(cnts[i], approx, spec_.contour_poly_eps * peri, true);
        if (approx.size() == 4 && cv::contourArea(cnts[i]) > h_orig * w_orig * spec_.contour_min_area_ratio) {
            std::array<cv::Point2f, 4> pts;
            for (int k = 0; k < 4; ++k) pts[k] = cv::Point2f(approx[k]);
            int tl = 0, tr = 0, br = 0, bl = 0;
            for (int k = 1; k < 4; ++k) {
                float sk = pts[k].x + pts[k].y;
                float dk = pts[k].y - pts[k].x;
                if (sk < pts[tl].x + pts[tl].y) tl = k;
                if (sk > pts[br].x + pts[br].y) br = k;
                if (dk < pts[tr].y - pts[tr].x) tr = k;
                if (dk > pts[bl].y - pts[bl].x) bl = k;
            }
            std::array<cv::Point2f, 4> rect_pts = {pts[tl], pts[tr], pts[br], pts[bl]};
            cv::Mat M = cv::getPerspectiveTransform(rect_pts.data(), canvas_corners.data());
            if (!M.empty()) {
                cv::Mat warped;
                cv::warpPerspective(bgr, warped, M, canvas_sz, cv::INTER_LANCZOS4);
                return {warped, true, "Document Contour Fallback"};
            }
        }
    }

    cv::Mat warped;
    cv::resize(bgr, warped, canvas_sz);
    return {warped, false, "Direct Scaling (No markers detected)"};
}

std::pair<std::vector<CellBox>, std::vector<QuestionRow>> NativeBlankScanner::extract_cells(
    const cv::Mat& rectified_bgr) {
    cv::Mat bgr = ensure_bgr(rectified_bgr);
    auto marker_map = detect_markers_padded(bgr, spec_.extract_pad);
    int img_h = bgr.rows;
    int img_w = bgr.cols;

    cv::Mat gray, enhanced, thresh;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    clahe_->apply(gray, enhanced);
    cv::adaptiveThreshold(enhanced, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                          spec_.adapt_block, spec_.adapt_c);

    cv::Mat h_closed, v_closed, h_lines, v_lines;
    cv::morphologyEx(thresh, h_closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, spec_.bridge_h));
    cv::morphologyEx(thresh, v_closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, spec_.bridge_v));
    cv::morphologyEx(h_closed, h_lines, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, spec_.long_h));
    cv::morphologyEx(v_closed, v_lines, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, spec_.long_v));

    // Student Name row
    auto [name_top_y, name_cell_h] = find_row_vertical_bounds(
        h_lines, spec_.name_h_bounds[0], spec_.name_h_bounds[1], spec_.name_h_bounds[2], spec_.name_h_bounds[3],
        spec_.name_peak_rel_thresh, spec_.name_row_h_range[0], spec_.name_row_h_range[1], spec_.name_default_top_y,
        spec_.name_default_bot_y - spec_.name_default_top_y, spec_.peak_radius, false);
    auto name_chain = snap_row_grid(v_lines, name_top_y, name_cell_h, spec_.name_v_x_range[0], spec_.name_v_x_range[1],
                                    spec_.name_v_inset_y, spec_.name_peak_rel_thresh, spec_.name_pitch_range[0],
                                    spec_.name_pitch_range[1], spec_.name_chain_tol, spec_.peak_radius,
                                    spec_.cluster_max_gap);

    double name_start_x = (name_chain.size() >= 2) ? name_chain.front() : spec_.name_default_start_x;
    double name_pitch = (name_chain.size() >= 2)
                            ? (name_chain.back() - name_chain.front()) / static_cast<double>(name_chain.size() - 1)
                            : spec_.name_default_pitch;
    int name_w = py_round(name_pitch);

    std::vector<CellBox> name_cells;
    name_cells.reserve(static_cast<size_t>(spec_.name_cell_count));
    for (int idx = 0; idx < spec_.name_cell_count; ++idx) {
        name_cells.push_back({"student_name", -1, idx + 1,
                              clamp_bbox(py_round(name_start_x + idx * name_pitch), name_top_y, name_w, name_cell_h,
                                         img_w, img_h)});
    }

    // Question rows
    int max_q_detected = 0;
    for (const auto& [mid, _] : marker_map) {
        if (mid > spec_.question_marker_base_id) max_q_detected = std::max(max_q_detected, mid - spec_.question_marker_base_id);
    }
    int max_q = std::max(spec_.min_questions,
                         std::min(spec_.max_questions, max_q_detected > 0 ? max_q_detected : spec_.min_questions));

    std::vector<QuestionRow> question_rows;
    question_rows.reserve(static_cast<size_t>(max_q));
    for (int q_num = 1; q_num <= max_q; ++q_num) {
        int mid = q_num + spec_.question_marker_base_id;
        auto anchor_it = spec_.canonical_anchors.find(mid);
        double default_my = (anchor_it != spec_.canonical_anchors.end())
                                ? static_cast<double>(anchor_it->second.y)
                                : spec_.q_fallback_base_y + (q_num - 1) * spec_.q_fallback_step_y;
        double est_my = default_my;
        if (marker_map.count(mid)) {
            const cv::Mat& m = marker_map.at(mid);
            est_my = (m.at<float>(0, 1) + m.at<float>(1, 1) + m.at<float>(2, 1) + m.at<float>(3, 1)) * 0.25;
        }

        auto [row_top, row_h] = find_row_vertical_bounds(
            h_lines, static_cast<int>(est_my - spec_.q_h_search_up), static_cast<int>(est_my + spec_.q_h_search_down),
            spec_.q_h_x_range[0], spec_.q_h_x_range[1], spec_.q_peak_rel_thresh, spec_.q_row_h_range[0],
            spec_.q_row_h_range[1], py_round(est_my + spec_.q_default_top_offset), spec_.q_default_row_h,
            spec_.peak_radius, true);
        auto best_chain = snap_row_grid(v_lines, row_top, row_h, spec_.q_v_x_range[0], spec_.q_v_x_range[1],
                                        spec_.q_v_inset_y, spec_.q_peak_rel_thresh, spec_.q_pitch_range[0],
                                        spec_.q_pitch_range[1], spec_.q_chain_tol, spec_.peak_radius,
                                        spec_.cluster_max_gap, spec_.q_v_min_x);

        double start_x = spec_.q_default_start_x;
        double pitch = spec_.q_default_pitch;
        int row_cell_count = spec_.q_default_cell_count;
        if (best_chain.size() >= 2) {
            start_x = best_chain.front();
            int detected_cnt = static_cast<int>(best_chain.size()) - 1;
            row_cell_count = std::max(spec_.q_cell_count_range[0], std::min(spec_.q_cell_count_range[1], detected_cnt));
            pitch = (static_cast<int>(best_chain.size()) > row_cell_count)
                        ? (best_chain[row_cell_count] - start_x) / static_cast<double>(row_cell_count)
                        : (best_chain.back() - best_chain.front()) / static_cast<double>(best_chain.size() - 1);
        }

        int cell_w = py_round(pitch);
        QuestionRow row;
        row.q_num = q_num;
        row.marker_id = mid;
        row.cells.reserve(static_cast<size_t>(row_cell_count));
        std::string sec = "question_" + std::to_string(q_num);
        for (int c = 0; c < row_cell_count; ++c) {
            row.cells.push_back(
                {sec, q_num, c + 1, clamp_bbox(py_round(start_x + c * pitch), row_top, cell_w, row_h, img_w, img_h)});
        }
        question_rows.push_back(std::move(row));
    }

    return {name_cells, question_rows};
}

std::vector<std::vector<float>> NativeBlankScanner::run_batch(const std::vector<float>& batch_data, int batch_size) {
    if (batch_size <= 0) return {};
    int sz = spec_.canonical_canvas_size;
    std::array<int64_t, 4> shape = {batch_size, 1, sz, sz};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(batch_data.data()),
                                                              batch_data.size(), shape.data(), shape.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_classes = (out_shape.size() >= 2) ? static_cast<int>(out_shape[1]) : static_cast<int>(spec_.alphabet.size());
    const float* logits = outputs[0].GetTensorData<float>();

    std::vector<std::vector<float>> probs(static_cast<size_t>(batch_size),
                                          std::vector<float>(static_cast<size_t>(num_classes)));
    for (int i = 0; i < batch_size; ++i) {
        const float* row = logits + i * num_classes;
        float max_l = *std::max_element(row, row + num_classes);
        double sum = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            float e = std::exp(row[c] - max_l);
            probs[i][c] = e;
            sum += e;
        }
        float inv = static_cast<float>(1.0 / sum);
        for (int c = 0; c < num_classes; ++c) probs[i][c] *= inv;
    }
    return probs;
}

CellResult NativeBlankScanner::classify_cell(const cv::Mat& cell_bgr, double margin_trim_pct, int top_k,
                                             bool force_predict, bool debug) {
    if (margin_trim_pct < 0.0) margin_trim_pct = spec_.default_margin_trim_pct;
    PreparedCell prep = prepare_cell(cell_bgr, margin_trim_pct, force_predict, debug, spec_);
    if (!prep.has_tensor) {
        CellResult empty;
        if (debug) {
            empty.has_debug = true;
            empty.ink_mask = prep.valid_ink;
            empty.ink_stats = std::move(prep.ink_stats);
            empty.has_ink_stats = prep.has_ink_stats;
        }
        return empty;
    }

    auto probs = run_batch(prep.tensor, 1);
    CellResult res = build_prediction(probs[0], prep.has_top_diacritic, prep.is_empty, top_k, spec_);
    if (debug) {
        res.has_debug = true;
        res.has_top_diacritic = prep.has_top_diacritic;
        res.resized_glyph = std::move(prep.resized_glyph);
        res.inp_64 = std::move(prep.inp_64);
        res.ink_mask = std::move(prep.valid_ink);
        res.ink_stats = std::move(prep.ink_stats);
        res.has_ink_stats = prep.has_ink_stats;
    }
    return res;
}

BlankResult NativeBlankScanner::process_blank(const cv::Mat& img_bgr) {
    auto [rectified, ok, method] = rectify_sheet(img_bgr);
    (void)ok;
    auto [name_cells, question_rows] = extract_cells(rectified);

    int img_h = rectified.rows;
    int img_w = rectified.cols;
    size_t cell_planes = static_cast<size_t>(spec_.canonical_canvas_size * spec_.canonical_canvas_size);

    std::vector<PreparedCell> prepared;
    std::vector<float> batch_buf;
    std::vector<int> batch_idx;

    auto enqueue_cells = [&](const std::vector<CellBox>& cells) {
        for (const auto& cell : cells) {
            auto [x, y, w, h] = clamp_bbox(cell.bbox[0], cell.bbox[1], cell.bbox[2], cell.bbox[3], img_w, img_h);
            cv::Mat crop = (w > 0 && h > 0) ? rectified(cv::Range(y, y + h), cv::Range(x, x + w)) : cv::Mat();
            PreparedCell p = crop.empty()
                                 ? PreparedCell{}
                                 : prepare_cell(crop, spec_.default_margin_trim_pct, false, false, spec_);
            if (p.has_tensor) {
                batch_idx.push_back(static_cast<int>(batch_buf.size() / cell_planes));
                batch_buf.insert(batch_buf.end(), p.tensor.begin(), p.tensor.end());
            } else {
                batch_idx.push_back(-1);
            }
            prepared.push_back(std::move(p));
        }
    };

    enqueue_cells(name_cells);
    for (const auto& q : question_rows) enqueue_cells(q.cells);

    int n_batch = static_cast<int>(batch_buf.size() / cell_planes);
    auto all_probs = run_batch(batch_buf, n_batch);

    size_t cursor = 0;
    auto collect_group = [&](const std::vector<CellBox>& cells) {
        std::string text;
        std::vector<ScannedCell> out_cells;
        out_cells.reserve(cells.size());
        int letters = 0;
        double conf_sum = 0.0;

        for (const auto& cell : cells) {
            int bi = batch_idx[cursor];
            const auto& p = prepared[cursor];
            ++cursor;

            CellResult cr;
            if (bi >= 0) cr = build_prediction(all_probs[bi], p.has_top_diacritic, p.is_empty, 3, spec_);
            if (cr.top3.empty()) cr.top3 = {{" ", 100.0}};

            out_cells.push_back({cell.cell_idx, cr.ch, cr.confidence, cr.is_empty, std::move(cr.top3), cell.bbox});
            if (!cr.is_empty) {
                text += cr.ch;
                ++letters;
                conf_sum += cr.confidence;
            }
        }
        return std::tuple{std::move(text), std::move(out_cells), letters, conf_sum};
    };

    BlankResult res;
    res.rectification_method = std::move(method);
    auto [name_raw, name_res, _nl, _nc] = collect_group(name_cells);
    (void)_nl;
    (void)_nc;
    res.student_name = trim_spaces(name_raw);
    res.student_name_cells = std::move(name_res);

    int total_letters = 0;
    double conf_sum = 0.0;
    for (const auto& q : question_rows) {
        auto [q_text, q_cells, q_letters, q_conf] = collect_group(q.cells);
        total_letters += q_letters;
        conf_sum += q_conf;
        res.questions.push_back({q.q_num, std::move(q_text), std::move(q_cells)});
    }

    double avg_conf = (total_letters > 0) ? (conf_sum / std::max(1, total_letters)) : 0.0;
    res.stats = {total_letters, round_dec(avg_conf, 1), static_cast<int>(res.questions.size())};
    res.rectified_bgr = std::move(rectified);
    return res;
}

}  // namespace tatar_ocr
