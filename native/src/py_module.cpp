#include "tatar_ocr.hpp"

#include <cstring>
#include <optional>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

cv::Mat numpy_to_mat(const py::array& arr) {
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> buf(arr);
    py::buffer_info info = buf.request();
    if (info.ndim == 2) {
        return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC1, info.ptr).clone();
    }
    if (info.ndim == 3) {
        int h = static_cast<int>(info.shape[0]);
        int w = static_cast<int>(info.shape[1]);
        int ch = static_cast<int>(info.shape[2]);
        if (ch == 3) return cv::Mat(h, w, CV_8UC3, info.ptr).clone();
        if (ch == 4) {
            cv::Mat bgra(h, w, CV_8UC4, info.ptr);
            cv::Mat bgr;
            cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
            return bgr;
        }
        if (ch == 1) return cv::Mat(h, w, CV_8UC1, info.ptr).clone();
    }
    throw std::invalid_argument("Expected uint8 image with shape (H, W) or (H, W, 3/4)");
}

py::array mat_to_numpy(const cv::Mat& mat) {
    cv::Mat c = mat.isContinuous() ? mat : mat.clone();
    if (c.empty()) return py::array_t<uint8_t>({0, 0});
    if (c.depth() == CV_32F) {
        std::vector<py::ssize_t> shape = (c.channels() == 1)
                                             ? std::vector<py::ssize_t>{c.rows, c.cols}
                                             : std::vector<py::ssize_t>{c.rows, c.cols, c.channels()};
        py::array_t<float> out(shape);
        std::memcpy(out.mutable_data(), c.data, c.total() * c.elemSize());
        return out;
    }
    std::vector<py::ssize_t> shape = (c.channels() == 1)
                                         ? std::vector<py::ssize_t>{c.rows, c.cols}
                                         : std::vector<py::ssize_t>{c.rows, c.cols, c.channels()};
    py::array_t<uint8_t> out(shape);
    std::memcpy(out.mutable_data(), c.data, c.total() * c.elemSize());
    return out;
}

py::tuple bbox_to_tuple(const std::array<int, 4>& b) {
    return py::make_tuple(b[0], b[1], b[2], b[3]);
}

py::list top3_to_list(const std::vector<std::pair<std::string, double>>& top3) {
    py::list lst;
    for (const auto& [ch, conf] : top3) lst.append(py::make_tuple(ch, conf));
    return lst;
}

py::dict scanned_cell_to_dict(const tatar_ocr::ScannedCell& sc) {
    py::dict d;
    d["cell_idx"] = sc.cell_idx;
    d["char"] = sc.ch;
    d["confidence"] = sc.confidence;
    d["is_empty"] = sc.is_empty;
    d["top3"] = top3_to_list(sc.top3);
    d["bbox"] = bbox_to_tuple(sc.bbox);
    return d;
}

py::dict cell_box_to_dict(const tatar_ocr::CellBox& cb) {
    py::dict d;
    d["section"] = cb.section;
    if (cb.q_num >= 0) d["q_num"] = cb.q_num;
    d["cell_idx"] = cb.cell_idx;
    d["bbox"] = bbox_to_tuple(cb.bbox);
    return d;
}

}  // namespace

PYBIND11_MODULE(tatar_ocr_native, m) {
    m.doc() = "Native C++20 OCR core for Tatar blank sheet scanning";

    py::class_<tatar_ocr::NativeBlankScanner>(m, "NativeBlankScanner")
        .def(py::init<const std::string&, const std::string&>(), py::arg("model_path"), py::arg("spec_path") = "")
        .def(
            "detect_markers_padded",
            [](tatar_ocr::NativeBlankScanner& self, const py::array& img_bgr, int pad) {
                cv::Mat mat = numpy_to_mat(img_bgr);
                std::map<int, cv::Mat> markers;
                {
                    py::gil_scoped_release release;
                    markers = self.detect_markers_padded(mat, pad);
                }
                py::dict out;
                for (const auto& [mid, pts] : markers) out[py::int_(mid)] = mat_to_numpy(pts);
                return out;
            },
            py::arg("img_bgr"), py::arg("pad") = 60)
        .def(
            "rectify_sheet",
            [](tatar_ocr::NativeBlankScanner& self, const py::array& img_bgr) {
                cv::Mat mat = numpy_to_mat(img_bgr);
                std::tuple<cv::Mat, bool, std::string> res;
                {
                    py::gil_scoped_release release;
                    res = self.rectify_sheet(mat);
                }
                return py::make_tuple(mat_to_numpy(std::get<0>(res)), std::get<1>(res), std::get<2>(res));
            },
            py::arg("img_bgr"))
        .def(
            "extract_cells",
            [](tatar_ocr::NativeBlankScanner& self, const py::array& rectified_bgr) {
                cv::Mat mat = numpy_to_mat(rectified_bgr);
                std::pair<std::vector<tatar_ocr::CellBox>, std::vector<tatar_ocr::QuestionRow>> res;
                {
                    py::gil_scoped_release release;
                    res = self.extract_cells(mat);
                }
                py::list name_list;
                for (const auto& cb : res.first) name_list.append(cell_box_to_dict(cb));
                py::list q_list;
                for (const auto& qr : res.second) {
                    py::dict qd;
                    qd["q_num"] = qr.q_num;
                    qd["marker_id"] = qr.marker_id;
                    py::list cl;
                    for (const auto& cb : qr.cells) cl.append(cell_box_to_dict(cb));
                    qd["cells"] = cl;
                    q_list.append(qd);
                }
                return py::make_tuple(name_list, q_list);
            },
            py::arg("rectified_bgr"))
        .def(
            "classify_cell",
            [](tatar_ocr::NativeBlankScanner& self, const py::array& cell_bgr, std::optional<double> margin_trim_pct,
               int top_k, bool force_predict, bool debug) {
                cv::Mat mat = numpy_to_mat(cell_bgr);
                double trim = margin_trim_pct.value_or(self.spec().default_margin_trim_pct);
                tatar_ocr::CellResult res;
                {
                    py::gil_scoped_release release;
                    res = self.classify_cell(mat, trim, top_k, force_predict, debug);
                }
                py::dict out;
                out["is_empty"] = res.is_empty;
                out["char"] = res.ch;
                out["confidence"] = res.confidence;
                out["top3"] = top3_to_list(res.top3);
                py::list items;
                for (const auto& it : res.top_items) {
                    py::dict id;
                    id["char"] = it.ch;
                    id["unicode"] = it.unicode;
                    id["confidence"] = it.confidence;
                    items.append(id);
                }
                out["top_items"] = items;
                if (debug) {
                    out["has_top_diacritic"] = res.has_top_diacritic;
                    if (!res.resized_glyph.empty()) out["resized_glyph"] = mat_to_numpy(res.resized_glyph);
                    if (!res.inp_64.empty()) out["inp_64"] = mat_to_numpy(res.inp_64);
                    if (res.has_ink_stats) out["ink_stats"] = res.ink_stats;
                    if (!res.ink_mask.empty()) out["ink_mask"] = mat_to_numpy(res.ink_mask);
                }
                return out;
            },
            py::arg("cell_bgr"), py::arg("margin_trim_pct") = py::none(), py::arg("top_k") = 3,
            py::arg("force_predict") = false, py::arg("debug") = false)
        .def(
            "process_blank",
            [](tatar_ocr::NativeBlankScanner& self, const py::array& img_bgr) {
                cv::Mat mat = numpy_to_mat(img_bgr);
                tatar_ocr::BlankResult res;
                {
                    py::gil_scoped_release release;
                    res = self.process_blank(mat);
                }
                py::dict out;
                out["status"] = res.status;
                out["rectification_method"] = res.rectification_method;
                out["student_name"] = res.student_name;

                py::list name_cells;
                for (const auto& sc : res.student_name_cells) name_cells.append(scanned_cell_to_dict(sc));
                out["student_name_cells"] = name_cells;

                py::list questions;
                for (const auto& q : res.questions) {
                    py::dict qd;
                    qd["q_num"] = q.q_num;
                    qd["text"] = q.text;
                    py::list qcells;
                    for (const auto& sc : q.cells) qcells.append(scanned_cell_to_dict(sc));
                    qd["cells"] = qcells;
                    questions.append(qd);
                }
                out["questions"] = questions;

                py::dict stats;
                stats["total_letters"] = res.stats.total_letters;
                stats["avg_confidence"] = res.stats.avg_confidence;
                stats["questions_count"] = res.stats.questions_count;
                out["stats"] = stats;

                out["rectified_bgr"] = mat_to_numpy(res.rectified_bgr);
                out["annotated_bgr"] = py::none();
                return out;
            },
            py::arg("img_bgr"));
}
