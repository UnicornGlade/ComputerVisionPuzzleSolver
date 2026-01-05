#include <cmath>
#include <filesystem>
#include <iostream>
#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image_io.h>
#include <limits>

namespace libimages::debug_io {

void ensure_dir_exists_for_file(const std::string &filepath) {
    namespace fs = std::filesystem;
    fs::path p(filepath);
    fs::path dir = p.parent_path();
    if (dir.empty()) {
        return;
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    rassert(!ec, "Failed to create directories", dir.string(), ec.message());
}

static void compute_roi(const image32f &img, bool ignore_border, int &j0, int &j1, int &i0, int &i1) {
    j0 = 0;
    i0 = 0;
    j1 = img.height();
    i1 = img.width();

    if (!ignore_border)
        return;

    if (img.width() >= 3 && img.height() >= 3) {
        j0 = 1;
        i0 = 1;
        j1 = img.height() - 1;
        i1 = img.width() - 1;
    }
}

image8u visualize_by_max(const image32f &img, bool ignore_border) {
    rassert(img.channels() == 1, "visualize_by_max expects 1-channel float image", img.channels());

    int j0, j1, i0, i1;
    compute_roi(img, ignore_border, j0, j1, i0, i1);

    float maxv = 0.0f;
    for (int j = j0; j < j1; ++j) {
        for (int i = i0; i < i1; ++i) {
            const float v = img(j, i);
            if (!std::isfinite(v))
                continue;
            if (v > maxv)
                maxv = v;
        }
    }

    image8u out(img.width(), img.height(), 1);
    out.fill(0);

    if (!(maxv > 0.0f))
        return out;

    const float inv = 255.0f / maxv;
    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            float v = img(j, i);
            if (!std::isfinite(v))
                v = 0.0f;
            float x = v * inv;
            if (x < 0.0f)
                x = 0.0f;
            if (x > 255.0f)
                x = 255.0f;
            out(j, i) = static_cast<std::uint8_t>(std::lround(x));
        }
    }
    return out;
}

image8u visualize_minmax(const image32f &img, bool ignore_border) {
    rassert(img.channels() == 1, "visualize_minmax expects 1-channel float image", img.channels());

    int j0, j1, i0, i1;
    compute_roi(img, ignore_border, j0, j1, i0, i1);

    float minv = std::numeric_limits<float>::infinity();
    float maxv = -std::numeric_limits<float>::infinity();

    for (int j = j0; j < j1; ++j) {
        for (int i = i0; i < i1; ++i) {
            const float v = img(j, i);
            if (!std::isfinite(v))
                continue;
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
    }

    image8u out(img.width(), img.height(), 1);
    out.fill(0);

    if (!std::isfinite(minv) || !std::isfinite(maxv) || !(maxv > minv))
        return out;

    const float inv = 255.0f / (maxv - minv);
    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            float v = img(j, i);
            if (!std::isfinite(v))
                v = minv;
            float x = (v - minv) * inv;
            if (x < 0.0f)
                x = 0.0f;
            if (x > 255.0f)
                x = 255.0f;
            out(j, i) = static_cast<std::uint8_t>(std::lround(x));
        }
    }
    return out;
}

void dump_image(const std::string &path, const image8u &img, bool verbose, bool force, int jpg_quality) {
#if DEBUG_IO_IMAGE_SAVE_ENABLED
    (void)force;
    if (verbose) {
        std::cerr << "[debug_io] saving " << path << " (" << img.width() << "x" << img.height() << "x" << img.channels()
                  << ")\n";
    }
    ensure_dir_exists_for_file(path);
    save_image(img, path, jpg_quality);
#else
    if (force || DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED) {
        if (verbose) {
            std::cerr << "[debug_io] forced saving " << path << " (" << img.width() << "x" << img.height() << "x"
                      << img.channels() << ")\n";
        }
        ensure_dir_exists_for_file(path);
        save_image(img, path, jpg_quality);
    }
#endif
}

void dump_image(const std::string &path, const image32f &img, bool verbose, bool force, int jpg_quality) {
    // Heuristic: if image has negative values -> min/max mapping; otherwise by max.
    float minv = std::numeric_limits<float>::infinity();
    float maxv = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            const float v = img(j, i);
            if (!std::isfinite(v))
                continue;
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
    }

    image8u vis = (std::isfinite(minv) && minv < 0.0f) ? visualize_minmax(img, true) : visualize_by_max(img, true);
    dump_image(path, vis, verbose, force, jpg_quality);
}

} // namespace libimages::debug_io
