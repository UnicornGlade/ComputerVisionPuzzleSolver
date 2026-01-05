#pragma once

#include <string>

#include <libimages/image.h>

#ifndef DEBUG_IO_IMAGE_SAVE_ENABLED
#define DEBUG_IO_IMAGE_SAVE_ENABLED 1
#endif

#ifndef DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED
#define DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED 0
#endif

namespace libimages::debug_io {

// Creates parent directories for a filepath (if needed). No-op if already exists.
void ensure_dir_exists_for_file(const std::string &filepath);

// Maps float image to 8-bit grayscale using max value (typical for magnitudes).
// If ignore_border == true and image is at least 3x3, border pixels are excluded from stats.
image8u visualize_by_max(const image32f &img, bool ignore_border = true);

// Maps float image to 8-bit grayscale using min/max range.
image8u visualize_minmax(const image32f &img, bool ignore_border = true);

// Save helpers (extension-driven: png/jpg/jpeg)
void dump_image(const std::string &path, const image8u &img, bool verbose = false, bool force = false,
                int jpg_quality = 95);
void dump_image(const std::string &path, const image32f &img, bool verbose = false, bool force = false,
                int jpg_quality = 95);

} // namespace libimages::debug_io
