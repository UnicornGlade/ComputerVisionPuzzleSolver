#include <libimages/image_io.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <libbase/runtime_assert.h>

#define LIBIMAGES_USE_STB

#if defined(LIBIMAGES_USE_STB)
#include <stb_image.h>
#include <stb_image_write.h>
#elif defined(LIBIMAGES_USE_SYSTEM)
#include <jpeglib.h>
#include <png.h>
#else
#error "Define either LIBIMAGES_USE_STB or LIBIMAGES_USE_SYSTEM"
#endif

namespace libimages {

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

static std::string file_ext_lower(const std::string &path) {
    const auto pos = path.find_last_of('.');
    if (pos == std::string::npos)
        return std::string();
    return to_lower_copy(path.substr(pos + 1));
}

#if defined(LIBIMAGES_USE_STB)

image8u load_image(const std::string &path) {
    const std::filesystem::path p = std::filesystem::u8path(path);
    rassert(std::filesystem::exists(p),
            "Please check working directory and relative input file path - input file does not exist", path);
    rassert(std::filesystem::is_regular_file(p), "Input path is not a regular file", path);

    int w = 0, h = 0, comp = 0;
    if (!stbi_info(path.c_str(), &w, &h, &comp)) {
        rassert(false, "stbi_info failed", path, stbi_failure_reason());
    }

    // Rule: RGB unless the file contains alpha (RGBA). Also: Gray+Alpha -> RGBA.
    const int req_comp = (comp == 4 || comp == 2) ? 4 : 3;

    int loaded_comp = 0;
    unsigned char *ptr = stbi_load(path.c_str(), &w, &h, &loaded_comp, req_comp);
    if (!ptr) {
        rassert(false, "stbi_load failed", path, stbi_failure_reason());
    }

    image8u img(w, h, req_comp);
    const std::size_t n =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(req_comp);
    std::memcpy(img.data(), ptr, n * sizeof(std::uint8_t));
    stbi_image_free(ptr);
    return img;
}

void save_image(const image8u &img, const std::string &path, int jpg_quality) {
    rassert(img.width() > 0 && img.height() > 0, "Empty image");
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported channel count",
            img.channels());

    const std::string ext = file_ext_lower(path);
    rassert(!ext.empty(), "Output path must have an extension", path);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();

    if (ext == "png") {
        const int stride_bytes = w * c;
        const int ok = stbi_write_png(path.c_str(), w, h, c, img.data(), stride_bytes);
        rassert(ok != 0, "stbi_write_png failed", path);
        return;
    }

    if (ext == "jpg" || ext == "jpeg") {
        rassert(jpg_quality >= 1 && jpg_quality <= 100, "Invalid JPEG quality", jpg_quality);

        if (c == 4) {
            // Drop alpha for JPEG.
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    const std::size_t src =
                        (static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i)) * 4;
                    const std::size_t dst =
                        (static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i)) * 3;
                    rgb[dst + 0] = img.data()[src + 0];
                    rgb[dst + 1] = img.data()[src + 1];
                    rgb[dst + 2] = img.data()[src + 2];
                }
            }
            const int ok = stbi_write_jpg(path.c_str(), w, h, 3, rgb.data(), jpg_quality);
            rassert(ok != 0, "stbi_write_jpg failed", path);
            return;
        }

        const int ok = stbi_write_jpg(path.c_str(), w, h, c, img.data(), jpg_quality);
        rassert(ok != 0, "stbi_write_jpg failed", path);
        return;
    }

    rassert(false, "Unsupported output extension", ext, path);
}

#elif defined(LIBIMAGES_USE_SYSTEM)

static image8u load_png(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    rassert(fp != nullptr, "Failed to open file", path);

    unsigned char header[8] = {};
    const std::size_t nread = std::fread(header, 1, 8, fp);
    rassert(nread == 8, "Failed to read PNG signature", path);
    rassert(png_sig_cmp(header, 0, 8) == 0, "Not a PNG file", path);

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    rassert(png != nullptr, "png_create_read_struct failed", path);
    png_infop info = png_create_info_struct(png);
    rassert(info != nullptr, "png_create_info_struct failed", path);

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        rassert(false, "libpng error while reading", path);
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = 0, h = 0;
    int bit_depth = 0, color_type = 0;
    png_get_IHDR(png, info, &w, &h, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    const int out_color_type = png_get_color_type(png, info);
    const int channels = (out_color_type == PNG_COLOR_TYPE_RGB_ALPHA) ? 4 : 3;

    const png_size_t rowbytes = png_get_rowbytes(png, info);
    rassert(rowbytes == static_cast<png_size_t>(w) * static_cast<png_size_t>(channels), "Unexpected PNG rowbytes", path,
            static_cast<unsigned long>(rowbytes));

    image8u img(static_cast<int>(w), static_cast<int>(h), channels);
    std::vector<png_bytep> rows(static_cast<std::size_t>(h));
    for (png_uint_32 j = 0; j < h; ++j) {
        rows[static_cast<std::size_t>(j)] =
            reinterpret_cast<png_bytep>(img.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(rowbytes));
    }

    png_read_image(png, rows.data());
    png_read_end(png, nullptr);

    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return img;
}

static image8u load_jpeg(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    rassert(fp != nullptr, "Failed to open file", path);

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);

    const int rc = jpeg_read_header(&cinfo, TRUE);
    rassert(rc == 1, "jpeg_read_header failed", path);

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    const int w = static_cast<int>(cinfo.output_width);
    const int h = static_cast<int>(cinfo.output_height);
    const int channels = static_cast<int>(cinfo.output_components);
    rassert(channels == 3, "Unexpected JPEG components", channels);

    image8u img(w, h, 3);

    std::vector<JSAMPLE> row(static_cast<std::size_t>(w) * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rowptr = row.data();
        const JDIMENSION got = jpeg_read_scanlines(&cinfo, &rowptr, 1);
        rassert(got == 1, "jpeg_read_scanlines failed", path);

        const int j = static_cast<int>(cinfo.output_scanline) - 1;
        std::memcpy(img.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(w) * 3, row.data(), row.size());
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);
    return img;
}

image8u load_image(const std::string &path) {
    const std::filesystem::path p = std::filesystem::u8path(path);
    rassert(std::filesystem::exists(p), "Input file does not exist", path);
    rassert(std::filesystem::is_regular_file(p), "Input path is not a regular file", path);

    const std::string ext = file_ext_lower(path);
    rassert(!ext.empty(), "Input path must have an extension", path);

    if (ext == "png")
        return load_png(path);
    if (ext == "jpg" || ext == "jpeg")
        return load_jpeg(path);

    rassert(false, "Unsupported input extension", ext, path);
    return {};
}

static void save_png(const image8u &img, const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "wb");
    rassert(fp != nullptr, "Failed to open file for writing", path);

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    rassert(png != nullptr, "png_create_write_struct failed", path);
    png_infop info = png_create_info_struct(png);
    rassert(info != nullptr, "png_create_info_struct failed", path);

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        rassert(false, "libpng error while writing", path);
    }

    png_init_io(png, fp);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();

    int color_type = PNG_COLOR_TYPE_RGB;
    if (c == 1)
        color_type = PNG_COLOR_TYPE_GRAY;
    else if (c == 3)
        color_type = PNG_COLOR_TYPE_RGB;
    else if (c == 4)
        color_type = PNG_COLOR_TYPE_RGB_ALPHA;
    else
        rassert(false, "Unsupported PNG channel count", c);

    png_set_IHDR(png, info, static_cast<png_uint_32>(w), static_cast<png_uint_32>(h), 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    const std::size_t rowbytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(c);
    std::vector<png_bytep> rows(static_cast<std::size_t>(h));
    for (int j = 0; j < h; ++j) {
        rows[static_cast<std::size_t>(j)] = const_cast<png_bytep>(
            reinterpret_cast<png_const_bytep>(img.data() + static_cast<std::size_t>(j) * rowbytes));
    }

    png_write_image(png, rows.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
}

static void save_jpeg(const image8u &img, const std::string &path, int quality) {
    rassert(quality >= 1 && quality <= 100, "Invalid JPEG quality", quality);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(c == 1 || c == 3 || c == 4, "Unsupported JPEG channel count", c);

    // JPEG has no alpha
    std::vector<std::uint8_t> tmp;
    const std::uint8_t *src = img.data();
    int write_c = c;

    if (c == 4) {
        tmp.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                const std::size_t s =
                    (static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i)) * 4;
                const std::size_t d =
                    (static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i)) * 3;
                tmp[d + 0] = src[s + 0];
                tmp[d + 1] = src[s + 1];
                tmp[d + 2] = src[s + 2];
            }
        }
        src = tmp.data();
        write_c = 3;
    }

    FILE *fp = std::fopen(path.c_str(), "wb");
    rassert(fp != nullptr, "Failed to open file for writing", path);

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = write_c;
    cinfo.in_color_space = (write_c == 1) ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const std::size_t rowbytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(write_c);
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPLE *row = const_cast<JSAMPLE *>(
            reinterpret_cast<const JSAMPLE *>(src + static_cast<std::size_t>(cinfo.next_scanline) * rowbytes));
        JSAMPROW rowptr = row;
        const JDIMENSION written = jpeg_write_scanlines(&cinfo, &rowptr, 1);
        rassert(written == 1, "jpeg_write_scanlines failed", path);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
}

void save_image(const image8u &img, const std::string &path, int jpg_quality) {
    rassert(img.width() > 0 && img.height() > 0, "Empty image");
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported channel count",
            img.channels());

    const std::string ext = file_ext_lower(path);
    rassert(!ext.empty(), "Output path must have an extension", path);

    if (ext == "png") {
        save_png(img, path);
        return;
    }
    if (ext == "jpg" || ext == "jpeg") {
        save_jpeg(img, path, jpg_quality);
        return;
    }

    rassert(false, "Unsupported output extension", ext, path);
}

#endif

} // namespace libimages
