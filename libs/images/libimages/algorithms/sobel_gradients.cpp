#include "sobel_gradients.h"

#include <algorithm>
#include <cmath>

#include <libbase/runtime_assert.h>

namespace libimages {

static constexpr float kPi = 3.14159265358979323846f;

float wrap_angle_deg(float a) {
  a = std::fmod(a, 360.0f);
  if (a < 0.0f) a += 360.0f;
  return a;
}

float angle_diff_deg(float a, float b) {
  float d = std::fabs(a - b);
  if (d > 180.0f) d = 360.0f - d;
  return d;
}

Gradients sobel_gradients(const image32f& gray_blurred) {
  rassert(gray_blurred.channels() == 1, "sobel_gradients expects grayscale", gray_blurred.channels());

  const int w = gray_blurred.width();
  const int h = gray_blurred.height();

  Gradients g{
      image32f(w, h, 1),
      image32f(w, h, 1),
      image32f(w, h, 1),
      image32f(w, h, 1),
  };

  g.dx.fill(0.0f);
  g.dy.fill(0.0f);
  g.mag.fill(0.0f);
  g.angle.fill(0.0f);

  if (w < 3 || h < 3) return g;

  for (int j = 1; j < h - 1; ++j) {
    for (int i = 1; i < w - 1; ++i) {
      const float p00 = gray_blurred(j - 1, i - 1);
      const float p01 = gray_blurred(j - 1, i);
      const float p02 = gray_blurred(j - 1, i + 1);

      const float p10 = gray_blurred(j, i - 1);
      const float p12 = gray_blurred(j, i + 1);

      const float p20 = gray_blurred(j + 1, i - 1);
      const float p21 = gray_blurred(j + 1, i);
      const float p22 = gray_blurred(j + 1, i + 1);

      const float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
      const float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

      g.dx(j, i) = gx;
      g.dy(j, i) = gy;

      const float m = std::sqrt(gx * gx + gy * gy);
      g.mag(j, i) = m;
      g.angle(j, i) = (m > 1e-6f) ? wrap_angle_deg(std::atan2(gy, gx) * 180.0f / kPi) : 0.0f;
    }
  }

  return g;
}

image8u visualize_signed_to_u8(const image32f& img) {
  rassert(img.channels() == 1, "visualize_signed_to_u8 expects 1-channel", img.channels());

  const int w = img.width();
  const int h = img.height();

  float max_abs = 0.0f;
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      max_abs = std::max(max_abs, std::fabs(img(j, i)));
  if (max_abs < 1e-6f) max_abs = 1.0f;

  image8u out(w, h, 1);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      const float v = img(j, i) / max_abs; // [-1,1]
      int u = static_cast<int>(std::lround(127.5f + 127.5f * v));
      u = std::clamp(u, 0, 255);
      out(j, i) = static_cast<std::uint8_t>(u);
    }
  }
  return out;
}

static void hsv_to_rgb(float h, float s, float v, std::uint8_t* r, std::uint8_t* g, std::uint8_t* b) {
  const float hh = h * 6.0f;
  const int i = static_cast<int>(std::floor(hh));
  const float f = hh - static_cast<float>(i);

  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));

  float rr = 0.0f, gg = 0.0f, bb = 0.0f;
  switch (i % 6) {
    case 0: rr = v; gg = t; bb = p; break;
    case 1: rr = q; gg = v; bb = p; break;
    case 2: rr = p; gg = v; bb = t; break;
    case 3: rr = p; gg = q; bb = v; break;
    case 4: rr = t; gg = p; bb = v; break;
    case 5: rr = v; gg = p; bb = q; break;
  }

  auto to_u8 = [](float x) -> std::uint8_t {
    int u = static_cast<int>(std::lround(x * 255.0f));
    u = std::clamp(u, 0, 255);
    return static_cast<std::uint8_t>(u);
  };

  *r = to_u8(rr);
  *g = to_u8(gg);
  *b = to_u8(bb);
}

image8u visualize_angle_hsv(const image32f& angle_deg, const image32f& mag) {
  rassert(angle_deg.channels() == 1 && mag.channels() == 1, "visualize_angle_hsv expects 1-channel inputs");
  const int w = angle_deg.width();
  const int h = angle_deg.height();
  rassert(mag.width() == w && mag.height() == h, "angle/mag size mismatch");

  float max_mag = 0.0f;
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      max_mag = std::max(max_mag, mag(j, i));
  if (max_mag < 1e-6f) max_mag = 1.0f;

  image8u out(w, h, 3);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      const float a = wrap_angle_deg(angle_deg(j, i));
      const float h01 = a / 360.0f;
      const float v01 = std::clamp(mag(j, i) / max_mag, 0.0f, 1.0f);

      std::uint8_t r, g, b;
      hsv_to_rgb(h01, 1.0f, v01, &r, &g, &b);
      out(j, i, 0) = r;
      out(j, i, 1) = g;
      out(j, i, 2) = b;
    }
  }
  return out;
}

} // namespace app
