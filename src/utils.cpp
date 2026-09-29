#include "utils.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>


void set_manual_seed(int seed){
  torch::manual_seed(seed);

  if(torch::mps::is_available()){
    torch::mps::manual_seed(seed);
  }
}

torch::Device get_device(){
    return torch::mps::is_available() ? torch::Device(torch::kMPS) : torch::Device(torch::kCPU);
}

bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data_path> <output_path>"
              << std::endl;
    return false;
  }

  data_path = argv[1];
  output_path = argv[2];
  return true;
}

std::vector<char> load_binary_file(const std::filesystem::path &file_path) {
  std::ifstream input(file_path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));
  input.close();
  return bytes;
}

torch::Tensor load_tensor(const std::filesystem::path &file_path) {
  std::vector<char> f = load_binary_file(file_path);
  torch::IValue x = torch::pickle_load(f);
  return x.toTensor();
}

float load_focal(const std::filesystem::path &file_path) {
  torch::Tensor focal_tensor = load_tensor(file_path);
  return focal_tensor.item<float>();
}

namespace {

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t length) {
  crc = ~crc;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

void write_be32(std::ostream &out, uint32_t value) {
  const uint8_t bytes[4] = {
      static_cast<uint8_t>((value >> 24) & 0xff),
      static_cast<uint8_t>((value >> 16) & 0xff),
      static_cast<uint8_t>((value >> 8) & 0xff),
      static_cast<uint8_t>(value & 0xff)};
  out.write(reinterpret_cast<const char *>(bytes), 4);
}

void write_png_chunk(std::ostream &out, const char type[4],
                     const std::vector<uint8_t> &data) {
  write_be32(out, static_cast<uint32_t>(data.size()));
  out.write(type, 4);
  if (!data.empty()) {
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
  }
  uint32_t crc = crc32_update(0, reinterpret_cast<const uint8_t *>(type), 4);
  crc = crc32_update(crc, data.data(), data.size());
  write_be32(out, crc);
}

uint32_t adler32(const std::vector<uint8_t> &data) {
  uint32_t a = 1;
  uint32_t b = 0;
  for (uint8_t byte : data) {
    a = (a + byte) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

std::vector<uint8_t> zlib_store(const std::vector<uint8_t> &raw) {
  std::vector<uint8_t> out;
  out.push_back(0x78);
  out.push_back(0x01);
  size_t offset = 0;
  while (offset < raw.size() || raw.empty()) {
    const size_t remaining = raw.size() - offset;
    const size_t count = std::min<size_t>(65535, remaining);
    const bool last = offset + count >= raw.size();
    out.push_back(last ? 0x01 : 0x00);
    const uint16_t len = static_cast<uint16_t>(count);
    const uint16_t nlen = static_cast<uint16_t>(~len);
    out.push_back(static_cast<uint8_t>(len & 0xff));
    out.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(nlen & 0xff));
    out.push_back(static_cast<uint8_t>((nlen >> 8) & 0xff));
    out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
               raw.begin() + static_cast<std::ptrdiff_t>(offset + count));
    offset += count;
    if (last) {
      break;
    }
  }
  const uint32_t checksum = adler32(raw);
  out.push_back(static_cast<uint8_t>((checksum >> 24) & 0xff));
  out.push_back(static_cast<uint8_t>((checksum >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((checksum >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>(checksum & 0xff));
  return out;
}

}  // namespace

void save_image(const torch::Tensor &tensor,
                const std::filesystem::path &file_path) {
  // HxWx3 RGB in [0, 1]. Clamp instead of per-image min-max so saved frames
  // stay comparable with PSNR computed on the float render.
  const int height = static_cast<int>(tensor.size(0));
  const int width = static_cast<int>(tensor.size(1));
  auto pixels = tensor.detach()
                    .to(torch::kFloat32)
                    .clamp(0.0f, 1.0f)
                    .mul(255.0f)
                    .to(torch::kU8)
                    .to(torch::kCPU)
                    .contiguous();
  const auto *rgb = pixels.data_ptr<uint8_t>();

  std::vector<uint8_t> raw;
  raw.reserve(static_cast<size_t>(height) * (static_cast<size_t>(width) * 3 + 1));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const auto *row = rgb + static_cast<size_t>(y) * static_cast<size_t>(width) * 3;
    raw.insert(raw.end(), row, row + static_cast<size_t>(width) * 3);
  }

  std::ofstream out(file_path, std::ios::binary);
  const uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.write(reinterpret_cast<const char *>(signature), 8);

  std::vector<uint8_t> ihdr(13);
  auto store_be = [](std::vector<uint8_t> &dst, size_t at, uint32_t value) {
    dst[at] = static_cast<uint8_t>((value >> 24) & 0xff);
    dst[at + 1] = static_cast<uint8_t>((value >> 16) & 0xff);
    dst[at + 2] = static_cast<uint8_t>((value >> 8) & 0xff);
    dst[at + 3] = static_cast<uint8_t>(value & 0xff);
  };
  store_be(ihdr, 0, static_cast<uint32_t>(width));
  store_be(ihdr, 4, static_cast<uint32_t>(height));
  ihdr[8] = 8;
  ihdr[9] = 2;
  write_png_chunk(out, "IHDR", ihdr);
  write_png_chunk(out, "IDAT", zlib_store(raw));
  write_png_chunk(out, "IEND", {});
}


void render_and_save_test_view(const Renderer &renderer,
                               const torch::Tensor &testpose,
                               const std::filesystem::path &output_folder,
                               float near, float far, int n_samples,
                               int ray_chunk, int cur_iteration) {
  auto rendered_image =
      renderer.render(testpose, near, far, n_samples, ray_chunk);
  std::string file_path =
      output_folder / ("preview_" + std::to_string(cur_iteration) + ".png");
  save_image(rendered_image, file_path);
  std::cout << "Saved preview image to " << file_path << std::endl;
}

void render_and_save_orbit_views(const Renderer &renderer, int n_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius, float near, float far,
                                 int n_samples, int ray_chunk) {
  float elevation = -30.0f;

  for (int i = 0; i < n_frames; i++) {
    float azimuth = static_cast<float>(i) * 360.0f / n_frames;
    auto pose = create_spherical_pose(azimuth, elevation, radius);
    auto rendered_image = renderer.render(pose, near, far, n_samples, ray_chunk);
    std::string file_path =
        output_folder / ("frame_" + std::to_string(i) + ".png");
    save_image(rendered_image, file_path);
  }

  std::cout << "Saved " << n_frames << " frame(s) to "
            << output_folder.string() << std::endl;
}

torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius) {
  float phi = elevation * (M_PI / 180.0f);
  float theta = azimuth * (M_PI / 180.0f);

  // phi -> elevation -> rotation around x-axis
  // theta -> azimuth -> rotation around y-axis

  torch::Tensor c2w = create_translation_matrix(radius);
  c2w = create_phi_rotation_matrix(phi).matmul(c2w);
  c2w = create_theta_rotation_matrix(theta).matmul(c2w);
  c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 1.0f, 0.0f},
                       {0.0f, 1.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f}})
            .matmul(c2w);

  return c2w;
}

torch::Tensor create_translation_matrix(float t) {
  torch::Tensor t_mat = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                       {0.0f, 1.0f, 0.0f, 0.0f},
                                       {0.0f, 0.0f, 1.0f, t},
                                       {0.0f, 0.0f, 0.0f, 1.0f}});
  return t_mat;
}

torch::Tensor create_phi_rotation_matrix(float phi) {
  torch::Tensor phi_mat =
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
                     {0.0f, std::sin(phi), std::cos(phi), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return phi_mat;
}

torch::Tensor create_theta_rotation_matrix(float theta) {
  torch::Tensor theta_mat =
      torch::tensor({{std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return theta_mat;
}

