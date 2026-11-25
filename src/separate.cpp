// SPDX-License-Identifier: GPL-3.0-only
#include "stems.hpp"
#include "onnx_backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr int rate = 44100;
constexpr size_t segment = 343980, overlap = segment / 4, stride = segment - overlap;
constexpr size_t channels = 2, sources = 4;
constexpr std::array<const char*, sources> names{"drums.wav", "bass.wav", "other.wav", "vocals.wav"};

template<class T> void write_le(std::ostream& out, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
void wav_header(std::ostream& out, uint32_t bytes) {
    out.write("RIFF", 4);
    write_le<uint32_t>(out, 36 + bytes);
    out.write("WAVEfmt ", 8);
    write_le<uint32_t>(out, 16);
    write_le<uint16_t>(out, 1);
    write_le<uint16_t>(out, channels);
    write_le<uint32_t>(out, rate);
    write_le<uint32_t>(out, rate * channels * 2);
    write_le<uint16_t>(out, channels * 2);
    write_le<uint16_t>(out, 16);
    out.write("data", 4);
    write_le<uint32_t>(out, bytes);
}
void publish_progress(const fs::path& path, uint64_t completed, uint64_t total) {
    if (path.empty()) return;
    // A polling reader may see an empty line during the update and retry.
    std::ofstream file(path, std::ios::trunc);
    file << completed << ' ' << total << '\n';
    file.close();
    if (!file) throw std::runtime_error("Cannot write progress file");
}
void validate_model(Ort::Session& session) {
    const Ort::AllocatorWithDefaultOptions allocator;
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1 ||
        std::string(session.GetInputNameAllocated(0, allocator).get()) != "mix" ||
        std::string(session.GetOutputNameAllocated(0, allocator).get()) != "stems")
        throw std::runtime_error("Expected the HTDemucs mix -> stems ONNX graph");
    auto input = session.GetInputTypeInfo(0);
    auto output = session.GetOutputTypeInfo(0);
    for (auto item : {std::make_pair(&input, std::vector<int64_t>{1, channels, segment}),
                      std::make_pair(&output, std::vector<int64_t>{1, sources, channels, segment})}) {
        auto type = item.first->GetTensorTypeAndShapeInfo();
        if (type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type.GetShape() != item.second)
            throw std::runtime_error("Unsupported HTDemucs tensor type or segment shape");
    }
}
}

void stems::separate(const fs::path& pcm, const fs::path& model, const fs::path& output, const fs::path& progress) {
    const uint64_t bytes = fs::file_size(pcm);
    const uint64_t frames = bytes / (channels * sizeof(float));
    if (frames == 0 || bytes % (channels * sizeof(float)) != 0)
        throw std::runtime_error("Empty or truncated decoded audio");
    if (frames > (std::numeric_limits<uint32_t>::max() - 36ULL) / (channels * 2))
        throw std::runtime_error("Audio exceeds the RIFF WAV size limit");
    const uint64_t windows = (frames + stride - 1) / stride;
    publish_progress(progress, 0, windows);

    revelation::onnx::initialize();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HTDemucs");
    Ort::SessionOptions settings;
    settings.SetIntraOpNumThreads(static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1U, 4U)));
    settings.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto session = revelation::onnx::session(env, settings, model.c_str());
    validate_model(session);
    std::ifstream input(pcm, std::ios::binary);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    std::array<std::ofstream, sources> tracks;
    for (size_t s = 0; s < sources; ++s) {
        tracks[s].exceptions(std::ios::failbit | std::ios::badbit);
        tracks[s].open(output / names[s], std::ios::binary);
        wav_header(tracks[s], static_cast<uint32_t>(frames * channels * 2));
    }
    std::vector<float> interleaved(channels * segment), mix(channels * segment);
    std::vector<float> previous(sources * channels * overlap);
    std::vector<int16_t> samples(channels * stride);
    const int64_t shape[]{1, channels, segment};
    const char* inputs[]{"mix"}, *outputs[]{"stems"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, mix.data(), mix.size(), shape, 3);
    for (uint64_t index = 0; index < windows; ++index) {
        const uint64_t start = index * stride;
        const size_t length = static_cast<size_t>(std::min<uint64_t>(segment, frames - start));
        input.seekg(static_cast<std::streamoff>(start * channels * sizeof(float)));
        input.read(reinterpret_cast<char*>(interleaved.data()), length * channels * sizeof(float));
        std::fill(mix.begin(), mix.end(), 0.0F);
        for (size_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < length; ++i) mix[c * segment + i] = interleaved[channels * i + c];
        auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        const float* stems = result[0].GetTensorData<float>();
        const size_t emit = std::min(length, stride);
        for (size_t s = 0; s < sources; ++s) {
            for (size_t c = 0; c < channels; ++c) {
                const float* current = stems + (s * channels + c) * segment;
                float* tail = previous.data() + (s * channels + c) * overlap;
                for (size_t i = 0; i < emit; ++i) {
                    float value = current[i];
                    if (index > 0 && i < overlap) {
                        float alpha = static_cast<float>(i) / (overlap - 1);
                        value = (1.0F - alpha) * tail[i] + alpha * value;
                    }
                    if (!std::isfinite(value)) throw std::runtime_error("Model returned a non-finite audio sample");
                    samples[channels * i + c] = static_cast<int16_t>(std::lround(
                        std::clamp(value * 32768.0F, -32768.0F, 32767.0F)));
                }
                if (index + 1 < windows) std::copy_n(current + stride, overlap, tail);
            }
            tracks[s].write(reinterpret_cast<const char*>(samples.data()), emit * channels * 2);
        }
        publish_progress(progress, index + 1, windows);
    }
    for (auto& track : tracks) track.close();
}
