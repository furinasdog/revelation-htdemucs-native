// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <filesystem>

namespace stems {
void decode(const std::filesystem::path& ffmpeg, const std::filesystem::path& audio,
            const std::filesystem::path& pcm);
void separate(const std::filesystem::path& pcm, const std::filesystem::path& model,
              const std::filesystem::path& output, const std::filesystem::path& progress);
}
