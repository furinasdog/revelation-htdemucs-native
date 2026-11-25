// SPDX-License-Identifier: GPL-3.0-only
#include "stems.hpp"
#include <windows.h>
#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int wmain(int argc, wchar_t** argv) {
    constexpr char usage[] = "Usage: revelation-stems --input AUDIO --model HTDEMUCS.onnx --output NEW_DIRECTORY [--ffmpeg PATH] [--progress FILE]";
    if (argc == 2 && std::wstring(argv[1]) == L"--help") {
        std::cout << usage << '\n';
        return 0;
    }
    try {
        std::map<std::wstring, fs::path> args;
        for (int i = 1; i < argc; i += 2) {
            std::wstring name(argv[i]);
            if (i + 1 >= argc || (name != L"--input" && name != L"--output" && name != L"--ffmpeg" &&
                                  name != L"--model" && name != L"--progress") ||
                !args.emplace(name, argv[i + 1]).second)
                throw std::invalid_argument("Invalid or duplicate command line option");
        }
        if (args[L"--input"].empty() || args[L"--model"].empty() || args[L"--output"].empty())
            throw std::invalid_argument(usage);
        const fs::path audio = fs::absolute(args[L"--input"]);
        const fs::path model = fs::absolute(args[L"--model"]);
        const fs::path output = fs::absolute(args[L"--output"]);
        const fs::path progress = args[L"--progress"];
        if (!fs::is_regular_file(audio) || !fs::is_regular_file(model))
            throw std::runtime_error("Input audio and model must be readable local files");
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists");
        if (!progress.empty() && fs::exists(progress))
            throw std::runtime_error("Progress file already exists; provide a fresh run path");
        fs::create_directories(output.parent_path());
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto temp = output.parent_path() / (L".htdemucs-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(nonce));
        if (!fs::create_directory(temp)) throw std::runtime_error("Cannot create a private staging directory");
        struct Cleanup { fs::path dir; ~Cleanup() { std::error_code ec; fs::remove_all(dir, ec); } } cleanup{temp};
        const auto pcm = temp / "mixture.f32";
        stems::decode(args[L"--ffmpeg"].empty() ? fs::path(L"ffmpeg.exe") : args[L"--ffmpeg"], audio, pcm);
        const auto staging = temp / "stems";
        fs::create_directory(staging);
        stems::separate(pcm, model, staging, progress);
        fs::rename(staging, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "revelation-stems: " << error.what() << '\n';
        return 1;
    }
}
