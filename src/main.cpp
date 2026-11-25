// SPDX-License-Identifier: GPL-3.0-only
#include "stems.hpp"
#include <windows.h>
#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {
fs::path native_path(fs::path path) {
    path.make_preferred();
    const auto& name = path.native();
    if (path.empty() || name.rfind(LR"(\\?\)", 0) == 0 || name.rfind(LR"(\\.\)", 0) == 0)
        return path;
    // Verbatim Windows paths must be absolute and have no "." or ".." components.
    if (path.is_relative() && !path.has_root_name())
        path = fs::current_path() / path;
    const auto absolute = fs::absolute(path).lexically_normal().make_preferred().wstring();
    if (absolute.rfind(LR"(\\)", 0) == 0)
        return LR"(\\?\UNC\)" + absolute.substr(2);
    return LR"(\\?\)" + absolute;
}
}

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
        const auto audio = native_path(args[L"--input"]);
        const auto model = native_path(args[L"--model"]);
        const auto output = native_path(args[L"--output"]);
        const auto progress = native_path(args[L"--progress"]);
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
        fs::path ffmpeg = args[L"--ffmpeg"];
        if (ffmpeg.empty()) ffmpeg = L"ffmpeg.exe";
        if (ffmpeg.has_parent_path()) ffmpeg = native_path(ffmpeg);
        stems::decode(ffmpeg, audio, pcm);
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
