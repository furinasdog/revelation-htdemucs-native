// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace revelation::onnx {
// Select one API table before constructing any ORT objects. Released sessions and tensors
// must use the same table for the lifetime of this module.
inline bool initialize() {
    static const bool accelerated = [] {
        Ort::InitApi();
#ifdef _WIN32
        HMODULE self{};
        wchar_t path[32768];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&initialize), &self)) return false;
        const DWORD length = GetModuleFileNameW(self, path, 32768);
        if (!length || length >= 32768) return false;
        const auto dir = std::filesystem::path(path).parent_path() / L"cuda";
        if (GetFileAttributesW((dir / L"onnxruntime_providers_cuda.dll").c_str()) == INVALID_FILE_ATTRIBUTES) return false;
        constexpr DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        // cuDNN loads its companion libraries by name when a graph is first executed.
        // Resolve them inside the private add-on without changing the process search path.
        const wchar_t* libraries[] = {
            L"zlibwapi.dll", L"cudart64_12.dll", L"cublasLt64_12.dll", L"cublas64_12.dll", L"nvJitLink_120_0.dll", L"cufft64_11.dll",
            L"nvrtc-builtins64_128.dll", L"nvrtc64_120_0.dll", L"cudnn64_9.dll", L"cudnn_graph64_9.dll",
            L"cudnn_ops64_9.dll", L"cudnn_adv64_9.dll", L"cudnn_cnn64_9.dll", L"cudnn_heuristic64_9.dll",
            L"cudnn_engines_precompiled64_9.dll", L"cudnn_engines_runtime_compiled64_9.dll",
            L"onnxruntime_providers_shared.dll", L"onnxruntime.dll"
        };
        constexpr auto count = sizeof(libraries) / sizeof(*libraries);
        HMODULE loaded[count]{};
        size_t n = 0;
        for (; n < count; ++n) {
            loaded[n] = LoadLibraryExW((dir / libraries[n]).c_str(), nullptr, flags);
            if (!loaded[n]) break;
        }
        if (n == count) {
            auto entry = reinterpret_cast<decltype(&OrtGetApiBase)>(GetProcAddress(loaded[n-1], "OrtGetApiBase"));
            const OrtApi* api = entry ? entry()->GetApi(ORT_API_VERSION) : nullptr;
            if (api) { Ort::InitApi(api); return true; }
        }
        while (n) FreeLibrary(loaded[--n]);
        std::fputs("ONNX: incomplete or incompatible CUDA add-on; using CPU.\n", stderr);
#endif
        return false;
    }();
    return accelerated;
}

template<class... Model>
Ort::Session session(const Ort::Env& env, const Ort::SessionOptions& cpu, Model... model) {
    if (initialize()) {
        try {
            auto gpu = cpu.Clone();
            OrtCUDAProviderOptions provider{};
            gpu.AppendExecutionProvider_CUDA(provider);
            Ort::Session result(env, model..., gpu);
            std::fputs("ONNX: CUDA execution provider active.\n", stderr);
            return result;
        } catch (const Ort::Exception& error) {
            std::fprintf(stderr, "ONNX: CUDA unavailable for model; using CPU: %s\n", error.what());
        }
    }
    return Ort::Session(env, model..., cpu);
}
}
