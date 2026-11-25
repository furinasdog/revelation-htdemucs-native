// SPDX-License-Identifier: GPL-3.0-only
#include "stems.hpp"
#include <windows.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
struct Handle {
    HANDLE value;
    explicit Handle(HANDLE handle) : value(handle) {}
    Handle(const Handle&) = delete;
    ~Handle() { if (value) CloseHandle(value); }
};
void check(BOOL ok, const char* action) {
    if (!ok) throw std::system_error(GetLastError(), std::system_category(), action);
}
// Quote an individual argv element using the Windows C runtime parsing rules.
std::wstring quoted(const std::wstring& arg) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t ch : arg) {
        if (ch == L'\\') { ++slashes; continue; }
        out.append(ch == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        out += ch;
        slashes = 0;
    }
    return out.append(2 * slashes, L'\\') + L"\"";
}
}

void stems::decode(const std::filesystem::path& ffmpeg, const std::filesystem::path& audio,
                   const std::filesystem::path& pcm) {
    std::wstring command = quoted(ffmpeg.wstring()) + L" -hide_banner -loglevel error -nostdin -n"
        L" -protocol_whitelist file -i " + quoted(audio.wstring()) +
        L" -map 0:a:0 -vn -sn -dn -ar 44100 -ac 2 -f f32le " + quoted(pcm.wstring());
    Handle job(CreateJobObjectW(nullptr, nullptr));
    check(job.value != nullptr, "create decoder job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    check(SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)), "configure decoder job");

    // Assign at creation so killing the separation process cannot orphan FFmpeg.
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    auto list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    check(InitializeProcThreadAttributeList(list, 1, 0, &bytes), "prepare decoder process");
    struct Cleanup { LPPROC_THREAD_ATTRIBUTE_LIST list; ~Cleanup() { DeleteProcThreadAttributeList(list); } } cleanup{list};
    check(UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST, &job.value, sizeof(HANDLE), nullptr, nullptr),
          "attach decoder to job");
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.lpAttributeList = list;
    PROCESS_INFORMATION child{};
    check(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE,
                         CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo, &child),
          "start FFmpeg");
    Handle process(child.hProcess), thread(child.hThread);
    check(WaitForSingleObject(process.value, INFINITE) == WAIT_OBJECT_0, "wait for FFmpeg");
    DWORD code = 0;
    check(GetExitCodeProcess(process.value, &code), "read FFmpeg exit code");
    if (code != 0) throw std::runtime_error("FFmpeg decoding failed (exit " + std::to_string(code) + ")");
}
