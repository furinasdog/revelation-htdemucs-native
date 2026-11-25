# Revelation HTDemucs Native

Windows x64 的 HTDemucs 四轨分离命令行工具。C++ 使用 ONNX Runtime 执行模型，FFmpeg 负责解码音频。输出为等长的人声、鼓、贝斯和其他伴奏，均采用 44.1 kHz 双声道 PCM16 WAV。

## 构建

需要 Visual Studio 的 x64 C++ 工具链、Windows 10 SDK 和 CMake 3.21 或更高版本。在项目根目录执行 PowerShell 命令。

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
cmake --install build --config Release --prefix dist
dist\revelation-stems.exe --help
```

CMake 从 NuGet 获取 Microsoft.ML.OnnxRuntime 1.23.2，并校验包的 SHA-256。离线配置时可加上 `-DONNXRUNTIME_DIR=C:/sdk/onnxruntime-1.23.2`，其中 `include/` 和 `lib/` 分别包含 C/C++ 头文件和 x64 的 `onnxruntime.lib`、`onnxruntime.dll`。安装目录内的 DLL 应与可执行文件一起使用。离线 SDK 的分发者还应附上对应版本的 ONNX Runtime 许可证与第三方声明。

## 模型

采用标准四轨 HTDemucs 的单文件 ONNX 转换。文件大小为 165,612,636 字节，权重使用 FP16 存储，输入输出为 FP32。[StemSplit 模型说明](https://huggingface.co/StemSplitio/htdemucs-onnx/blob/d54ed9eb60e258ea82131c6ee14578628816456a/README.md)

```powershell
$model = 'htdemucs_fp16weights.onnx'
$url = 'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/d54ed9eb60e258ea82131c6ee14578628816456a/htdemucs_fp16weights.onnx'
$expected = 'd05c269d0178d2a72ad484b10b11dd370193fc923201c3b27a99f848745db70a'
if (!(Test-Path -LiteralPath $model)) {
    Invoke-WebRequest -Uri $url -OutFile "$model.part"
    if ((Get-FileHash -LiteralPath "$model.part" -Algorithm SHA256).Hash -ne $expected) {
        throw 'Downloaded model failed SHA-256 verification'
    }
    Move-Item -LiteralPath "$model.part" -Destination $model
}
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash -ne $expected) {
    throw 'Model failed SHA-256 verification'
}
```

## 运行

```powershell
dist\revelation-stems.exe --input "song.flac" --model htdemucs_fp16weights.onnx --output "work/take-01/stems" --ffmpeg "C:/tools/ffmpeg.exe" --progress "work/take-01/progress.txt"
```

- `--input` 接受本地音频文件。FFmpeg 读取第一条音频流，再重采样至双声道 44.1 kHz。省略 `--ffmpeg` 时使用系统可找到的 `ffmpeg.exe`。
- `--output` 指定一个尚未存在的目录。程序在其父目录暂存解码音频和分轨，全部写入成功后发布 `drums.wav`、`bass.wav`、`other.wav`、`vocals.wav`。每轨最大长度受 RIFF 的 4 GiB 文件限制。
- `--progress` 指定可选的新文件。开始推理时写入 `0 N`，随后每完成一个窗口更新成 `K N`。轮询遇到空行时可在下次重读；退出码 0 表示四轨均已发布。错误写到标准错误。
- 桌面程序可为每次推理分配独立工作目录。强制结束进程时，Windows Job Object 同时结束 FFmpeg，调用方随后可删除整个工作目录中的临时文件。

模型按 343,980 帧（7.8 秒）分段，步长 257,985 帧，重叠区以线性权重相加。尾段右侧补零，输出裁回解码得到的原始帧数。程序只在内存中保留当前段与上一段的重叠部分。CPU 后端可独立运行。安装有兼容 Revelation CUDA 组件时，程序会尝试可执行文件旁的 `cuda/`，并在创建 GPU 会话失败时使用 CPU。

## 与原模型比对

使用 44.1 kHz 双声道 WAV 音乐片段，时长宜超过一个窗口。先用上面的命令输出原生四轨，再在安装了 Demucs 4.0.1、PyTorch、SoundFile、NumPy、SciPy 的 Python 环境执行以下命令。验证脚本对官方检查点作 SHA-256 校验，采用相同窗口和重叠策略，按声道比较误差与采样延迟，并生成试听 WAV。[官方 HTDemucs 检查点](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th)

```powershell
python tests/compare_official.py mix.wav 955717e8-8726e21a.th comparison --native work/take-01/stems
```

在 Windows x64 CPU 上，以两段 13.17 秒和 14.29 秒的立体声音乐做过对照。两段的四轨各自都与官方模型输出等长，两声道的互相关峰值均在 0 采样偏移。后一段包含较清楚的鼓、贝斯和人声，逐轨逐声道的信噪比为 53.4–83.9 dB。可用 Python 3.11 或更高版本复测，`*-AB.wav` 依次播放官方和原生输出，中间留有短暂停顿。

同一段 14.29 秒音乐使用 Revelation CUDA 附加包复测，四轨两声道仍为 0 采样偏移，逐轨逐声道信噪比为 50.4–75.4 dB。

Windows 长路径回归会将本地模型与 FFmpeg 放在超过 320 个字符的中文目录，用真实推理检查四轨、进度与临时目录的清理。此项只需 Python 标准库。

```powershell
python tests/windows_long_paths.py dist/revelation-stems.exe htdemucs_fp16weights.onnx C:/tools/ffmpeg.exe
```

本项目代码采用 GPL-3.0-only。模型、推理库及外部解码器的来源和条款见[第三方声明](THIRD_PARTY.md)。
