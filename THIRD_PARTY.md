# Third-party materials

| Component | Source and license |
| --- | --- |
| Native command-line program and ONNX session selection | GPL-3.0-only. The optional CUDA loader in `src/onnx_backend.hpp` comes from the [Revelation chart editor](https://github.com/furinasdog/revelation-chart-editor). The program's license is included in [LICENSE](LICENSE). |
| HTDemucs | The original Hybrid Transformer Demucs model is by Meta Platforms, Inc. and affiliates. The official Demucs 4.0.1 distribution includes its [MIT notice](licenses/Demucs-MIT.txt). [Demucs 4.0.1](https://pypi.org/project/demucs/4.0.1/) · [Upstream source](https://github.com/facebookresearch/demucs) |
| ONNX conversion and reference windowing | The [StemSplitio model card](https://huggingface.co/StemSplitio/htdemucs-onnx/blob/d54ed9eb60e258ea82131c6ee14578628816456a/README.md) identifies the ONNX export and accompanying reference program as MIT-licensed. The C++ runtime implements the model's documented input, output and overlap convention. Users obtain model weights separately. [Reference implementation](https://huggingface.co/StemSplitio/htdemucs-onnx/blob/d54ed9eb60e258ea82131c6ee14578628816456a/infer.py) |
| Microsoft ONNX Runtime | The CPU SDK comes from the official [Microsoft.ML.OnnxRuntime 1.23.2 package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.23.2). Its MIT license and bundled third-party notices are included by the default install target. [Source at v1.23.2](https://github.com/microsoft/onnxruntime/tree/v1.23.2) |
| FFmpeg | Invoked as a separate process. Supply a local executable and retain notices and corresponding sources as required by its build configuration. [FFmpeg licenses](https://ffmpeg.org/legal.html) |
