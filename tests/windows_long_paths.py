"""Run real HTDemucs inference with ordinary Windows paths longer than MAX_PATH."""

import argparse
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


def extended(path):
    # Use verbatim names for fixture setup even on hosts with LongPathsEnabled=0.
    name = str(path)
    if name.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + name[2:])
    return Path("\\\\?\\" + name)


def stage(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("ffmpeg", type=Path, help="FFmpeg executable used by the editor")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This regression exercises the Windows path and process APIs")
    root = Path(tempfile.mkdtemp(prefix="htdemucs-"))
    try:
        deep = root
        while len(str(deep)) < 320:
            deep /= "音乐工程 长目录 " + "编曲素材" * 7
        extended(deep).mkdir(parents=True)
        audio, model, ffmpeg = deep / "立体声.wav", deep / "模型.onnx", deep / "解码器.exe"
        stage(args.model.resolve(), extended(model))
        stage(args.ffmpeg.resolve(), extended(ffmpeg))
        frames = 44100
        with wave.open(str(extended(audio)), "wb") as f:
            f.setparams((2, 2, 44100, frames, "NONE", "not compressed"))
            # 441 repetitions of a bipolar stereo click exercise the real decoder.
            click = b"\x00\x20\x00\xe0" + b"\x00" * (4 * 99)
            f.writeframes(click * (frames // 100))
        output, progress = deep / "分析 缓存" / "四轨", deep / "进度.txt"
        env = os.environ.copy()
        env["PATH"] = str(args.ffmpeg.resolve().parent) + os.pathsep + env.get("PATH", "")
        subprocess.run([args.executable.resolve(), "--input", audio, "--model", model,
                        "--ffmpeg", ffmpeg, "--output", output, "--progress", progress],
                       check=True, env=env)
        audible = False
        for name in ("drums", "bass", "other", "vocals"):
            with wave.open(str(extended(output / (name + ".wav")))) as f:
                assert (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) == (2, 2, 44100, frames)
                data = f.readframes(frames)
                assert len(data) == frames * 4
                audible |= any(data)
        assert audible
        assert extended(progress).read_text().strip() == "1 1"
        assert not list(extended(output.parent).glob(".htdemucs-*"))
        print(f"PASS: four full stems, progress and cleanup at {len(str(output))} characters")
    finally:
        shutil.rmtree(extended(root))


if __name__ == "__main__":
    main()
