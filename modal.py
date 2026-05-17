"""
Modal App for GPU-accelerated tasks
Deploy with: modal deploy modal_app.py
"""

import os
import shutil
from pathlib import Path

import modal

# Create the Modal app
app = modal.App("octavia-heavy-tasks")


# Define the image with all dependencies
def get_ml_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "numpy<2.0.0",
            "boto3",
            "torch",
            "torchaudio",
            "transformers<=4.40.2",
            "faster-whisper",
            "ctranslate2",
            "pydub",
            "piper-tts",
            "audio-separator",
            "spleeter",
            "tensorflow",
            "onnxruntime-gpu",
            "deep-translator",
            "gtts",
            "edge-tts",
            "scipy",
            "huggingface_hub",
        )
    )


# Separate image for pyannote 4.0 with required system dependencies
def get_pyannote4_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ffmpeg",
            "libsndfile1",
            "libavdevice-dev",
            "pkg-config",
            "libavformat-dev",
            "libavcodec-dev",
            "libswscale-dev",
            "libavutil-dev",
        )
        .pip_install("numpy<2.0.0", "torch", "torchaudio", "boto3")
        .run_commands("pip install uv")
        .uv_pip_install("pyannote.audio>=4.0.0", "av")
    )


# Separate image for WhisperX - unified transcription + diarization using uv
def get_whisperx_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ffmpeg",
            "libsndfile1",
            "libavdevice-dev",
            "pkg-config",
            "libavformat-dev",
            "libavcodec-dev",
            "libswscale-dev",
            "libavutil-dev",
        )
        .pip_install("numpy<2.0.0")
        .run_commands(
            "pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.2.2+cu121 torchaudio==2.2.2+cu121"
        )
        .run_commands("pip install uv")
        .uv_pip_install(
            "whisperx>=3.2.0", "pandas", "av", "soundfile", "pyannote.audio>=3.2.1", "boto3"
        )
    )


# Combined image for Demucs FT separation + WhisperX transcription/diarization
def get_demucs_whisperx_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ffmpeg",
            "libsndfile1",
            "libavdevice-dev",
            "pkg-config",
            "libavformat-dev",
            "libavcodec-dev",
            "libswscale-dev",
            "libavutil-dev",
        )
        .pip_install("numpy<2.0.0")
        .run_commands(
            "pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.2.2+cu121 torchaudio==2.2.2+cu121"
        )
        .run_commands("pip install uv")
        .uv_pip_install(
            "demucs",
            "whisperx>=3.2.0",
            "pandas",
            "av",
            "soundfile",
            "pyannote.audio>=3.2.1",
            "ffmpeg-python",
            "torchcodec",
            "boto3",
        )
    )


_R2_SECRET = modal.Secret.from_name("r2-credentials")
_PIPELINE_VOLUME = modal.Volume.from_name("octavia-pipeline", create_if_missing=True)
_PIPELINE_MOUNT = "/pipeline"


def _r2_secret_list():
    return [_R2_SECRET] if _R2_SECRET else []


def _pipeline_path(*parts: str) -> str:
    return os.path.join(_PIPELINE_MOUNT, *parts)


def _write_pipeline_file(local_path: str, *parts: str, commit: bool = True) -> str:
    dest_path = _pipeline_path(*parts)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(local_path, dest_path)
    if commit:
        _PIPELINE_VOLUME.commit()
    return dest_path


def _resolve_input_path(audio_source, suffix: str = "") -> tuple[str, bool]:
    """
    Resolve a function input into a local filesystem path.

    Returns:
        (path, cleanup_needed)
    """
    import tempfile
    import time

    if isinstance(audio_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio_source)
            return temp_audio.name, True

    if isinstance(audio_source, str):
        if _storage_scheme(audio_source):
            return _download_storage_object(audio_source, suffix=suffix), True
        if audio_source.startswith(_PIPELINE_MOUNT):
            for _ in range(10):
                _PIPELINE_VOLUME.reload()
                if os.path.exists(audio_source):
                    return audio_source, False
                time.sleep(0.25)
            return audio_source, False
        if os.path.exists(audio_source):
            return audio_source, False

    return audio_source, False


def _storage_scheme(storage_path: str) -> bool:
    return isinstance(storage_path, str) and storage_path.startswith(("r2://", "b2://"))


def _parse_storage_path(storage_path: str):
    if not _storage_scheme(storage_path):
        return None, None
    bucket_name, remote_path = storage_path.split("://", 1)[1].split("/", 1)
    return bucket_name, remote_path


def _get_storage_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["R2_ENDPOINT"],
        region_name=os.getenv("R2_REGION", "auto"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def _download_storage_object(storage_path: str, suffix: str = "") -> str:
    bucket_name, remote_path = _parse_storage_path(storage_path)
    if not bucket_name or not remote_path:
        raise ValueError(f"Invalid storage path: {storage_path}")

    import tempfile

    client = _get_storage_client()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        local_path = temp_file.name

    client.download_file(bucket_name, remote_path, local_path)
    return local_path


def _upload_storage_file(local_path: str, bucket_name: str, remote_path: str, content_type: str):
    client = _get_storage_client()
    client.upload_file(
        local_path,
        bucket_name,
        remote_path,
        ExtraArgs={"ContentType": content_type},
    )
    return f"r2://{bucket_name}/{remote_path}"


def _publish_audio_output(
    local_path: str,
    bucket_name: str,
    remote_path: str,
    volume_parts: tuple[str, ...],
    content_type: str = "audio/wav",
    commit: bool = True,
) -> tuple[str, str]:
    """Persist a Modal-produced audio file to both R2 and the shared Volume."""
    storage_path = _upload_storage_file(local_path, bucket_name, remote_path, content_type)
    volume_path = _write_pipeline_file(local_path, *volume_parts, commit=commit)
    return storage_path, volume_path


def _publish_volume_output(local_path: str, *volume_parts: str) -> str:
    """Persist a Modal-produced audio file only to the shared Volume."""
    return _write_pipeline_file(local_path, *volume_parts)


def _publish_progress_event(
    job_id: str | None,
    progress: int,
    message: str,
    stage: str,
    metadata: dict | None = None,
) -> str | None:
    """
    Write a lightweight progress snapshot to R2 so the backend can surface it.
    """
    if not job_id:
        return None

    import json
    from datetime import datetime, timezone

    payload = {
        "job_id": job_id,
        "progress": int(progress),
        "message": message,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = metadata

    bucket_name = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
    remote_path = f"jobs/{job_id}/modal-progress.json"
    client = _get_storage_client()
    try:
        existing = client.get_object(Bucket=bucket_name, Key=remote_path)
        existing_payload = json.loads(existing["Body"].read().decode("utf-8"))
        existing_progress = int(existing_payload.get("progress", 0))
        if existing_progress > payload["progress"]:
            return f"r2://{bucket_name}/{remote_path}"
    except Exception:
        pass
    client.put_object(
        Bucket=bucket_name,
        Key=remote_path,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return f"r2://{bucket_name}/{remote_path}"


@app.function(image=get_ml_image(), timeout=86400, secrets=_r2_secret_list(), volumes={"/pipeline": _PIPELINE_VOLUME})
def remote_chunk_audio_by_speaker_boundaries(
    audio_source: str,
    diarization_result: dict,
    max_chunk_size_ms: int = 40000,
    min_chunk_size_ms: int = 500,
    job_id: str = None,
) -> dict:
    """
    Split audio into chunks on Modal using WhisperX speaker boundaries.

    The backend sends only diarization metadata; this function loads the source
    audio remotely, exports chunk files, and returns a manifest of storage paths.
    """
    import concurrent.futures
    import json
    import logging
    import math
    import os
    import subprocess
    import tempfile

    logger = logging.getLogger(__name__)

    def _get_audio_duration(audio_path: str) -> float:
        """Get audio duration in milliseconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to get audio duration: {result.stderr}")
        data = json.loads(result.stdout)
        return float(data["format"]["duration"]) * 1000

    def _extract_and_upload_chunk(spec: dict, audio_path: str, output_bucket: str, output_volume_job: str) -> dict:
        """Extract a single chunk using FFmpeg and upload it."""
        chunk_id = spec["chunk_id"]
        start_ms = spec["start_ms"]
        end_ms = spec["end_ms"]
        speaker_label = spec.get("speaker") or "UNKNOWN"

        chunk_filename = f"chunk_{chunk_id:04d}.wav"
        remote_path = f"jobs/{output_volume_job}/modal-chunks/{chunk_filename}"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_chunk_path = temp_file.name

        try:
            # Extract chunk using FFmpeg — codec copy for speed on WAV input
            start_sec = start_ms / 1000.0
            duration_sec = (end_ms - start_ms) / 1000.0
            cmd = [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-ss", str(start_sec),
                "-t", str(duration_sec),
                "-c", "copy",
                temp_chunk_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg extraction failed: {result.stderr}")

            # Upload to R2
            storage_path = _upload_storage_file(temp_chunk_path, output_bucket, remote_path, "audio/wav")

            # Copy to volume — defer commit, batched at the end
            volume_path_actual = _write_pipeline_file(
                temp_chunk_path, *("chunks", output_volume_job, chunk_filename), commit=False
            )

            return {
                "chunk_id": chunk_id,
                "speaker": speaker_label,
                "voice_id": diarization_result.get("speaker_voice_map", {}).get(speaker_label, {}).get("voice_id"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "audio_storage_path": storage_path,
                "audio_volume_path": volume_path_actual,
            }
        finally:
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)

    try:
        _publish_progress_event(
            job_id,
            24,
            "Modal speaker-boundary chunking started",
            "speaker_chunking",
            {
                "max_chunk_size_ms": int(max_chunk_size_ms),
                "min_chunk_size_ms": int(min_chunk_size_ms),
            },
        )

        if not diarization_result or not diarization_result.get("segments"):
            return {"success": False, "error": "No diarization segments provided"}

        audio_path, cleanup_needed = _resolve_input_path(audio_source, suffix=".wav")
        try:
            duration_ms = _get_audio_duration(audio_path)

            segments = diarization_result.get("segments", [])
            voice_map = diarization_result.get("speaker_voice_map", {})
            speaker_turns = []
            current_turn = None

            for segment in segments:
                speaker = segment.get("speaker")
                start_s = float(segment.get("start", 0.0))
                end_s = float(segment.get("end", start_s))

                if (
                    current_turn
                    and speaker == current_turn["speaker"]
                    and (start_s - current_turn["end_s"]) < 1.0
                ):
                    current_turn["end_s"] = end_s
                else:
                    if current_turn:
                        speaker_turns.append(current_turn)
                    current_turn = {
                        "speaker": speaker,
                        "start_s": start_s,
                        "end_s": end_s,
                    }

            if current_turn:
                speaker_turns.append(current_turn)

            if not speaker_turns:
                return {"success": False, "error": "No speaker turns could be derived"}

            chunk_specs = []
            chunk_id = 0
            max_size_ms = max(int(max_chunk_size_ms or 40000), int(min_chunk_size_ms or 500))
            min_size_ms = int(min_chunk_size_ms or 500)

            for turn in speaker_turns:
                start_ms = max(0, int(turn["start_s"] * 1000))
                end_ms = min(duration_ms, int(turn["end_s"] * 1000))
                if end_ms <= start_ms:
                    continue

                turn_duration = end_ms - start_ms
                if turn_duration > max_size_ms:
                    num_splits = max(1, int(math.ceil(turn_duration / max_size_ms)))
                    split_duration = max(int(turn_duration / num_splits), min_size_ms)
                    for split_index in range(num_splits):
                        split_start = start_ms + (split_index * split_duration)
                        split_end = min(split_start + split_duration, end_ms)
                        if split_end - split_start < min_size_ms:
                            continue
                        chunk_specs.append(
                            {
                                "chunk_id": chunk_id,
                                "speaker": turn["speaker"],
                                "start_ms": split_start,
                                "end_ms": split_end,
                            }
                        )
                        chunk_id += 1
                else:
                    chunk_specs.append(
                        {
                            "chunk_id": chunk_id,
                            "speaker": turn["speaker"],
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                        }
                    )
                    chunk_id += 1

            if not chunk_specs:
                return {"success": False, "error": "No valid chunks were generated"}

            logger.info(
                f"[SPEAKER_CHUNK] Derived {len(chunk_specs)} chunks from "
                f"{len(speaker_turns)} speaker turns"
            )

            output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
            output_volume_job = job_id or "adhoc"

            total_chunks = len(chunk_specs)

            milestone_map = {
                math.ceil(total_chunks * 0.10): 10,
                math.ceil(total_chunks * 0.25): 25,
                math.ceil(total_chunks * 0.50): 50,
                math.ceil(total_chunks * 0.75): 75,
                total_chunks: 100,
            }

            # Parallel extraction and upload
            completed = 0
            manifest = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, total_chunks)) as executor:
                futures = {
                    executor.submit(_extract_and_upload_chunk, spec, audio_path, output_bucket, output_volume_job): spec
                    for spec in chunk_specs
                }
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    manifest.append(record)
                    completed += 1
                    if completed == 1 or completed == total_chunks or completed % 50 == 0:
                        logger.info(
                            f"[SPEAKER_CHUNK] Processed chunk {record['chunk_id']:04d}: "
                            f"{record['start_ms'] / 1000:.2f}s - {record['end_ms'] / 1000:.2f}s ({record['speaker']})"
                        )
                    if completed in milestone_map:
                        _publish_progress_event(
                            job_id,
                            milestone_map[completed],
                            f"Modal speaker-boundary chunking progress: {milestone_map[completed]}%",
                            "speaker_chunking",
                            {
                                "total_chunks": total_chunks,
                                "completed_chunks": completed,
                            },
                        )

            # Single batched volume commit after all chunks written
            _PIPELINE_VOLUME.commit()

            manifest.sort(key=lambda x: x["chunk_id"])

            _publish_progress_event(
                job_id,
                25,
                f"Modal speaker-boundary chunking complete ({len(manifest)} chunks)",
                "speaker_chunking",
                {"total_chunks": len(manifest)},
            )
            logger.info(f"[SPEAKER_CHUNK] Modal chunking complete: {len(manifest)} chunks")
            return {
                "success": True,
                "total_chunks": len(manifest),
                "chunks": manifest,
                "duration_ms": duration_ms,
            }
        finally:
            if cleanup_needed and os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        logger.error(f"[SPEAKER_CHUNK] Failed: {e}")
        return {"success": False, "error": str(e)}


PIPER_VOICE_CACHE = {}


def _generate_piper_tts(text: str) -> bytes:
    """Generate audio using Piper TTS for Armenian"""
    import io

    import numpy as np
    import scipy.io.wavfile
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    global_cache_key = "davit312/piper-TTS-Armenian:v3/hy_AM-gor-medium.onnx"

    if global_cache_key in PIPER_VOICE_CACHE:
        voice = PIPER_VOICE_CACHE[global_cache_key]
    else:
        model_path = hf_hub_download(
            repo_id="davit312/piper-TTS-Armenian", filename="v3/hy_AM-gor-medium.onnx"
        )
        voice = PiperVoice.load(model_path)
        PIPER_VOICE_CACHE[global_cache_key] = voice

    audio_generator = voice.synthesize(text)
    audio = None
    for chunk in audio_generator:
        chunk_audio = chunk.audio_float_array
        if audio is None:
            audio = chunk_audio
        else:
            audio = np.concatenate([audio, chunk_audio])

    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, 22050, audio)
    buffer.seek(0)

    return buffer.getvalue()


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_transcribe(audio_source, model_size: str = "base"):
    """Transcription using Faster-Whisper on Modal T4 GPU"""
    from faster_whisper import WhisperModel

    temp_path, cleanup_needed = _resolve_input_path(audio_source, suffix=".wav")

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        segments, info = model.transcribe(temp_path, beam_size=5)
        result = {"text": "", "segments": [], "language": info.language}
        for segment in segments:
            result["text"] += segment.text + " "
            result["segments"].append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
        result["text"] = result["text"].strip()
        return result
    finally:
        if cleanup_needed and os.path.exists(temp_path):
            os.remove(temp_path)


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_generate_subtitles(
    media_source,
    output_format: str = "srt",
    language: str = "auto",
    model_size: str = "base",
    job_id: str = None,
) -> dict:
    """
    Generate subtitles from an audio/video source on Modal.

    The source file is resolved from R2/B2 or the shared pipeline volume, then
    transcribed with Faster-Whisper. The resulting subtitle artifact is written
    to the shared volume and uploaded back to R2 for the backend to consume.
    """
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from faster_whisper import WhisperModel

    logger = logging.getLogger(__name__)

    def _format_srt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _segments_to_srt(segments: list[dict]) -> str:
        lines = []
        for index, segment in enumerate(segments, start=1):
            lines.append(str(index))
            lines.append(
                f"{_format_srt_timestamp(float(segment.get('start', 0.0)))} --> "
                f"{_format_srt_timestamp(float(segment.get('end', 0.0)))}"
            )
            lines.append(segment.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    def _segments_to_vtt(segments: list[dict]) -> str:
        lines = ["WEBVTT", ""]
        for segment in segments:
            lines.append(
                f"{_format_vtt_timestamp(float(segment.get('start', 0.0)))} --> "
                f"{_format_vtt_timestamp(float(segment.get('end', 0.0)))}"
            )
            lines.append(segment.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    output_format = (output_format or "srt").lower()
    if output_format not in {"srt", "vtt"}:
        return {"success": False, "error": f"Unsupported subtitle format: {output_format}"}

    _publish_progress_event(job_id, 12, "Subtitle generation started", "subtitle_generation")

    source_path, cleanup_needed = _resolve_input_path(media_source, suffix=".mp4")
    temp_dir = tempfile.mkdtemp()
    audio_path = source_path

    try:
        source_ext = Path(source_path).suffix.lower()
        if source_ext in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            audio_path = os.path.join(temp_dir, "subtitle_audio.wav")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                source_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "error": f"FFmpeg audio extraction failed: {result.stderr}"}

        _publish_progress_event(job_id, 22, "Subtitle transcription started", "subtitle_generation")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        transcribe_kwargs = {"beam_size": 5}
        if language and language.lower() != "auto":
            transcribe_kwargs["language"] = language

        segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)
        segments = []
        text_parts = []
        for segment in segments_iter:
            segment_data = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            segments.append(segment_data)
            if segment_data["text"]:
                text_parts.append(segment_data["text"])

        if not segments:
            return {"success": False, "error": "No subtitle segments were generated"}

        subtitle_content = _segments_to_srt(segments) if output_format == "srt" else _segments_to_vtt(segments)
        output_filename = f"subtitles_{job_id}.{output_format}" if job_id else f"subtitles.{output_format}"
        temp_output_path = os.path.join(temp_dir, output_filename)

        with open(temp_output_path, "w", encoding="utf-8") as f:
            f.write(subtitle_content)

        volume_path = _write_pipeline_file(
            temp_output_path,
            "subtitles",
            job_id or "adhoc",
            output_filename,
        )

        output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
        storage_path = _upload_storage_file(
            temp_output_path,
            output_bucket,
            f"jobs/{job_id or 'adhoc'}/outputs/{output_filename}",
            "text/plain" if output_format == "srt" else "text/vtt",
        )

        _publish_progress_event(
            job_id,
            95,
            f"Subtitle generation complete ({len(segments)} segments)",
            "subtitle_generation",
            {"segment_count": len(segments), "format": output_format},
        )

        return {
            "success": True,
            "language": info.language,
            "text": " ".join(text_parts).strip(),
            "segments": segments,
            "segment_count": len(segments),
            "output_format": output_format,
            "download_url": storage_path,
            "output_storage_path": storage_path,
            "output_volume_path": volume_path,
        }
    finally:
        if cleanup_needed and os.path.exists(source_path):
            os.remove(source_path)
        if audio_path != source_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_spleeter_separation(
    audio_source, model_variant: str = "spleeter:2stems", job_id: str = None
):
    """Vocal separation using Spleeter on Modal GPU - faster than UVR5"""
    import logging
    import os
    import shutil
    import tempfile

    from spleeter.separator import Separator

    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting Spleeter vocal separation on Modal GPU with model: {model_variant}"
    )
    _publish_progress_event(job_id, 25, f"Spleeter separation started ({model_variant})", "vocal_separation")

    if isinstance(audio_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_source)
            temp_path = temp_audio.name
    elif isinstance(audio_source, str) and audio_source.startswith(_PIPELINE_MOUNT):
        temp_path = audio_source
    elif _storage_scheme(audio_source):
        temp_path = _download_storage_object(audio_source, suffix=".wav")
    else:
        temp_path = audio_source

    output_dir = tempfile.mkdtemp()

    try:
        separator = Separator(model_variant)

        os.chdir(output_dir)
        # Override Spleeter's hard-coded 600s limit by passing a very large duration (10 hours)
        separator.separate_to_file(temp_path, output_dir, duration=36000.0)
        os.chdir(os.path.dirname(output_dir))

        return_dict = {}
        vocals_path = ""
        instrumental_path = ""

        for root, dirs, files in os.walk(output_dir):
            for filename in files:
                if filename.endswith(".wav"):
                    file_path = os.path.join(root, filename)
                    stem_lower = Path(file_path).stem.lower()

                    if "vocals" in stem_lower:
                        vocals_path = file_path
                    elif "accompaniment" in stem_lower or "instrumental" in stem_lower:
                        instrumental_path = file_path

        if (
            not instrumental_path
            and len([f for f in os.listdir(output_dir) if f.endswith(".wav")]) > 1
        ):
            for f in os.listdir(output_dir):
                if f.endswith(".wav") and "vocals" not in f.lower():
                    instrumental_path = os.path.join(output_dir, f)
                    break

        output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
        if vocals_path and os.path.exists(vocals_path):
            return_dict["vocals_storage_path"], return_dict["vocals_volume_path"] = _publish_audio_output(
                vocals_path,
                output_bucket,
                f"modal/stems/{os.path.basename(vocals_path)}",
                ("stems", os.path.basename(vocals_path)),
                "audio/wav",
            )
            logger.info(f"Vocals uploaded: {return_dict['vocals_storage_path']}")

        if instrumental_path and os.path.exists(instrumental_path):
            return_dict["background_storage_path"], return_dict["background_volume_path"] = _publish_audio_output(
                instrumental_path,
                output_bucket,
                f"modal/stems/{os.path.basename(instrumental_path)}",
                ("stems", os.path.basename(instrumental_path)),
                "audio/wav",
            )
            logger.info(f"Background uploaded: {return_dict['background_storage_path']}")
        else:
            logger.warning("No instrumental track found!")

        _publish_progress_event(job_id, 35, "Spleeter separation complete", "vocal_separation")

        return return_dict

    except Exception as e:
        logger.error(f"Spleeter separation failed: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@app.function(
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_vocal_separation(audio_source, job_id: str = None):
    """Vocal separation using UVR5 with GPU acceleration on Modal"""
    import logging
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from audio_separator.separator import Separator

    logger = logging.getLogger(__name__)
    logger.info("Starting UVR5 vocal separation on Modal GPU")
    _publish_progress_event(job_id, 25, "UVR5 separation started", "vocal_separation")

    if isinstance(audio_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_source)
            temp_path = temp_audio.name
    elif isinstance(audio_source, str) and audio_source.startswith(_PIPELINE_MOUNT):
        temp_path = audio_source
    elif _storage_scheme(audio_source):
        temp_path = _download_storage_object(audio_source, suffix=".wav")
    else:
        temp_path = audio_source

    output_dir = tempfile.mkdtemp()

    try:
        # Set environment for GPU acceleration
        os.environ["ORT_CUDA"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Initialize separator with GPU support
        # Use environment variable for model directory (configurable for different environments)
        model_file_dir = os.environ.get(
            "AUDIO_SEPARATOR_MODEL_DIR", "/tmp/audio-separator-models"
        )
        separator = Separator(output_format="WAV", model_file_dir=model_file_dir)

        # Load HP-UVR model (good quality, supports GPU)
        separator.load_model("1_HP-UVR.pth")
        logger.info(f"Loaded UVR5 model: {separator.model_friendly_name}")

        # Run separation in output directory
        original_cwd = os.getcwd()
        os.chdir(output_dir)
        results = separator.separate(temp_path)
        os.chdir(original_cwd)

        logger.info(f"Separation returned {len(results)} files")

        # Parse results
        return_dict = {}
        vocals_path = ""
        instrumental_path = ""

        base_name = os.path.splitext(os.path.basename(temp_path))[0]

        for output_file in results:
            filename = os.path.basename(output_file)
            final_path = os.path.join(output_dir, filename)

            # Move to output_dir if needed
            if os.path.abspath(output_file) != os.path.abspath(final_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                shutil.move(output_file, final_path)

            stem_lower = Path(final_path).stem.lower()

            if "vocals" in stem_lower:
                vocals_path = final_path
            elif (
                "instrumental" in stem_lower
                or "no_vocals" in stem_lower
                or "accompaniment" in stem_lower
            ):
                instrumental_path = final_path

        # Fallback: use remaining file as instrumental
        if not instrumental_path and len(results) > 1:
            for f in results:
                moved = os.path.join(output_dir, os.path.basename(f))
                if moved != vocals_path:
                    instrumental_path = moved
                    break

        # Return results
        output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
        if vocals_path and os.path.exists(vocals_path):
            return_dict["vocals_storage_path"], return_dict["vocals_volume_path"] = _publish_audio_output(
                vocals_path,
                output_bucket,
                f"modal/stems/{os.path.basename(vocals_path)}",
                ("stems", os.path.basename(vocals_path)),
                "audio/wav",
            )
            logger.info(f"Vocals uploaded: {return_dict['vocals_storage_path']}")

        if instrumental_path and os.path.exists(instrumental_path):
            return_dict["background_storage_path"], return_dict["background_volume_path"] = _publish_audio_output(
                instrumental_path,
                output_bucket,
                f"modal/stems/{os.path.basename(instrumental_path)}",
                ("stems", os.path.basename(instrumental_path)),
                "audio/wav",
            )
            logger.info(f"Background uploaded: {return_dict['background_storage_path']}")
        else:
            logger.warning("No instrumental track found!")

        _publish_progress_event(job_id, 35, "UVR5 separation complete", "vocal_separation")

        return return_dict

    except Exception as e:
        logger.error(f"UVR5 separation failed: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@app.function(image=get_ml_image(), timeout=300)
def remote_tts(text: str, voice_id: str, language: str):
    """Speech synthesis using gTTS or Edge-TTS on Modal (CPU is fine for TTS)"""
    import io

    from gtts import gTTS

    lang_map = {
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ja": "ja",
        "ko": "ko",
        "zh-cn": "zh-cn",
        "ar": "ar",
        "hi": "hi",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "sv": "sv",
        "swe": "sv",
        "te": "te",
        "kn": "kn",
        "swa": "sw",
        "sw": "sw",
        "hy": "hy",
        "nn": "no",
        "nb": "no",
    }

    # Male voices as default for edge_tts
    edge_tts_voices = {
        "en": "en-US-GuyNeural",
        "es": "es-MX-DarioNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "it": "it-IT-DiegoNeural",
        "pt": "pt-PT-AntonioNeural",
        "ru": "ru-RU-DmitryNeural",
        "ja": "ja-JP-KeitaNeural",
        "ko": "ko-KR-InhoNeural",
        "zh-cn": "zh-CN-YunxiNeural",
        "ar": "ar-SA-HamedNeural",
        "hi": "hi-IN-MadhurNeural",
        "nl": "nl-NL-MaartenNeural",
        "pl": "pl-PL-MarekNeural",
        "tr": "tr-TR-AhmetNeural",
        "sv": "sv-SE-MattiasNeural",
        "te": "te-IN-ChaitanyaNeural",
        "kn": "kn-IN-GaneshNeural",
        "swa": "sw-KE-RafikiNeural",
        "sw": "sw-TZ-DaudiNeural",
        "hy": "hy-AM-ArturNeural",
        "nn": "nb-NO-FelixNeural",
        "nb": "nb-NO-FelixNeural",
        "swe": "sv-SE-MattiasNeural",
    }

    tts_lang = lang_map.get(language, "en")

    audio_bytes = None

    if not audio_bytes and language.lower() in edge_tts_voices:
        try:
            import asyncio

            import edge_tts

            async def generate_edge():
                communicate = edge_tts.Communicate(text, edge_tts_voices[language])
                audio_data = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.write(chunk["data"])
                return audio_data.getvalue()

            audio_bytes = asyncio.run(generate_edge())
        except Exception as e:
            pass

    if not audio_bytes:
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes = audio_bytes.getvalue()

    return audio_bytes


@app.function(image=get_ml_image(), timeout=300)
def remote_translate_subtitles(
    subtitle_data: dict, source_lang: str = "auto", target_lang: str = "en"
) -> dict:
    """Translate subtitle segments using Google Translate on Modal (CPU is fine for translation)"""
    from deep_translator import GoogleTranslator

    lang_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "japanese": "ja",
        "korean": "ko",
        "chinese": "zh-CN",
        "arabic": "ar",
        "hindi": "hi",
        "dutch": "nl",
        "polish": "pl",
        "turkish": "tr",
        "swedish": "sv",
        "telugu": "te",
        "kannada": "kn",
        "swahili": "sw",
        "armenian": "hy",
        "norwegian": "no",
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ja": "ja",
        "ko": "ko",
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "ar": "ar",
        "hi": "hi",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "sv": "sv",
        "swe": "sv",
        "te": "te",
        "kn": "kn",
        "swa": "sw",
        "sw": "sw",
        "hy": "hy",
        "nn": "no",
        "nb": "no",
        "auto": "auto",
    }

    # Get language codes
    source_code = lang_map.get(source_lang.lower(), "auto")
    target_code = lang_map.get(target_lang.lower(), "en")

    # Initialize translator
    translator = GoogleTranslator(source=source_code, target=target_code)

    # Translate segments
    translated_segments = []
    total = len(subtitle_data.get("segments", []))

    for i, segment in enumerate(subtitle_data.get("segments", [])):
        text = segment.get("text", "")
        if text and text.strip():
            try:
                translated_text = translator.translate(text)
                segment["text"] = translated_text
            except Exception as e:
                # Keep original text if translation fails
                pass
        translated_segments.append(segment)

    return {
        "segments": translated_segments,
        "source_language": source_lang,
        "target_language": target_lang,
        "total_segments": total,
    }


@app.function(image=get_ml_image(), timeout=300)
def _generate_single_segment_audio(args: tuple) -> dict:
    """Generate audio for a single subtitle segment - runs in parallel"""
    import base64
    import io

    from gtts import gTTS
    from pydub import AudioSegment

    segment, tts_lang = args

    # Male voices as default for edge_tts
    edge_tts_voices = {
        "en": "en-US-GuyNeural",
        "es": "es-MX-DarioNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "it": "it-IT-DiegoNeural",
        "pt": "pt-PT-AntonioNeural",
        "ru": "ru-RU-DmitryNeural",
        "ja": "ja-JP-KeitaNeural",
        "ko": "ko-KR-InhoNeural",
        "zh-cn": "zh-CN-YunxiNeural",
        "ar": "ar-SA-HamedNeural",
        "hi": "hi-IN-MadhurNeural",
        "nl": "nl-NL-MaartenNeural",
        "pl": "pl-PL-MarekNeural",
        "tr": "tr-TR-AhmetNeural",
        "sv": "sv-SE-MattiasNeural",
        "te": "te-IN-ChaitanyaNeural",
        "kn": "kn-IN-GaneshNeural",
        "swa": "sw-KE-RafikiNeural",
        "sw": "sw-TZ-DaudiNeural",
        "hy": "hy-AM-ArturNeural",
        "nn": "nb-NO-FelixNeural",
        "nb": "nb-NO-FelixNeural",
        "swe": "sv-SE-MattiasNeural",
    }

    gtts_lang_map = {
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ja": "ja",
        "ko": "ko",
        "zh-cn": "zh-cn",
        "ar": "ar",
        "hi": "hi",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "sv": "sv",
        "swe": "sv",
        "te": "te",
        "kn": "kn",
        "swa": "sw",
        "sw": "sw",
        "hy": "hy",
        "nn": "no",
        "nb": "no",
    }

    try:
        segment_text = segment.get("text", "").strip()

        if not segment_text:
            silent_duration = int(
                (segment.get("end", 0) - segment.get("start", 0)) * 1000
            )
            silent_audio = AudioSegment.silent(duration=max(100, silent_duration))

            buffer = io.BytesIO()
            silent_audio.export(buffer, format="mp3")
            audio_bytes = buffer.getvalue()
        else:
            audio_bytes = None

            if not audio_bytes and tts_lang.lower() in edge_tts_voices:
                try:
                    import asyncio

                    import edge_tts

                    async def generate_edge():
                        communicate = edge_tts.Communicate(
                            segment_text, edge_tts_voices[tts_lang]
                        )
                        audio_data = io.BytesIO()
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                audio_data.write(chunk["data"])
                        return audio_data.getvalue()

                    audio_bytes = asyncio.run(generate_edge())
                except Exception:
                    pass

            if not audio_bytes:
                gtts_lang = gtts_lang_map.get(tts_lang, "en")
                tts = gTTS(text=segment_text, lang=gtts_lang, slow=False)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes = audio_bytes.getvalue()

            tts_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

            buffer = io.BytesIO()
            tts_audio.export(buffer, format="mp3")
            audio_bytes = buffer.getvalue()

        return {
            "index": segment.get("index", 0),
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "index": segment.get("index", 0),
            "audio_base64": None,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "success": False,
            "error": str(e),
        }


@app.function(image=get_ml_image(), timeout=600)
def remote_subtitle_audio(segments_data: dict, target_lang: str = "en") -> dict:
    """Generate audio from subtitle segments using gTTS or Edge-TTS on Modal - PARALLEL execution"""

    lang_map = {
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ja": "ja",
        "ko": "ko",
        "zh-cn": "zh-cn",
        "ar": "ar",
        "hi": "hi",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "sv": "sv",
        "swe": "sv",
        "te": "te",
        "kn": "kn",
        "swa": "sw",
        "sw": "sw",
        "hy": "hy",
        "nn": "no",
        "nb": "no",
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
    }

    tts_lang = lang_map.get(target_lang.lower(), "en")
    segments = segments_data.get("segments", [])

    # Use Modal's parallel map to generate audio for all segments concurrently
    # This spawns multiple containers to process segments in parallel
    results = list(
        _generate_single_segment_audio.map([(seg, tts_lang) for seg in segments])
    )

    # Separate successful results from errors
    audio_segments = [r for r in results if r.get("success")]
    errors = [
        {"index": r["index"], "error": r["error"]}
        for r in results
        if not r.get("success")
    ]

    return {
        "audio_segments": audio_segments,
        "total_segments": len(segments),
        "successful_segments": len(audio_segments),
        "errors": errors,
    }


@app.function(
    image=get_ml_image(),
    timeout=3600,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_generate_subtitle_audio(
    segments_data: dict,
    target_lang: str = "en",
    output_format: str = "mp3",
    job_id: str = None,
) -> dict:
    """
    Generate the final subtitle-to-audio artifact on Modal and persist it to R2/Volume.
    """
    import base64
    import io
    import logging
    import os
    import tempfile

    from pydub import AudioSegment

    logger = logging.getLogger(__name__)

    try:
        _publish_progress_event(job_id, 60, "Modal subtitle audio generation started", "subtitle_audio")

        raw_result = remote_subtitle_audio.remote(segments_data, target_lang)
        audio_segments = raw_result.get("audio_segments", [])
        if not audio_segments:
            return {
                "success": False,
                "error": "No audio segments were generated",
                "total_segments": raw_result.get("total_segments", 0),
                "successful_segments": 0,
                "errors": raw_result.get("errors", []),
            }

        ordered_segments = sorted(audio_segments, key=lambda x: x.get("index", 0))
        final_audio = AudioSegment.silent(duration=0)

        for index, segment in enumerate(ordered_segments):
            audio_base64 = segment.get("audio_base64")
            if not audio_base64:
                continue
            audio_bytes = base64.b64decode(audio_base64)
            segment_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            final_audio += segment_audio
            if index < len(ordered_segments) - 1:
                final_audio += AudioSegment.silent(duration=500)

        output_format = (output_format or "mp3").lower()
        if output_format not in {"mp3", "wav", "m4a", "ogg"}:
            output_format = "mp3"

        output_filename = f"translated_subtitle_audio_{job_id}.{output_format}" if job_id else f"translated_subtitle_audio.{output_format}"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_path = os.path.join(temp_dir, output_filename)
            final_audio.export(temp_output_path, format=output_format)

            output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
            storage_path, volume_path = _publish_audio_output(
                temp_output_path,
                output_bucket,
                f"jobs/{job_id or 'adhoc'}/outputs/{output_filename}",
                ("subtitle_audio", job_id or "adhoc", output_filename),
                "audio/mpeg" if output_format == "mp3" else f"audio/{output_format}",
            )

        _publish_progress_event(
            job_id,
            95,
            f"Subtitle audio complete ({len(ordered_segments)} segments)",
            "subtitle_audio",
            {"segment_count": len(ordered_segments), "format": output_format},
        )

        return {
            "success": True,
            "download_url": storage_path,
            "output_storage_path": storage_path,
            "output_volume_path": volume_path,
            "output_format": output_format,
            "total_segments": raw_result.get("total_segments", len(ordered_segments)),
            "successful_segments": raw_result.get("successful_segments", len(ordered_segments)),
            "errors": raw_result.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Remote subtitle audio generation failed: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_preprocess_audio(
    audio_source, enable_denoising: bool = True, enable_normalization: bool = True, job_id: str = None
) -> dict:
    """Preprocess audio with FFmpeg filters on Modal (CPU intensive, fast remote execution)"""
    import os
    import subprocess
    import tempfile

    # Save input audio to temp file
    input_path, cleanup_input = _resolve_input_path(audio_source, suffix=".wav")
    _publish_progress_event(job_id, 10, "Modal audio preprocessing started", "preprocess_audio")

    # Create output path
    output_path = input_path.replace(".wav", "_preprocessed.wav")

    try:
        original_source = audio_source
        # Build FFmpeg filter chain
        filters = []

        if enable_denoising:
            # Apply audio noise reduction using anlmdn filter
            filters.append("anlmdn")

        if enable_normalization:
            # Apply gentle normalization to avoid artifacts
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        if not filters:
            # No filters needed, return original
            if isinstance(original_source, str) and _storage_scheme(original_source):
                storage_path = original_source
            elif isinstance(original_source, str) and original_source.startswith(_PIPELINE_MOUNT):
                storage_path = original_source
            else:
                output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
                storage_path, volume_path = _publish_audio_output(
                    input_path,
                    output_bucket,
                    f"modal/preprocessed/{os.path.basename(output_path)}",
                    ("preprocessed", os.path.basename(output_path)),
                    "audio/wav",
                )
                _publish_progress_event(job_id, 20, "Modal preprocessing skipped (no filters)", "preprocess_audio")
                return {
                    "success": True,
                    "audio_storage_path": storage_path,
                    "audio_volume_path": volume_path,
                    "filters_applied": [],
                    "message": "No filters applied",
                }

            volume_path = _write_pipeline_file(
                input_path, "preprocessed", os.path.basename(output_path)
            )
            _publish_progress_event(job_id, 20, "Modal preprocessing skipped (no filters)", "preprocess_audio")
            return {
                "success": True,
                "audio_storage_path": storage_path,
                "audio_volume_path": volume_path,
                "filters_applied": [],
                "message": "No filters applied",
            }

        filter_string = ",".join(filters)

        # Execute FFmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-filter:a",
            filter_string,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # Whisper expects 16kHz
            "-ac",
            "1",  # Mono
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return {
                "success": False,
                "audio_storage_path": None,
                "error": f"FFmpeg failed: {result.stderr}",
                "filters_applied": filters,
            }

        # Read preprocessed audio
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
            audio_data, volume_path = _publish_audio_output(
                output_path,
                output_bucket,
                f"modal/preprocessed/{os.path.basename(output_path)}",
                ("preprocessed", os.path.basename(output_path)),
                "audio/wav",
            )
            _publish_progress_event(job_id, 20, "Modal audio preprocessing complete", "preprocess_audio")
            return {
                "success": True,
                "audio_storage_path": audio_data,
                "audio_volume_path": volume_path,
                "filters_applied": filters,
                    "message": f"Applied filters: {filter_string}",
                }
        else:
            return {
                "success": False,
                "audio_storage_path": None,
                "audio_volume_path": None,
                "error": "Preprocessed file is empty",
                "filters_applied": filters,
            }

    except Exception as e:
        return {
            "success": False,
            "audio_storage_path": None,
            "audio_volume_path": None,
            "error": str(e),
            "filters_applied": filters if "filters" in locals() else [],
        }

    finally:
        # Cleanup temp files
        if cleanup_input and os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


@app.function(gpu="T4", image=get_ml_image(), timeout=3600, retries=2, secrets=_r2_secret_list())
def remote_process_video_chunk(chunk_data: dict) -> dict:
    """
    Process a single video chunk: transcribe, translate, and synthesize speech
    This runs on Modal T4 GPU for maximum speed
    """
    import io
    import logging
    import os
    import subprocess
    import tempfile

    from deep_translator import GoogleTranslator
    from faster_whisper import WhisperModel
    from gtts import gTTS
    from pydub import AudioSegment

    logger = logging.getLogger(__name__)

    try:
        cleanup_needed = False
        # Extract data from chunk
        audio_bytes = chunk_data.get("audio_bytes")
        chunk_id = chunk_data.get("chunk_id", 0)
        job_id = chunk_data.get("job_id")
        total_chunks = max(int(chunk_data.get("total_chunks", 1) or 1), 1)
        source_lang = chunk_data.get("source_lang", "en")
        target_lang = chunk_data.get("target_lang", "en")
        model_size = chunk_data.get("model_size", "base")

        if not audio_bytes:
            return {
                "success": False,
                "error": "No audio data provided",
                "chunk_id": chunk_id,
            }

        audio_path, cleanup_needed = _resolve_input_path(audio_bytes, suffix=".wav")
        input_size = (
            len(audio_bytes)
            if isinstance(audio_bytes, (bytes, bytearray))
            else os.path.getsize(audio_path)
            if os.path.exists(audio_path)
            else 0
        )
        logger.info(f"[CHUNK_{chunk_id}] Starting processing ({input_size} bytes)")
        _publish_progress_event(
            job_id,
            70 + int((min(chunk_id + 1, total_chunks) / total_chunks) * 20),
            f"Processing chunk {chunk_id + 1}/{total_chunks}",
            "chunk_processing",
            {"chunk_id": chunk_id, "total_chunks": total_chunks},
        )

        try:
            # Step 1: Transcribe with Faster-Whisper on GPU
            logger.info(f"[CHUNK_{chunk_id}] Transcribing with Faster-Whisper...")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            segments_gen, info = model.transcribe(audio_path, beam_size=5)

            segments = []
            full_text = ""
            for segment in segments_gen:
                segments.append(
                    {
                        "id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                )
                full_text += segment.text + " "
            full_text = full_text.strip()
            detected_language = info.language

            if not full_text:
                # No speech detected, return silent audio
                chunk_duration = chunk_data.get("duration_ms", 5000)
                silent_audio = AudioSegment.silent(duration=chunk_duration)
                buffer = io.BytesIO()
                silent_audio.export(buffer, format="wav")
                _publish_progress_event(
                    job_id,
                    70 + int((min(chunk_id + 1, total_chunks) / total_chunks) * 20),
                    f"Chunk {chunk_id + 1}/{total_chunks} complete (silence)",
                    "chunk_processing",
                    {"chunk_id": chunk_id, "total_chunks": total_chunks},
                )

                logger.info(f"[CHUNK_{chunk_id}] No speech detected, returning silence")
                return {
                    "success": True,
                    "chunk_id": chunk_id,
                    "audio_bytes": buffer.getvalue(),
                    "text": "",
                    "translated_text": "",
                    "language": detected_language,
                    "segments": [],
                }

            logger.info(
                f"[CHUNK_{chunk_id}] Transcribed: {len(full_text)} chars, detected lang: {detected_language}"
            )

            # Step 2: Translate if needed
            translated_segments = []
            if source_lang != target_lang and full_text:
                lang_map = {
                    "en": "en",
                    "es": "es",
                    "fr": "fr",
                    "de": "de",
                    "it": "it",
                    "pt": "pt",
                    "ru": "ru",
                    "ja": "ja",
                    "ko": "ko",
                    "zh-cn": "zh-CN",
                    "zh": "zh-CN",
                    "ar": "ar",
                    "hi": "hi",
                    "nl": "nl",
                    "pl": "pl",
                    "tr": "tr",
                    "sv": "sv",
                    "swe": "sv",
                    "te": "te",
                    "kn": "kn",
                    "swa": "sw",
                    "sw": "sw",
                    "hy": "hy",
                    "nn": "no",
                    "nb": "no",
                }
                target_code = lang_map.get(target_lang.lower(), "en")

                translator = GoogleTranslator(source="auto", target=target_code)

                for seg in segments:
                    text = seg.get("text", "").strip()
                    if text:
                        try:
                            translated = translator.translate(text)
                            seg["translated_text"] = translated
                            translated_segments.append(seg)
                        except:
                            seg["translated_text"] = text
                            translated_segments.append(seg)
                    else:
                        seg["translated_text"] = text
                        translated_segments.append(seg)

                translated_text = " ".join(
                    [
                        s.get("translated_text", s.get("text", ""))
                        for s in translated_segments
                    ]
                )
                logger.info(
                    f"[CHUNK_{chunk_id}] Translated: {len(translated_text)} chars to {target_lang}"
                )
            else:
                translated_segments = segments
                translated_text = full_text

            # Step 3: Synthesize speech with gTTS or Edge-TTS
            lang_map_tts = {
                "en": "en",
                "es": "es",
                "fr": "fr",
                "de": "de",
                "it": "it",
                "pt": "pt",
                "ru": "ru",
                "ja": "ja",
                "ko": "ko",
                "zh-cn": "zh-cn",
                "zh": "zh-cn",
                "ar": "ar",
                "hi": "hi",
                "nl": "nl",
                "pl": "pl",
                "tr": "tr",
                "sv": "sv",
                "swe": "sv",
                "te": "te",
                "kn": "kn",
                "swa": "sw",
                "sw": "sw",
                "hy": "hy",
                "nn": "no",
                "nb": "no",
            }

            # Male voices as default for edge_tts
            edge_tts_voices = {
                "en": "en-US-GuyNeural",
                "es": "es-MX-DarioNeural",
                "fr": "fr-FR-HenriNeural",
                "de": "de-DE-ConradNeural",
                "it": "it-IT-DiegoNeural",
                "pt": "pt-PT-AntonioNeural",
                "ru": "ru-RU-DmitryNeural",
                "ja": "ja-JP-KeitaNeural",
                "ko": "ko-KR-InhoNeural",
                "zh-cn": "zh-CN-YunxiNeural",
                "ar": "ar-SA-HamedNeural",
                "hi": "hi-IN-MadhurNeural",
                "nl": "nl-NL-MaartenNeural",
                "pl": "pl-PL-MarekNeural",
                "tr": "tr-TR-AhmetNeural",
                "sv": "sv-SE-MattiasNeural",
                "te": "te-IN-ChaitanyaNeural",
                "kn": "kn-IN-GaneshNeural",
                "swa": "sw-KE-RafikiNeural",
                "sw": "sw-TZ-DaudiNeural",
                "hy": "hy-AM-ArturNeural",
                "nn": "nb-NO-FelixNeural",
                "nb": "nb-NO-FelixNeural",
                "swe": "sv-SE-MattiasNeural",
            }

            tts_lang = lang_map_tts.get(target_lang.lower(), "en")

            logger.info(f"[CHUNK_{chunk_id}] Generating voiceover ({tts_lang})...")

            if translated_text:
                audio_buffer = None
                audio_format = "mp3"

                if not audio_buffer and target_lang.lower() in edge_tts_voices:
                    try:
                        import asyncio

                        import edge_tts

                        # Use provided voice_id if available, otherwise fallback to default
                        voice_id = chunk_data.get("voice_id")
                        if not voice_id:
                            voice_id = edge_tts_voices[target_lang.lower()]

                        logger.info(f"[CHUNK_{chunk_id}] Using voice: {voice_id}")

                        async def generate_edge():
                            communicate = edge_tts.Communicate(
                                translated_text, voice_id
                            )
                            audio_data = io.BytesIO()
                            async for chunk in communicate.stream():
                                if chunk["type"] == "audio":
                                    audio_data.write(chunk["data"])
                            return audio_data.getvalue()

                        audio_buffer = io.BytesIO(asyncio.run(generate_edge()))

                        # --- SILENCE STRIPPING ---
                        try:
                            from pydub import AudioSegment
                            from pydub.silence import detect_leading_silence

                            def trim_silence(
                                sound, silence_threshold=-50.0, chunk_size=10
                            ):
                                iterate_start = 0
                                while (
                                    detect_leading_silence(
                                        sound[iterate_start:],
                                        silence_threshold,
                                        chunk_size,
                                    )
                                    > 0
                                ):
                                    iterate_start += chunk_size
                                    if iterate_start >= len(sound):
                                        break

                                iterate_end = 0
                                while (
                                    detect_leading_silence(
                                        sound.reverse()[iterate_end:],
                                        silence_threshold,
                                        chunk_size,
                                    )
                                    > 0
                                ):
                                    iterate_end += chunk_size
                                    if iterate_end >= len(sound):
                                        break

                                return sound[iterate_start : len(sound) - iterate_end]

                            # Load generated audio and trim padding silence
                            full_audio = AudioSegment.from_file(
                                audio_buffer, format="mp3"
                            )
                            trimmed_audio = trim_silence(full_audio)

                            # Export back to buffer
                            audio_buffer = io.BytesIO()
                            trimmed_audio.export(audio_buffer, format="mp3")
                            audio_buffer.seek(0)
                        except Exception as silence_err:
                            logger.warning(f"Silence stripping failed: {silence_err}")
                            audio_buffer.seek(0)
                    except Exception as e:
                        logger.warning(f"Edge-TTS failed, falling back to gTTS: {e}")

                if not audio_buffer:
                    tts = gTTS(text=translated_text, lang=tts_lang, slow=False)
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                # Load the audio
                tts_audio = AudioSegment.from_file(audio_buffer, format=audio_format)
                target_duration = chunk_data.get("duration_ms", len(tts_audio))
                tts_duration = len(tts_audio)

                # Only adjust if significantly different (more than 25% difference - more lenient)
                if tts_duration > 0 and target_duration > 0:
                    duration_diff_ratio = (
                        abs(tts_duration - target_duration) / target_duration
                    )

                    if duration_diff_ratio > 0.10:  # Tightened threshold (was 25%)
                        # Calculate speed factor: if TTS is longer than target, we need to speed up
                        speed_factor = tts_duration / target_duration

                        # Clamp to safe bounds (0.85x to 1.30x speed) - maintains high quality
                        speed_factor = max(0.85, min(1.30, speed_factor))

                        logger.info(
                            f"[CHUNK_{chunk_id}] Speed-matching: {tts_duration:.0f}ms -> {target_duration:.0f}ms (factor: {speed_factor:.2f}x)"
                        )

                        try:
                            # Export to temporary file for processing
                            import tempfile

                            with tempfile.NamedTemporaryFile(
                                suffix=".wav", delete=False
                            ) as temp_in:
                                tts_audio.export(temp_in.name, format="wav")
                                temp_input_path = temp_in.name

                            temp_output_path = temp_input_path.replace(
                                ".wav", "_stretched.wav"
                            )

                            # Use ffmpeg atempo for time-stretching
                            # atempo>1.0 speeds up (shorter duration), atempo<1.0 slows down (longer duration)
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                temp_input_path,
                                "-filter:a",
                                f"atempo={speed_factor:.3f}",
                                "-ar",
                                "44100",
                                "-ac",
                                "1",
                                temp_output_path,
                            ]

                            result = subprocess.run(
                                cmd, capture_output=True, timeout=30
                            )

                            if result.returncode == 0 and os.path.exists(
                                temp_output_path
                            ):
                                tts_audio = AudioSegment.from_file(temp_output_path)
                                actual_duration = len(tts_audio)
                                logger.info(
                                    f"Speed adjustment: {tts_duration}ms -> {actual_duration}ms (target: {target_duration}ms, factor: {speed_factor:.3f}x)"
                                )
                                # Clean up temp files
                                os.remove(temp_input_path)
                                os.remove(temp_output_path)
                        except Exception as stretch_error:
                            # If time-stretching fails, just use original TTS audio
                            logger.warning(
                                f"Time-stretching failed, using original: {stretch_error}"
                            )

                # Export final audio
                output_buffer = io.BytesIO()
                tts_audio.export(output_buffer, format="wav")
                output_audio_bytes = output_buffer.getvalue()
            else:
                # Silent audio
                chunk_duration = chunk_data.get("duration_ms", 5000)
                silent_audio = AudioSegment.silent(duration=chunk_duration)
                buffer = io.BytesIO()
                silent_audio.export(buffer, format="wav")
                output_audio_bytes = buffer.getvalue()

            output_size = len(output_audio_bytes)
            logger.info(f"[CHUNK_{chunk_id}] Complete: {output_size} bytes")
            _publish_progress_event(
                job_id,
                70 + int((min(chunk_id + 1, total_chunks) / total_chunks) * 20),
                f"Chunk {chunk_id + 1}/{total_chunks} complete",
                "chunk_processing",
                {"chunk_id": chunk_id, "total_chunks": total_chunks},
            )

            return {
                "success": True,
                "chunk_id": chunk_id,
                "audio_bytes": output_audio_bytes,
                "text": full_text,
                "translated_text": translated_text,
                "language": detected_language,
                "segments": translated_segments,
            }

        finally:
            # Cleanup
            if cleanup_needed and os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        logger.error(f"[CHUNK_{chunk_data.get('chunk_id', 0)}] Error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "chunk_id": chunk_data.get("chunk_id", 0),
        }


@app.function(gpu="T4", image=get_ml_image(), timeout=86400, retries=1, secrets=_r2_secret_list())
def remote_process_video_chunks_batch(chunks_data: list) -> dict:
    """
    Process multiple video chunks in parallel using Modal's map
    Each chunk gets its own GPU container for maximum parallelism
    Returns granular progress updates for UI
    """
    import logging
    import time

    logger = logging.getLogger(__name__)

    total_chunks = len(chunks_data)
    start_time = time.time()
    job_id = chunks_data[0].get("job_id") if chunks_data else None

    # Log batch start with timestamp for progress tracking
    logger.info(f"[BATCH_START] Processing {total_chunks} chunks")
    _publish_progress_event(job_id, 70, f"Processing {total_chunks} chunks in Modal", "chunk_batch", {"total_chunks": total_chunks})

    try:
        # Process chunks in parallel via Modal's map
        results = list(remote_process_video_chunk.map(chunks_data))

        # Count results
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        # Calculate progress percentage
        success_count = len(successful)
        progress_percent = (
            int((success_count / total_chunks) * 100) if total_chunks > 0 else 100
        )

        elapsed_time = time.time() - start_time
        chunks_per_second = success_count / elapsed_time if elapsed_time > 0 else 0

        logger.info(
            f"[BATCH_PROGRESS] {success_count}/{total_chunks} chunks completed ({progress_percent}%)"
        )
        logger.info(
            f"[BATCH_COMPLETE] Processing rate: {chunks_per_second:.2f} chunks/sec"
        )
        _publish_progress_event(
            job_id,
            90,
            f"Modal chunk batch complete ({success_count}/{total_chunks})",
            "chunk_batch",
            {"successful_count": len(successful), "failed_count": len(failed), "total_chunks": total_chunks},
        )

        return {
            "success": True,
            "results": results,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "total_chunks": total_chunks,
            "progress_percent": progress_percent,
            "elapsed_seconds": elapsed_time,
            "chunks_per_second": chunks_per_second,
        }

    except Exception as e:
        logger.error(f"[BATCH_ERROR] {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "successful_count": 0,
            "failed_count": total_chunks,
            "total_chunks": total_chunks,
            "progress_percent": 0,
            "elapsed_seconds": time.time() - start_time,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": []}


@app.function(image=get_ml_image(), timeout=86400, secrets=_r2_secret_list(), volumes={"/pipeline": _PIPELINE_VOLUME})
def remote_merge_audio_chunks_absolute(
    chunk_manifest: list,
    total_duration_ms: int,
    output_filename: str = "merged.wav",
    job_id: str = None,
) -> dict:
    """
    Merge translated audio chunks on Modal so the backend never has to hold the full
    mixed waveform in RAM.
    """
    import logging
    import os
    import tempfile

    from pydub import AudioSegment

    logger = logging.getLogger(__name__)
    total_chunks = len(chunk_manifest)

    try:
        _publish_progress_event(
            job_id,
            80,
            f"Modal merge started for {total_chunks} chunks",
            "audio_merge",
            {"total_chunks": total_chunks},
        )

        if not chunk_manifest:
            return {"success": False, "error": "No chunk manifest provided"}

        ordered_chunks = sorted(
            chunk_manifest,
            key=lambda item: (
                int(item.get("start_ms", 0)),
                int(item.get("chunk_id", 0)),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            final_audio = AudioSegment.silent(duration=int(total_duration_ms + 5000))

            for item in ordered_chunks:
                audio_source = (
                    item.get("audio_storage_path")
                    or item.get("audio_source")
                    or item.get("path")
                )
                if not audio_source:
                    continue

                chunk_id = item.get("chunk_id", 0)
                start_ms = int(item.get("start_ms", 0))
                original_duration = int(
                    item.get("duration_ms")
                    or (item.get("end_ms", 0) - item.get("start_ms", 0))
                    or 0
                )

                local_audio_path, cleanup_needed = _resolve_input_path(
                    audio_source, suffix=".wav"
                )
                try:
                    chunk_audio = AudioSegment.from_file(local_audio_path)
                    translated_duration = len(chunk_audio)

                    if original_duration > 0 and translated_duration > original_duration:
                        speedup_ratio = translated_duration / original_duration
                        logger.info(
                            f"[MODAL MERGE] Chunk {chunk_id} is {translated_duration}ms "
                            f"(target {original_duration}ms). Speeding up by {speedup_ratio:.4f}x"
                        )
                        try:
                            if speedup_ratio >= 1.0:
                                chunk_audio = chunk_audio.speedup(
                                    playback_speed=speedup_ratio,
                                    chunk_size=150,
                                    crossfade=25,
                                )
                        except Exception as speed_err:
                            logger.warning(
                                f"[MODAL MERGE] Speedup failed for chunk {chunk_id}: {speed_err}"
                            )

                    chunk_audio = chunk_audio.fade_in(10).fade_out(10)
                    final_audio = final_audio.overlay(chunk_audio, position=start_ms)
                finally:
                    if cleanup_needed and os.path.exists(local_audio_path):
                        os.remove(local_audio_path)

            output_path = os.path.join(temp_dir, output_filename)
            final_audio[: int(total_duration_ms)].export(output_path, format="wav")

            output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
            output_remote_path = f"jobs/{job_id or 'adhoc'}/modal-merged/{output_filename}"
            storage_path, volume_path = _publish_audio_output(
                output_path,
                output_bucket,
                output_remote_path,
                ("merged", job_id or "adhoc", output_filename),
                "audio/wav",
            )

            _publish_progress_event(
                job_id,
                90,
                "Modal audio merge complete",
                "audio_merge",
                {"merged_chunks": total_chunks},
            )

            return {
                "success": True,
                "audio_storage_path": storage_path,
                "audio_volume_path": volume_path,
                "merged_duration_ms": len(final_audio[: int(total_duration_ms)]),
                "total_chunks": total_chunks,
            }

    except Exception as e:
        logger.error(f"[MODAL MERGE] Failed: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    image=get_ml_image(),
    timeout=86400,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_finalize_video_translation(
    video_source: str,
    chunk_manifest: list,
    total_duration_ms: int,
    output_filename: str,
    job_id: str = None,
    instrumental_source: str = None,
) -> dict:
    """
    Merge translated chunks, mux them into the original video, and upload the
    final MP4 to object storage.
    """
    import logging
    import os
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from pydub import AudioSegment

    logger = logging.getLogger(__name__)

    try:
        _publish_progress_event(
            job_id,
            82,
            "Modal finalization started",
            "video_finalize",
            {"total_chunks": len(chunk_manifest)},
        )

        if not chunk_manifest:
            return {"success": False, "error": "No chunk manifest provided"}

        video_path, video_cleanup = _resolve_input_path(video_source, suffix=".mp4")
        instrumental_path = None
        instrumental_cleanup = False
        if instrumental_source:
            instrumental_path, instrumental_cleanup = _resolve_input_path(
                instrumental_source, suffix=".wav"
            )

        try:
            ordered_chunks = sorted(
                chunk_manifest,
                key=lambda item: (
                    int(item.get("start_ms", 0)),
                    int(item.get("chunk_id", 0)),
                ),
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                merged_audio_path = os.path.join(temp_dir, "merged.wav")

                # --- Optimization 1: Download all chunks in parallel ---
                def _download_chunk(item):
                    audio_source = item.get("audio_storage_path") or item.get("path")
                    if not audio_source:
                        return None
                    local_path, cleanup = _resolve_input_path(audio_source, suffix=".wav")
                    return {**item, "_local_path": local_path, "_cleanup": cleanup}

                with ThreadPoolExecutor(max_workers=8) as dl_pool:
                    download_futures = {
                        dl_pool.submit(_download_chunk, item): item
                        for item in ordered_chunks
                    }
                    downloaded_chunks = []
                    for future in as_completed(download_futures):
                        result = future.result()
                        if result is not None:
                            downloaded_chunks.append(result)

                # Re-sort after parallel download (as_completed returns in completion order)
                downloaded_chunks.sort(
                    key=lambda item: (
                        int(item.get("start_ms", 0)),
                        int(item.get("chunk_id", 0)),
                    )
                )

                final_audio = AudioSegment.silent(duration=int(total_duration_ms + 5000))

                for item in downloaded_chunks:
                    local_audio_path = item["_local_path"]
                    cleanup_needed = item["_cleanup"]
                    chunk_id = item.get("chunk_id", 0)
                    start_ms = int(item.get("start_ms", 0))
                    original_duration = int(
                        item.get("duration_ms")
                        or (item.get("end_ms", 0) - item.get("start_ms", 0))
                        or 0
                    )

                    try:
                        chunk_audio = AudioSegment.from_file(local_audio_path)

                        # --- Optimization 2: Skip speedup if already duration-fitted ---
                        if not item.get("duration_fitted"):
                            translated_duration = len(chunk_audio)
                            if original_duration > 0 and translated_duration > original_duration:
                                speedup_ratio = translated_duration / original_duration
                                try:
                                    if speedup_ratio >= 1.0:
                                        chunk_audio = chunk_audio.speedup(
                                            playback_speed=speedup_ratio,
                                            chunk_size=150,
                                            crossfade=25,
                                        )
                                except Exception as speed_err:
                                    logger.warning(
                                        f"[MODAL FINALIZE] Speedup failed for chunk {chunk_id}: {speed_err}"
                                    )

                        chunk_audio = chunk_audio.fade_in(10).fade_out(10)
                        final_audio = final_audio.overlay(chunk_audio, position=start_ms)
                    finally:
                        if cleanup_needed and os.path.exists(local_audio_path):
                            os.remove(local_audio_path)

                # --- Optimization 3: Keep merged audio in memory, avoid repeated reads ---
                merged_audio = final_audio[: int(total_duration_ms)]
                merged_duration_ms = len(merged_audio)

                video_duration_ms = 0.0
                try:
                    probe = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            video_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if probe.returncode == 0 and probe.stdout.strip():
                        video_duration_ms = float(probe.stdout.strip()) * 1000
                except Exception:
                    video_duration_ms = 0.0

                if video_duration_ms > 0 and merged_duration_ms < video_duration_ms:
                    merged_audio = merged_audio + AudioSegment.silent(
                        duration=int(video_duration_ms - merged_duration_ms)
                    )
                    merged_duration_ms = len(merged_audio)

                merged_audio.export(merged_audio_path, format="wav")
                del merged_audio
                del final_audio

                input_audio = merged_audio_path
                if instrumental_path and os.path.exists(instrumental_path):
                    mixed_audio = os.path.join(temp_dir, "mixed_final.wav")
                    mix_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_audio,
                        "-i",
                        instrumental_path,
                        "-filter_complex",
                        "amix=inputs=2:duration=longest:dropout_transition=2",
                        "-ac",
                        "2",
                        "-ar",
                        "44100",
                        mixed_audio,
                    ]
                    result = subprocess.run(mix_cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0 or not os.path.exists(mixed_audio):
                        raise RuntimeError(f"Audio mix failed: {result.stderr}")
                    input_audio = mixed_audio

                output_path = os.path.join(temp_dir, output_filename)
                mux_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-i",
                    input_audio,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-threads",
                    "0",
                    "-loglevel",
                    "error",
                    output_path,
                ]
                result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0 or not os.path.exists(output_path):
                    raise RuntimeError(f"Video mux failed: {result.stderr}")

                # --- Optimization 4: Upload to R2 and copy to volume concurrently ---
                output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
                remote_path = f"jobs/{job_id or 'adhoc'}/final/{output_filename}"

                storage_path = None
                volume_path = None

                def _do_upload():
                    return _upload_storage_file(
                        output_path, output_bucket, remote_path, "video/mp4"
                    )

                def _do_volume_copy():
                    return _write_pipeline_file(
                        output_path, "final", job_id or "adhoc", output_filename
                    )

                with ThreadPoolExecutor(max_workers=2) as pub_pool:
                    upload_future = pub_pool.submit(_do_upload)
                    volume_future = pub_pool.submit(_do_volume_copy)
                    storage_path = upload_future.result()
                    volume_path = volume_future.result()

                _publish_progress_event(
                    job_id,
                    95,
                    "Modal video finalization complete",
                    "video_finalize",
                    {"output_filename": output_filename},
                )

                return {
                    "success": True,
                    "output_storage_path": storage_path,
                    "output_volume_path": volume_path,
                    "output_filename": output_filename,
                    "merged_duration_ms": merged_duration_ms,
                }
        finally:
            if video_cleanup and os.path.exists(video_path):
                os.remove(video_path)
            if instrumental_source and instrumental_cleanup and instrumental_path and os.path.exists(instrumental_path):
                os.remove(instrumental_path)
    except Exception as e:
        logger.error(f"[MODAL FINALIZE] Failed: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_extract_audio(video_source, target_sample_rate: int = 16000, job_id: str = None) -> dict:
    """
    Extract audio from video file using FFmpeg on Modal GPU
    Returns audio bytes in WAV format optimized for Whisper
    """
    import os
    import subprocess
    import tempfile

    try:
        _publish_progress_event(job_id, 12, "Modal audio extraction started", "extract_audio")
        # Save video to temp file
        if isinstance(video_source, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(video_source)
                video_path = temp_video.name
        elif isinstance(video_source, str) and video_source.startswith(_PIPELINE_MOUNT):
            video_path = video_source
        elif _storage_scheme(video_source):
            video_path = _download_storage_object(video_source, suffix=".mp4")
        else:
            video_path = video_source

        audio_path = video_path.replace(".mp4", "_audio.wav")

        try:
            # Extract audio with FFmpeg
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(target_sample_rate),
                "-ac",
                "1",  # Mono
                "-y",
                audio_path,
            ]

            # Long videos can take a while to remux/extract even when the wrapper has
            # plenty of headroom, so give ffmpeg a much larger internal timeout.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"FFmpeg extraction failed: {result.stderr}",
                }

            # Read extracted audio
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
                audio_data, volume_path = _publish_audio_output(
                    audio_path,
                    output_bucket,
                    f"modal/extracted/{os.path.basename(audio_path)}",
                    ("extracted", os.path.basename(audio_path)),
                    "audio/wav",
                )
                _publish_progress_event(job_id, 18, "Modal audio extraction complete", "extract_audio")

                return {
                    "success": True,
                    "audio_storage_path": audio_data,
                    "audio_volume_path": volume_path,
                    "sample_rate": target_sample_rate,
                }
            else:
                return {"success": False, "error": "Extracted audio file is empty"}

        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_pyannote4_image(),
    secrets=_r2_secret_list() + [modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_speaker_diarization(
    audio_source,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    job_id: str = None,
) -> dict:
    """
    Speaker diarization using pyannote.audio community-1 on Modal T4 GPU.

    Requirements:
    - Accept pyannote/segmentation-3.0 user conditions at hf.co
    - Accept pyannote/speaker-diarization-3.1 user conditions at hf.co

    Args:
        audio_bytes: WAV audio data (mono, 16kHz recommended)
        num_speakers: Fixed number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)

    Returns:
        Dictionary with diarization results in RTTM format and segment data
    """
    import os
    import tempfile

    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return {"success": False, "error": "HF_TOKEN not available"}

        audio_path, cleanup_needed = _resolve_input_path(audio_source, suffix=".wav")
        _publish_progress_event(job_id, 45, "Speaker diarization started", "speaker_diarization")

        try:
            # Initialize pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=hf_token
            )

            # Move to GPU
            pipeline.to(torch.device("cuda"))

            # Build kwargs
            kwargs = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            # Run diarization
            diarization = pipeline(audio_path, **kwargs)

            # Convert to RTTM format (pyannote 4.0 returns DiarizeOutput wrapper)
            # Need to access .speaker_diarization to get the Annotation object
            annotation = diarization.speaker_diarization
            import io

            rttm_buffer = io.StringIO()
            annotation.write_rttm(rttm_buffer)
            rttm_text = rttm_buffer.getvalue()

            # Extract segment data for easier processing
            segments = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                )

            # Get unique speakers
            speakers = sorted(set(seg["speaker"] for seg in segments))
            _publish_progress_event(
                job_id,
                55,
                f"Speaker diarization complete ({len(speakers)} speakers)",
                "speaker_diarization",
                {"speakers": speakers},
            )

            return {
                "success": True,
                "rttm": rttm_text,
                "segments": segments,
                "num_speakers": len(speakers),
                "speakers": speakers,
            }

        finally:
            # Cleanup
            if cleanup_needed and os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_whisperx_transcribe_diarize_from_path(
    audio_path: str, model_size: str = "medium", language: str = None
) -> dict:
    import os

    import torch
    import whisperx

    print(f"[WhisperX] Loading WhisperX model '{model_size}' on CUDA (float16)...")
    model = whisperx.load_model(model_size, device="cuda", compute_type="float16")
    print("[WhisperX] Model loaded successfully.")

    import torchaudio

    print("[WhisperX] Loading audio with torchaudio...")
    waveform, sr = torchaudio.load(audio_path)
    print(f"[WhisperX] Audio loaded - shape={tuple(waveform.shape)}, sample_rate={sr}")
    if waveform.ndim > 1 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("[WhisperX] Converted multi-channel to mono.")
    if sr != 16000:
        print(f"[WhisperX] Resampling from {sr}Hz to 16000Hz...")
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        print("[WhisperX] Resampling done.")
    audio = waveform.squeeze(0).cpu().numpy()
    print(
        f"[WhisperX] Audio array ready - length={len(audio)} samples ({len(audio) / 16000:.1f}s)"
    )

    lang_code = language if language else None
    print(f"[WhisperX] Starting transcription (language={lang_code!r})...")
    result = model.transcribe(audio, language=lang_code)

    detected_lang = result.get("language", "en")
    num_raw_segments = len(result.get("segments", []))
    raw_text_preview = " ".join(s.get("text", "") for s in result.get("segments", []))[:300]
    print(
        f"[WhisperX] Transcription done - detected_language={detected_lang!r}, raw_segments={num_raw_segments}"
    )
    print(f"[WhisperX] Transcribed text preview: {raw_text_preview!r}")

    print(f"[WhisperX] Loading alignment model for language '{detected_lang}'...")
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_lang, device="cuda"
    )
    print("[WhisperX] Alignment model loaded. Running word-level alignment...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda")
    print("[WhisperX] Alignment complete.")

    from whisperx.diarize import DiarizationPipeline

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        return {"success": False, "error": "HF_TOKEN not available"}

    print("[WhisperX] Initializing DiarizationPipeline...")
    diarize_model = DiarizationPipeline(token=hf_token, device="cuda")
    print("[WhisperX] DiarizationPipeline ready. Running diarization...")

    diarize_segments = diarize_model(audio)
    print("[WhisperX] Diarization complete.")

    print("[WhisperX] Assigning speakers to words/segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print("[WhisperX] Speaker assignment done.")

    segments = []
    all_speakers = set()

    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        all_speakers.add(speaker)

        segments.append(
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "speaker": speaker,
            }
        )

    words = []
    for seg in result.get("segments", []):
        for word_data in seg.get("words", []):
            words.append(
                {
                    "start": word_data.get("start", 0),
                    "end": word_data.get("end", 0),
                    "word": word_data.get("word", ""),
                    "speaker": word_data.get("speaker", "UNKNOWN"),
                }
            )

    full_text = " ".join([seg["text"] for seg in segments])
    speakers = sorted(list(all_speakers))

    print(
        f"[WhisperX] SUCCESS - segments={len(segments)}, words={len(words)}, speakers={speakers}, text_length={len(full_text)}"
    )
    print(
        f"[WhisperX] Full transcription: {full_text[:500]!r}{'...' if len(full_text) > 500 else ''}"
    )
    for seg in segments:
        print(
            f"[WhisperX]   [{seg['speaker']}] {seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text']!r}"
        )

    return {
        "success": True,
        "text": full_text,
        "segments": segments,
        "words": words,
        "language": detected_lang,
        "num_speakers": len(speakers),
        "speakers": speakers,
    }


@app.function(
    gpu="T4",
    image=get_whisperx_image(),
    secrets=_r2_secret_list() + [modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_whisperx_transcribe_diarize(
    audio_bytes: bytes, model_size: str = "medium", language: str = None, job_id: str = None
) -> dict:
    """
    Unified transcription + speaker diarization using WhisperX.

    WhisperX provides:
    - Word-level timestamps
    - Speaker diarization (who speaks when)
    - Handles overlapping speech better than separate Whisper + pyannote

    This eliminates timestamp alignment issues between separate transcription and diarization.

    Args:
        audio_bytes: WAV audio data (mono, 16kHz recommended)
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (auto-detect if None)

    Returns:
        Dictionary with:
        - text: Full transcription
        - segments: [{start, end, text, speaker}]
        - words: [{start, end, word, speaker}]
        - language: Detected language
        - num_speakers: Number of speakers detected
        - speakers: List of speaker IDs
    """
    import os

    import torch
    import whisperx

    try:
        audio_path, cleanup_needed = _resolve_input_path(audio_bytes, suffix=".wav")
        _publish_progress_event(job_id, 40, "WhisperX transcription started", "whisperx")
        print(
            f"[WhisperX] Starting — input={audio_path!r}, model_size={model_size!r}, language={language!r}"
        )
        print(f"[WhisperX] Audio available at: {audio_path}")

        try:
            # Load WhisperX model
            print(f"[WhisperX] Loading WhisperX model '{model_size}' on CUDA (float16)...")
            model = whisperx.load_model(
                model_size, device="cuda", compute_type="float16"
            )
            print(f"[WhisperX] Model loaded successfully.")

            # Load audio manually to bypass WhisperX AudioDecoder
            import torchaudio

            print(f"[WhisperX] Loading audio with torchaudio...")
            waveform, sr = torchaudio.load(audio_path)
            print(f"[WhisperX] Audio loaded — shape={tuple(waveform.shape)}, sample_rate={sr}")
            if waveform.ndim > 1 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print(f"[WhisperX] Converted multi-channel to mono.")
            if sr != 16000:
                print(f"[WhisperX] Resampling from {sr}Hz to 16000Hz...")
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                print(f"[WhisperX] Resampling done.")
            audio = waveform.squeeze(0).cpu().numpy()
            print(f"[WhisperX] Audio array ready — length={len(audio)} samples ({len(audio)/16000:.1f}s)")

            # Transcribe with word-level timestamps
            lang_code = language if language else None
            print(f"[WhisperX] Starting transcription (language={lang_code!r})...")
            result = model.transcribe(audio, language=lang_code)

            # Get detected language
            detected_lang = result.get("language", "en")
            num_raw_segments = len(result.get("segments", []))
            raw_text_preview = " ".join(s.get("text", "") for s in result.get("segments", []))[:300]
            print(f"[WhisperX] Transcription done — detected_language={detected_lang!r}, raw_segments={num_raw_segments}")
            print(f"[WhisperX] Transcribed text preview: {raw_text_preview!r}")

            # Align word timestamps
            print(f"[WhisperX] Loading alignment model for language '{detected_lang}'...")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, device="cuda"
            )
            print(f"[WhisperX] Alignment model loaded. Running word-level alignment...")
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device="cuda"
            )
            print(f"[WhisperX] Alignment complete.")

            # Diarization (speaker segmentation) - WhisperX recommended pipeline
            from whisperx.diarize import DiarizationPipeline

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGINGFACE_HUB_TOKEN"
            )
            if not hf_token:
                print("[WhisperX] ERROR: HF_TOKEN not found in environment.")
                return {"success": False, "error": "HF_TOKEN not available"}

            print(f"[WhisperX] Initializing DiarizationPipeline...")
            diarize_model = DiarizationPipeline(token=hf_token, device="cuda")
            print(f"[WhisperX] DiarizationPipeline ready. Running diarization...")

            # Run diarization
            diarize_segments = diarize_model(audio)
            print(f"[WhisperX] Diarization complete.")

            # Assign speakers to words/segments
            print(f"[WhisperX] Assigning speakers to words/segments...")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"[WhisperX] Speaker assignment done.")

            # Extract segments with speakers
            segments = []
            all_speakers = set()

            for seg in result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")
                all_speakers.add(speaker)

                segments.append(
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip(),
                        "speaker": speaker,
                    }
                )

            # Extract words with speakers for finer granularity
            words = []
            for seg in result.get("segments", []):
                for word_data in seg.get("words", []):
                    words.append(
                        {
                            "start": word_data.get("start", 0),
                            "end": word_data.get("end", 0),
                            "word": word_data.get("word", ""),
                            "speaker": word_data.get("speaker", "UNKNOWN"),
                        }
                    )

            # Build full text
            full_text = " ".join([seg["text"] for seg in segments])

            speakers = sorted(list(all_speakers))

            print(f"[WhisperX] SUCCESS — segments={len(segments)}, words={len(words)}, speakers={speakers}, text_length={len(full_text)}")
            print(f"[WhisperX] Full transcription: {full_text[:500]!r}{'...' if len(full_text) > 500 else ''}")
            for seg in segments:
                print(f"[WhisperX]   [{seg['speaker']}] {seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text']!r}")

            _publish_progress_event(
                job_id,
                60,
                f"WhisperX complete ({len(speakers)} speakers detected)",
                "whisperx",
                {"speakers": speakers, "language": detected_lang},
            )

            return {
                "success": True,
                "text": full_text,
                "segments": segments,
                "words": words,
                "language": detected_lang,
                "num_speakers": len(speakers),
                "speakers": speakers,
            }

        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print(f"[WhisperX] Temp file cleaned up.")

    except Exception as e:
        print(f"[WhisperX] EXCEPTION: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_demucs_whisperx_image(),
    secrets=_r2_secret_list() + [modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_demucs_ft_whisperx(
    audio_bytes: bytes, filename: str, model_size: str = "medium", language: str = None, job_id: str = None
) -> dict:
    """
    Run Demucs FT separation and WhisperX transcription/diarization in one job.

    Returns both:
    - stems: Demucs WAV outputs keyed by filename
    - whisperx: transcript + diarization built from the vocals stem
    """
    import os
    import subprocess
    import tempfile

    try:
        if isinstance(audio_bytes, str) and audio_bytes.startswith(_PIPELINE_MOUNT):
            input_path = audio_bytes
            with tempfile.TemporaryDirectory() as temp_dir:
                local_input = os.path.join(temp_dir, filename)
                import shutil
                shutil.copy2(input_path, local_input)
                base_name = os.path.splitext(filename)[0]
                audio_path = local_input if filename.lower().endswith(".wav") else os.path.join(temp_dir, f"{base_name}_demucs.wav")
                if not filename.lower().endswith(".wav"):
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        local_input,
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        audio_path,
                        "-y",
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        return {"success": False, "error": result.stderr}

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                demucs_cmd = ["demucs", "-n", "htdemucs_ft", "--out", output_dir, audio_path]
                result = subprocess.run(demucs_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}

                result_dir = os.path.join(output_dir, "htdemucs_ft", base_name)
                if not os.path.exists(result_dir):
                    return {"success": False, "error": f"Demucs output directory not found: {result_dir}"}

                stems = {}
                vocals_volume_path = None
                background_volume_path = None
                for file in os.listdir(result_dir):
                    if file.endswith(".wav"):
                        file_path = os.path.join(result_dir, file)
                        volume_path = _publish_volume_output(file_path, "demucs", filename, file)
                        stems[file] = volume_path
                        if "vocals" in file.lower():
                            vocals_volume_path = volume_path
                        else:
                            background_volume_path = volume_path

                if not stems or vocals_volume_path is None or background_volume_path is None:
                    return {"success": False, "error": "Demucs stems missing"}

                whisperx_result = _run_whisperx_transcribe_diarize_from_path(
                    vocals_volume_path,
                    model_size=model_size,
                    language=language,
                )
                return {
                    "success": whisperx_result.get("success", False),
                    "input_filename": filename,
                    "demucs_model": "htdemucs_ft",
                    "stems": stems,
                    "whisperx": whisperx_result,
                    "text": whisperx_result.get("text", ""),
                    "segments": whisperx_result.get("segments", []),
                    "words": whisperx_result.get("words", []),
                    "language": whisperx_result.get("language"),
                    "num_speakers": whisperx_result.get("num_speakers", 0),
                    "speakers": whisperx_result.get("speakers", []),
                    "vocals_volume_path": vocals_volume_path,
                    "background_volume_path": background_volume_path,
                }
        if isinstance(audio_bytes, str) and _storage_scheme(audio_bytes):
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = _download_storage_object(audio_bytes, suffix=os.path.splitext(filename)[1] or ".wav")
                local_input = os.path.join(temp_dir, filename)
                import shutil
                shutil.copy2(input_path, local_input)
                base_name = os.path.splitext(filename)[0]
                audio_path = local_input if filename.lower().endswith(".wav") else os.path.join(temp_dir, f"{base_name}_demucs.wav")
                if not filename.lower().endswith(".wav"):
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        local_input,
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        audio_path,
                        "-y",
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        return {"success": False, "error": result.stderr}

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                demucs_cmd = ["demucs", "-n", "htdemucs_ft", "--out", output_dir, audio_path]
                result = subprocess.run(demucs_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}

                result_dir = os.path.join(output_dir, "htdemucs_ft", base_name)
                if not os.path.exists(result_dir):
                    return {"success": False, "error": f"Demucs output directory not found: {result_dir}"}

                stems = {}
                vocals_volume_path = None
                for file in os.listdir(result_dir):
                    if file.endswith(".wav"):
                        file_path = os.path.join(result_dir, file)
                        volume_path = _publish_volume_output(file_path, "demucs", filename, file)
                        stems[file] = volume_path
                        if "vocals" in file.lower():
                            vocals_volume_path = volume_path

                if not stems or vocals_volume_path is None:
                    return {"success": False, "error": "Demucs stems missing"}

                whisperx_result = _run_whisperx_transcribe_diarize_from_path(
                    vocals_volume_path,
                    model_size=model_size,
                    language=language,
                )
                return {
                    "success": whisperx_result.get("success", False),
                    "input_filename": filename,
                    "demucs_model": "htdemucs_ft",
                    "stems": stems,
                    "whisperx": whisperx_result,
                    "text": whisperx_result.get("text", ""),
                    "segments": whisperx_result.get("segments", []),
                    "words": whisperx_result.get("words", []),
                    "language": whisperx_result.get("language"),
                    "num_speakers": whisperx_result.get("num_speakers", 0),
                    "speakers": whisperx_result.get("speakers", []),
                    "vocals_volume_path": vocals_volume_path,
                }

        print(
            f"[Demucs+WhisperX] Starting - filename={filename!r}, audio_bytes={len(audio_bytes):,}, model_size={model_size!r}, language={language!r}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, filename)
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            base_name = os.path.splitext(filename)[0]
            if filename.lower().endswith(".wav"):
                audio_path = input_path
                print(f"[Demucs+WhisperX] Input is already WAV, using {audio_path}")
            else:
                audio_path = os.path.join(temp_dir, f"{base_name}_demucs.wav")
                print(f"[Demucs+WhisperX] Extracting audio from {filename}...")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    audio_path,
                    "-y",
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[Demucs+WhisperX] FFmpeg stdout: {result.stdout}")
                    print(f"[Demucs+WhisperX] FFmpeg stderr: {result.stderr}")
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            demucs_cmd = [
                "demucs",
                "-n",
                "htdemucs_ft",
                "--out",
                output_dir,
                audio_path,
            ]

            print(f"[Demucs+WhisperX] Running demucs command: {' '.join(demucs_cmd)}")
            result = subprocess.run(demucs_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[Demucs+WhisperX] Demucs stdout: {result.stdout}")
                print(f"[Demucs+WhisperX] Demucs stderr: {result.stderr}")
                raise RuntimeError(f"Demucs failed: {result.stderr}")

            result_dir = os.path.join(output_dir, "htdemucs_ft", base_name)
            if not os.path.exists(result_dir):
                raise RuntimeError(f"Demucs output directory not found: {result_dir}")

            stems = {}
            vocals_path = None
            for file in os.listdir(result_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(result_dir, file)
                    with open(file_path, "rb") as f:
                        stems[file] = f.read()
                    if "vocals" in file.lower():
                        vocals_path = file_path

            if not stems:
                raise RuntimeError("Demucs did not produce any WAV stems")
            if vocals_path is None:
                raise RuntimeError("Demucs vocals stem was not found")

            print("[Demucs+WhisperX] Running WhisperX on vocals stem...")
            whisperx_result = _run_whisperx_transcribe_diarize_from_path(
                vocals_path, model_size=model_size, language=language
            )
            if not whisperx_result.get("success"):
                return {
                    "success": False,
                    "error": whisperx_result.get("error", "WhisperX failed"),
                    "stems": stems,
                }

            return {
                "success": True,
                "input_filename": filename,
                "demucs_model": "htdemucs_ft",
                "stems": stems,
                "whisperx": whisperx_result,
                "text": whisperx_result.get("text", ""),
                "segments": whisperx_result.get("segments", []),
                "words": whisperx_result.get("words", []),
                "language": whisperx_result.get("language"),
                "num_speakers": whisperx_result.get("num_speakers", 0),
                "speakers": whisperx_result.get("speakers", []),
                "vocals_stem_name": os.path.basename(vocals_path),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_demucs_whisperx_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_demucs_ft_separation(audio_bytes: bytes, filename: str, job_id: str = None) -> dict:
    """
    Run Demucs FT separation only and return WAV stems.

    Returns:
    - stems: Demucs WAV outputs keyed by filename
    """
    import os
    import io
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    try:
        _publish_progress_event(job_id, 25, "Demucs separation started", "vocal_separation")
        if isinstance(audio_bytes, str) and _storage_scheme(audio_bytes):
            input_path = _download_storage_object(audio_bytes, suffix=os.path.splitext(filename)[1] or ".wav")
            with tempfile.TemporaryDirectory() as temp_dir:
                local_input = os.path.join(temp_dir, filename)
                import shutil
                shutil.copy2(input_path, local_input)
                base_name = os.path.splitext(filename)[0]
                audio_path = local_input if filename.lower().endswith(".wav") else os.path.join(temp_dir, f"{base_name}_demucs.wav")
                if not filename.lower().endswith(".wav"):
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        local_input,
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        audio_path,
                        "-y",
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        return {"success": False, "error": result.stderr}

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                demucs_cmd = ["demucs", "-n", "htdemucs_ft", "--out", output_dir, audio_path]
                result = subprocess.run(demucs_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}

                result_dir = os.path.join(output_dir, "htdemucs_ft", base_name)
                if not os.path.exists(result_dir):
                    return {"success": False, "error": f"Demucs output directory not found: {result_dir}"}

                stems = {}
                stem_arrays = {}
                stem_sr = 44100
                output_bucket = os.getenv("R2_OUTPUT_BUCKET", "octavia-outputs")
                vocals_storage_path = None
                vocals_volume_path = None
                background_storage_path = None
                background_volume_path = None
                for file in os.listdir(result_dir):
                    if file.endswith(".wav"):
                        file_path = os.path.join(result_dir, file)
                        storage_path, volume_path = _publish_audio_output(
                            file_path,
                            output_bucket,
                            f"modal/demucs/{filename}/{file}",
                            ("demucs", filename, file),
                            "audio/wav",
                        )
                        stems[file] = storage_path
                        if "vocals" in file.lower():
                            vocals_storage_path = storage_path
                            vocals_volume_path = volume_path
                        else:
                            background_storage_path = storage_path
                            background_volume_path = volume_path
                        audio_data, sr = sf.read(file_path, always_2d=True)
                        stem_arrays[file] = audio_data
                        stem_sr = sr

                vocals_file = next((name for name in stems.keys() if "vocals" in name.lower()), None)
                if vocals_file is None:
                    return {"success": False, "error": "Demucs vocals stem was not found"}

                background_arrays = [
                    data for name, data in stem_arrays.items() if "vocals" not in name.lower()
                ]
                if not background_arrays:
                    return {"success": False, "error": "Demucs background stems were not found"}

                max_len = max(arr.shape[0] for arr in background_arrays)
                mix = np.zeros((max_len, background_arrays[0].shape[1]), dtype=np.float32)
                for arr in background_arrays:
                    if arr.shape[0] < max_len:
                        pad = np.zeros((max_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                        arr = np.vstack([arr, pad])
                    mix += arr.astype(np.float32)

                peak = np.max(np.abs(mix))
                if peak > 1.0:
                    mix = mix / peak

                background_buffer = io.BytesIO()
                sf.write(background_buffer, mix, stem_sr, format="WAV")
                background_path = os.path.join(temp_dir, "background.wav")
                with open(background_path, "wb") as f:
                    f.write(background_buffer.getvalue())
                background_storage_path = _upload_storage_file(
                    background_path,
                    output_bucket,
                    f"modal/demucs/{filename}/background.wav",
                    "audio/wav",
                )

                return {
                    "success": True,
                    "input_filename": filename,
                    "demucs_model": "htdemucs_ft",
                    "stems": stems,
                    "vocals_storage_path": vocals_storage_path or stems[vocals_file],
                    "vocals_volume_path": vocals_volume_path,
                    "background_storage_path": background_storage_path,
                    "background_volume_path": background_volume_path,
                }

        print(
            f"[DemucsFT] Starting - filename={filename!r}, audio_bytes={len(audio_bytes):,}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, filename)
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            base_name = os.path.splitext(filename)[0]
            if filename.lower().endswith(".wav"):
                audio_path = input_path
                print(f"[DemucsFT] Input is already WAV, using {audio_path}")
            else:
                audio_path = os.path.join(temp_dir, f"{base_name}_demucs.wav")
                print(f"[DemucsFT] Extracting audio from {filename}...")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    audio_path,
                    "-y",
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[DemucsFT] FFmpeg stdout: {result.stdout}")
                    print(f"[DemucsFT] FFmpeg stderr: {result.stderr}")
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            demucs_cmd = [
                "demucs",
                "-n",
                "htdemucs_ft",
                "--out",
                output_dir,
                audio_path,
            ]

            print(f"[DemucsFT] Running demucs command: {' '.join(demucs_cmd)}")
            result = subprocess.run(demucs_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[DemucsFT] Demucs stdout: {result.stdout}")
                print(f"[DemucsFT] Demucs stderr: {result.stderr}")
                raise RuntimeError(f"Demucs failed: {result.stderr}")

            result_dir = os.path.join(output_dir, "htdemucs_ft", base_name)
            if not os.path.exists(result_dir):
                raise RuntimeError(f"Demucs output directory not found: {result_dir}")

            stems = {}
            stem_arrays = {}
            stem_sr = 44100
            for file in os.listdir(result_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(result_dir, file)
                    with open(file_path, "rb") as f:
                        stems[file] = f.read()
                    audio_data, sr = sf.read(file_path, always_2d=True)
                    stem_arrays[file] = audio_data
                    stem_sr = sr

            vocals_file = next((name for name in stems.keys() if "vocals" in name.lower()), None)
            if vocals_file is None:
                raise RuntimeError("Demucs vocals stem was not found")

            vocals_bytes = stems[vocals_file]

            background_arrays = [
                data for name, data in stem_arrays.items() if "vocals" not in name.lower()
            ]
            if not background_arrays:
                raise RuntimeError("Demucs background stems were not found")

            max_len = max(arr.shape[0] for arr in background_arrays)
            mix = np.zeros((max_len, background_arrays[0].shape[1]), dtype=np.float32)
            for arr in background_arrays:
                if arr.shape[0] < max_len:
                    pad = np.zeros((max_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    arr = np.vstack([arr, pad])
                mix += arr.astype(np.float32)

            peak = np.max(np.abs(mix))
            if peak > 1.0:
                mix = mix / peak

            background_buffer = io.BytesIO()
            sf.write(background_buffer, mix, stem_sr, format="WAV")
            background_bytes = background_buffer.getvalue()

            if not stems:
                raise RuntimeError("Demucs did not produce any WAV stems")

            _publish_progress_event(job_id, 35, "Demucs separation complete", "vocal_separation")
            return {
                "success": True,
                "input_filename": filename,
                "demucs_model": "htdemucs_ft",
                "stems": stems,
                "vocals": vocals_bytes,
                "background": background_bytes,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_gender_detection(audio_bytes: bytes, job_id: str = None) -> dict:
    """
    Detect speaker gender from audio using prithivMLmods/Common-Voice-Gender-Detection.

    Uses wav2vec2-based model with ~98.5% accuracy.

    Args:
        audio_bytes: WAV audio data (mono, 16kHz recommended)

    Returns:
        Dictionary with gender detection results:
        - gender: 'male' or 'female'
        - confidence: float (0.0-1.0)
        - female_prob: float
        - male_prob: float
    """
    import logging
    import os
    import tempfile

    from transformers import pipeline as hf_pipeline

    logger = logging.getLogger(__name__)

    try:
        audio_path, cleanup_needed = _resolve_input_path(audio_bytes, suffix=".wav")
        _publish_progress_event(job_id, 45, "Gender detection started", "gender_detection")

        try:
            # Load gender detection model
            logger.info("Loading gender detection model...")
            classifier = hf_pipeline(
                "audio-classification",
                model="prithivMLmods/Common-Voice-Gender-Detection",
                device=0,  # Use GPU
            )

            # Run classification
            results = classifier(audio_path)

            # Parse results
            female_prob = 0.0
            male_prob = 0.0
            for result in results:
                label = result["label"].lower()
                score = result["score"]
                if "female" in label:
                    female_prob = score
                elif "male" in label:
                    male_prob = score

            gender = "female" if female_prob > male_prob else "male"
            confidence = max(female_prob, male_prob)

            logger.info(f"Gender detected: {gender} (confidence: {confidence:.2%})")
            _publish_progress_event(
                job_id,
                50,
                f"Gender detected: {gender}",
                "gender_detection",
                {"gender": gender, "confidence": confidence},
            )

            return {
                "success": True,
                "gender": gender,
                "confidence": confidence,
                "female_prob": female_prob,
                "male_prob": male_prob,
            }

        finally:
            if cleanup_needed and os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        logger.error(f"Gender detection failed: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    gpu="T4",
    image=get_ml_image(),
    timeout=43200,
    secrets=_r2_secret_list(),
    volumes={"/pipeline": _PIPELINE_VOLUME},
)
def remote_speaker_gender_detection(audio_bytes: bytes, segments: list, job_id: str = None) -> dict:
    """
    Detect gender per speaker by combining diarization segments with gender analysis.

    Extracts audio slices for each speaker, runs gender detection, and aggregates results.

    Args:
        audio_bytes: WAV audio data (full audio file)
        segments: List of diarization segment dicts with 'start', 'end', 'speaker' keys

    Returns:
        Dictionary with per-speaker gender results:
        - speaker_genders: {speaker_id: {gender, confidence, female_prob, male_prob}}
    """
    import io
    import logging
    import os
    import tempfile
    from collections import defaultdict

    from pydub import AudioSegment
    from transformers import pipeline as hf_pipeline

    logger = logging.getLogger(__name__)

    try:
        audio_path, cleanup_needed = _resolve_input_path(audio_bytes, suffix=".wav")
        _publish_progress_event(job_id, 45, "Speaker gender detection started", "speaker_gender_detection")

        try:
            # Load full audio
            audio = AudioSegment.from_file(audio_path)

            # Load gender detection model
            logger.info("Loading gender detection model for per-speaker analysis...")
            classifier = hf_pipeline(
                "audio-classification",
                model="prithivMLmods/Common-Voice-Gender-Detection",
                device=0,  # Use GPU
            )

            # Group segments by speaker
            speaker_segments = defaultdict(list)
            for seg in segments:
                speaker = seg.get("speaker", "SPEAKER_00")
                speaker_segments[speaker].append(seg)

            speaker_genders = {}

            for speaker, segs in speaker_segments.items():
                # Concatenate all segments for this speaker (up to 30 seconds for efficiency)
                speaker_audio = AudioSegment.empty()
                total_duration_ms = 0
                max_sample_ms = 30000  # 30 seconds max per speaker

                for seg in segs:
                    start_ms = int(seg["start"] * 1000)
                    end_ms = int(seg["end"] * 1000)

                    # Clamp to audio bounds
                    start_ms = max(0, start_ms)
                    end_ms = min(len(audio), end_ms)

                    if start_ms >= end_ms:
                        continue

                    speaker_audio += audio[start_ms:end_ms]
                    total_duration_ms += end_ms - start_ms

                    if total_duration_ms >= max_sample_ms:
                        break

                if len(speaker_audio) < 500:  # Less than 0.5 seconds
                    logger.warning(
                        f"Speaker {speaker} has too little audio ({len(speaker_audio)}ms), skipping"
                    )
                    continue

                # Export speaker audio to temp file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_speaker:
                    speaker_audio.export(temp_speaker, format="wav")
                    speaker_audio_path = temp_speaker.name

                try:
                    # Run classification
                    results = classifier(speaker_audio_path)

                    female_prob = 0.0
                    male_prob = 0.0
                    for result in results:
                        label = result["label"].lower()
                        score = result["score"]
                        if "female" in label:
                            female_prob = score
                        elif "male" in label:
                            male_prob = score

                    gender = "female" if female_prob > male_prob else "male"
                    confidence = max(female_prob, male_prob)

                    speaker_genders[speaker] = {
                        "gender": gender,
                        "confidence": confidence,
                        "female_prob": female_prob,
                        "male_prob": male_prob,
                    }

                    logger.info(
                        f"Speaker {speaker}: {gender} (confidence: {confidence:.2%})"
                    )

                finally:
                    if os.path.exists(speaker_audio_path):
                        os.remove(speaker_audio_path)

            _publish_progress_event(
                job_id,
                50,
                f"Speaker gender detection complete ({len(speaker_genders)} speakers)",
                "speaker_gender_detection",
                {"speaker_genders": speaker_genders},
            )

            return {"success": True, "speaker_genders": speaker_genders}

        finally:
            if cleanup_needed and os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        logger.error(f"Speaker gender detection failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # This allows running locally for testing
    # modal run modal_app.py --help
    app
