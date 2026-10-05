"""Owned offline adapter for pinned IndexTTS-2; no model weights are bundled."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import resource
import sys
import types
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .logic import configuration, EMOTIONS
    from .process import IndexTtsError
else:
    package = types.ModuleType("index_worker_owned")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    from index_worker_owned.assets_store import AssetError, prepare
    from index_worker_owned.logic import configuration, EMOTIONS
    from index_worker_owned.process import IndexTtsError

SDK_REVISION = "d9e41aac89fd00b3d71497fddb287b7f24613712"


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    try:
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(cap) - used))
    except (OSError, ValueError):
        pass
    return available



def require_dependencies():
    from importlib.metadata import distribution, version
    if not (3,10) <= sys.version_info[:2] < (3,12):
        raise IndexTtsError("dependency")
    for name, expected in (("torch","2.8.0"),("torchaudio","2.8.0"),("indextts","2.0.0"),
                           ("transformers","4.52.1"),("accelerate","1.8.1"),("WeTextProcessing","1.2.0")):
        if version(name).split("+")[0] != expected:
            raise IndexTtsError("dependency")
    try:
        origin=json.loads(distribution("indextts").read_text("direct_url.json") or "{}")
        if origin.get("url","").removesuffix(".git")!="https://github.com/index-tts/index-tts" or origin.get("vcs_info",{}).get("commit_id")!=SDK_REVISION:
            raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise IndexTtsError("dependency") from None


def load_model(assets, cfg, job):
    from indextts import infer_v2 as sdk
    original=sdk.TextNormalizer

    class OwnedNormalizer(original):
        def load(self):
            from tn.chinese.normalizer import Normalizer as Chinese
            from tn.english.normalizer import Normalizer as English
            zh=job/"tn-zh";en=job/"tn-en"
            zh.mkdir(mode=0o700);en.mkdir(mode=0o700)
            self.zh_normalizer=Chinese(cache_dir=str(zh),remove_interjections=False,remove_erhua=False,overwrite_cache=False)
            self.en_normalizer=English(cache_dir=str(en),overwrite_cache=False)

    # SDK defaults write caches into the installed package. Keep those caches in
    # this single-request worker's owned job directory without editing the SDK.
    sdk.TextNormalizer=OwnedNormalizer
    try:
        model=sdk.IndexTTS2(cfg_path=str(assets/"model/config.yaml"),model_dir=str(assets/"model"),
            device=cfg["device"],use_fp16=cfg["precision"]=="float16",
            use_cuda_kernel=False,use_deepspeed=False,use_accel=False,use_torch_compile=False,
            use_qwen_emo=False,aux_paths={
                "w2v_bert":str(assets/"w2v-bert"),
                "semantic_codec":str(assets/"maskgct/model.safetensors"),
                "campplus":str(assets/"campplus/campplus_cn_common.bin"),
                "bigvgan":str(assets/"bigvgan")})
    finally:
        sdk.TextNormalizer=original
    if str(model.model_version)!="2.0" or model.stop_mel_token!=8193:
        raise IndexTtsError("assets_corrupt")
    return model


def guard_segments(model, cfg):
    """Observe real segment boundaries and EOS before the SDK discards stop tokens."""
    original_split=model.tokenizer.split_segments
    original_generate=model.gpt.inference_speech
    state={"expected":0,"segments":0,"tokens":0}

    def split(tokens, maximum, **kwargs):
        if state["expected"] or maximum!=cfg["max_segment_tokens"]:
            raise IndexTtsError("model_failed")
        segments=original_split(tokens,maximum,**kwargs)
        if not isinstance(segments,list) or not 1<=len(segments)<=cfg["max_segments"] or any(
                not isinstance(segment,list) or not segment for segment in segments):
            raise IndexTtsError("audio_limit")
        state["expected"]=len(segments)
        return segments

    def generate(*args,**kwargs):
        if not state["expected"] or state["segments"]>=state["expected"] or kwargs.get("max_generate_length")!=cfg["max_mel_tokens"]:
            raise IndexTtsError("model_failed")
        result=original_generate(*args,**kwargs)
        if not isinstance(result,tuple) or len(result)!=2:
            raise IndexTtsError("model_failed")
        codes=result[0]
        if getattr(codes,"ndim",None)!=2 or codes.shape[0]!=1 or not 2<=codes.shape[1]<=cfg["max_mel_tokens"]:
            raise IndexTtsError("model_failed")
        tokens=codes[0].tolist()
        if any(type(token) is not int for token in tokens):
            raise IndexTtsError("model_failed")
        if tokens[-1]!=8193 or 8193 in tokens[:-1]:
            raise IndexTtsError("audio_limit")
        state["segments"]+=1;state["tokens"]+=len(tokens)
        return result

    model.tokenizer.split_segments=split
    model.gpt.inference_speech=generate
    return original_split,original_generate,state


def generate_audio(model,text,cfg,job):
    import numpy as np
    original_split,original_generate,state=guard_segments(model,cfg)
    emotion=None
    if cfg["emotion"]!="reference":
        emotion=[cfg["emotion_strength"] if e==cfg["emotion"] else 0. for e in EMOTIONS[1:]]
    try:
        rate,audio=model.infer(spk_audio_prompt=str(job/"reference.wav"),text=text,output_path=None,
            emo_audio_prompt=None,emo_alpha=1.,emo_vector=emotion,use_emo_text=False,
            use_random=False,interval_silence=cfg["interval_silence_ms"],verbose=False,
            max_text_tokens_per_segment=cfg["max_segment_tokens"],stream_return=False,
            more_segment_before=0,do_sample=True,top_p=cfg["top_p"],top_k=cfg["top_k"],
            temperature=cfg["temperature"],num_beams=3,length_penalty=0.,
            repetition_penalty=cfg["repetition_penalty"],max_mel_tokens=cfg["max_mel_tokens"])
    finally:
        model.tokenizer.split_segments=original_split
        model.gpt.inference_speech=original_generate
    if not state["expected"] or state["segments"]!=state["expected"]:
        raise IndexTtsError("audio_limit")
    audio=np.asarray(audio)
    if type(rate) is not int or rate!=22050 or audio.dtype!=np.dtype("int16") or audio.ndim!=2 or audio.shape[1]!=1:
        raise IndexTtsError("audio_invalid")
    if not 0<audio.shape[0]<=rate*cfg["max_audio_sec"] or not np.any(audio):
        raise IndexTtsError("audio_invalid")
    path=job/"generated.wav"
    with path.open("xb") as target:
        os.chmod(path,0o600)
        with wave.open(target,"wb") as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(audio.astype("<i2").tobytes())
    return {"complete":True,"eos":True,"tokens":state["tokens"],"segments":state["segments"],
            "sample_rate":rate,"frames":int(audio.shape[0])}


def infer(item):
    cfg=configuration(item["config"])
    if cfg["memory_limit_mib"]:
        limit=cfg["memory_limit_mib"]*1048576
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cpu_limit=cfg["timeout_sec"]*cfg["threads"]
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_limit,cpu_limit+1))
    require_dependencies()
    if available_ram()<cfg["min_available_ram_mib"]*1048576:
        raise IndexTtsError("resources")
    import torch
    torch.set_num_threads(cfg["threads"]);torch.set_num_interop_threads(1)
    if cfg["device"]=="cuda:0" and not torch.cuda.is_available():
        raise IndexTtsError("device")
    assets=prepare(item["assets"])
    job=Path(item["job"])
    model=load_model(assets,cfg,job)
    with torch.inference_mode():
        return generate_audio(model,item["request"]["text"],cfg,job)


def main():
    data = sys.stdin.buffer.read(32769)
    if len(data) > 32768:
        raise IndexTtsError("input")
    with redirect_stdout(sys.stderr):
        result = infer(json.loads(data))
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc, IndexTtsError): code = exc.code
        elif isinstance(exc, AssetError): code = str(exc)
        elif isinstance(exc, (ImportError, PackageNotFoundError)): code = "dependency"
        elif isinstance(exc, MemoryError) or isinstance(exc, RuntimeError) and any(s in str(exc) for s in ("out of memory", "DefaultCPUAllocator", "Cannot allocate memory")):
            code = "resources"
        else: code = "model_failed"
        print(json.dumps({"error_code": code})); raise SystemExit(1) from None
