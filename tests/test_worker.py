"""Pinned SDK contract doubles, not actual Index model inference."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import wave
import numpy as np
from blocs.index_tts import logic,worker
from index_fixture import settings

class Model:
    def __init__(self,audio=None,tokens=None,segments=2,skip=0,rate=22050):
        self.audio=np.ones((11025,1),dtype=np.int16) if audio is None else audio
        self.rate,self.segments,self.skip=rate,segments,skip
        self.tokenizer=NS(split_segments=lambda tokens,maximum,**kw:[["word"]]*segments)
        self.gpt=NS(inference_speech=lambda *a,**kw:(np.array([tokens if tokens is not None else [12,13,8193]]),None))
    def infer(self,**kw):
        self.arguments=kw
        segments=self.tokenizer.split_segments(["word"]*self.segments,kw["max_text_tokens_per_segment"],quick_streaming_tokens=0)
        for part in segments[:len(segments)-self.skip]:
            self.gpt.inference_speech(None,None,max_generate_length=kw["max_mel_tokens"])
        return self.rate,self.audio

class WorkerTests(unittest.TestCase):
    def test_configuration_matches_all_fields(self):
        package=Path(__file__).parents[1]
        fields=json.loads((package/"fields.json").read_text())
        self.assertEqual({f["key"] for f in fields},set(logic.DEFAULTS))
        self.assertEqual(json.loads((package/"model.json").read_text())["config"],logic.DEFAULTS)
        for f in fields:
            if f["type"] in {"integer","number"}:
                for value in (True,None,"1",float("nan"),f["min"]-1,f["max"]+1):
                    with self.subTest(key=f["key"],value=value),self.assertRaises(logic.IndexTtsError):
                        logic.configuration({f["key"]:value})
        for cfg in ({"emotion":"fake"},{"precision":"bfloat16"},{"device":"cpu","precision":"float16"},{"reference_audio":"\ud800"},{"memory_limit_mib":4095}):
            with self.subTest(cfg=cfg),self.assertRaises(logic.IndexTtsError):logic.configuration(cfg)

    def test_complete_multi_segment_audio_and_exact_arguments(self):
        cfg=logic.configuration({"emotion":"happy","emotion_strength":.6})
        model=Model();split=model.tokenizer.split_segments;generate=model.gpt.inference_speech
        with TemporaryDirectory() as directory:
            job=Path(directory);info=worker.generate_audio(model,"Hello world",cfg,job)
            self.assertEqual(info["segments"],2);self.assertEqual(info["tokens"],6)
            self.assertEqual(logic.audio_info(job/"generated.wav",cfg,json.dumps(info)),info)
            self.assertEqual(model.arguments,dict(spk_audio_prompt=str(job/"reference.wav"),text="Hello world",output_path=None,
                emo_audio_prompt=None,emo_alpha=1.,emo_vector=[.6,0.,0.,0.,0.,0.,0.,0.],use_emo_text=False,
                use_random=False,interval_silence=200,verbose=False,max_text_tokens_per_segment=120,
                stream_return=False,more_segment_before=0,do_sample=True,top_p=.8,top_k=30,temperature=.8,
                num_beams=3,length_penalty=0.,repetition_penalty=10,max_mel_tokens=1500))
        self.assertIs(model.tokenizer.split_segments,split);self.assertIs(model.gpt.inference_speech,generate)

    def test_missing_eos_or_segment_never_publishes(self):
        for model in (Model(tokens=[1,2]),Model(tokens=[8193,8193]),Model(tokens=[]),Model(skip=1),Model(segments=0),Model(segments=17)):
            original=model.gpt.inference_speech
            with TemporaryDirectory() as directory,self.subTest(model=model):
                with self.assertRaises(logic.IndexTtsError):worker.generate_audio(model,"Hello",logic.DEFAULTS,Path(directory))
                self.assertFalse(list(Path(directory).iterdir()))
                self.assertIs(model.gpt.inference_speech,original)

    def test_bad_pcm_is_refused(self):
        for audio in (np.zeros((3,1),dtype=np.int16),np.ones(3,dtype=np.int16),np.ones((3,2),dtype=np.int16),
                      np.ones((3,1),dtype=np.float32),np.ones((22051,1),dtype=np.int16),np.empty((0,1),dtype=np.int16)):
            with TemporaryDirectory() as directory,self.subTest(shape=audio.shape),self.assertRaises(logic.IndexTtsError):
                worker.generate_audio(Model(audio=audio),"Hello",logic.configuration({"max_audio_sec":1}),Path(directory))
        with TemporaryDirectory() as directory,self.assertRaises(logic.IndexTtsError):
            worker.generate_audio(Model(rate=24000),"Hello",logic.DEFAULTS,Path(directory))

    def test_local_paths_and_owned_normalizer_caches(self):
        calls=[]
        class Normalizer:
            def __init__(self,**kw):pass
        def create(**kw):
            calls.append(kw);normalizer=sdk.TextNormalizer(enable_glossary=True);normalizer.load()
            return NS(model_version=2.0,stop_mel_token=8193)
        sdk=NS(TextNormalizer=Normalizer,IndexTTS2=create)
        constructed=[]
        def record(**kw):constructed.append(kw);return NS()
        with TemporaryDirectory() as directory,patch.dict(sys.modules,{
                "indextts":NS(infer_v2=sdk),"tn.chinese.normalizer":NS(Normalizer=record),
                "tn.english.normalizer":NS(Normalizer=record)}):
            root=Path(directory)
            model=worker.load_model(root,logic.DEFAULTS,root)
            self.assertEqual(model.model_version,2.0);self.assertIs(sdk.TextNormalizer,Normalizer)
            self.assertEqual(calls[0],dict(cfg_path=str(root/"model/config.yaml"),model_dir=str(root/"model"),
                device="cuda:0",use_fp16=False,use_cuda_kernel=False,use_deepspeed=False,use_accel=False,use_torch_compile=False,
                use_qwen_emo=False,aux_paths={"w2v_bert":str(root/"w2v-bert"),
                "semantic_codec":str(root/"maskgct/model.safetensors"),"campplus":str(root/"campplus/campplus_cn_common.bin"),
                "bigvgan":str(root/"bigvgan")}))
            self.assertEqual({p.name for p in root.iterdir()},{"tn-zh","tn-en"})
            self.assertEqual([x["cache_dir"] for x in constructed],[str(root/"tn-zh"),str(root/"tn-en")])

    def test_reference_limits_preservation_and_real_codecs(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg=logic.configuration(settings(root));source=Path(cfg["reference_audio"]);original=source.read_bytes()
            for rate,frames,accepted in ((8000,8000,True),(16000,240000,True),(22050,330750,True),
                                         (22050,330751,False),(44100,661500,True),(48000,720000,True),
                                         (48000,724800,False),(22050,100,False)):
                audio=(np.sin(np.arange(frames)*440*2*np.pi/rate)*8000).astype("<i2")
                with wave.open(str(source),"wb") as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(audio.tobytes())
                before=source.read_bytes()
                with TemporaryDirectory(dir=root) as temporary:
                    try:result=logic.prepare_reference(cfg,root,Path(temporary),time.monotonic()+10,lambda:False)
                    except logic.IndexTtsError as exc:self.assertFalse(accepted,(rate,frames,exc))
                    else:self.assertTrue(accepted,(rate,frames,result));self.assertLessEqual(result["duration_sec"],15)
                self.assertEqual(source.read_bytes(),before)
            source.write_bytes(original)
            import subprocess
            for suffix in (".ogg",".webm",".mp3",".flac",".m4a"):
                compressed=root/("reference"+suffix)
                subprocess.run(["ffmpeg","-v","error","-i",str(source),str(compressed)],check=True,timeout=10,capture_output=True)
                with TemporaryDirectory(dir=root) as temporary:
                    result=logic.prepare_reference({**cfg,"reference_audio":str(compressed)},root,Path(temporary),time.monotonic()+10,lambda:False)
                    self.assertGreater(result["duration_sec"],.9)
            for raw in ("",str(root/"missing.wav")):
                with TemporaryDirectory(dir=root) as temporary,self.assertRaises(logic.IndexTtsError):
                    logic.prepare_reference({**cfg,"reference_audio":raw},root,Path(temporary),time.monotonic()+10,lambda:False)
            silent=root/"silence.wav"
            with wave.open(str(silent),"wb") as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(22050);wav.writeframes(b"\0"*44100)
            with TemporaryDirectory(dir=root) as temporary,self.assertRaises(logic.IndexTtsError):
                logic.prepare_reference({**cfg,"reference_audio":str(silent)},root,Path(temporary),time.monotonic()+10,lambda:False)

if __name__=="__main__":unittest.main()
