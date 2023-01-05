# Japanese GSLM

This is an Japanese implementation of [Generative Spoken Language Model](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm) to support textless NLP in Japanese. </br> Submitted to Acoustical Society of Japan, 2023 Spring.
</br>

## How to use
- PyTorch version >= 1.10.0
- Python version >= 3.8

### Install requirements
It is pre-required to install the [fairseq](https://github.com/facebookresearch/fairseq/) library and all the requirements the library needs.

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

pip install librosa, unidecode, inflect
```

## Re-synthesis of voice signal
### speech2unit

The procedure for speech2unit is the same as the gslm example in [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit). 


You can convert the Japanese voice signal to discrete unit through this [pre-trained quantization model](). Route the downloaded model to ```KM_MODEL_PATH```. 


This file replaces the ```HuBERT Base + KM200``` model provided by fariseq, so it is required to download ```HuBERT-Base``` model as a pretrained acoustic model.

```
TYPE='hubert'
CKPT_PATH=<path_of_pretrained_acoustic_model>
LAYER=6
KM_MODEL_PATH=<output_path_of_the_kmeans_model>
MANIFEST=<tab_separated_manifest_of_audio_files_to_quantize>
OUT_QUANTIZED_FILE=<output_quantized_audio_file_path>

python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"
```

### unit2speech

unit2speech model is modified Tacotron2 model that learns to synthesize speech from discrete speech units. 
You can convert the discrete unit to synthesized voice through this [model](). Also, it is required to download [Waveglow checkpoint](https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_new.pt) for Vocoder. 


Conversion from unit to speech is available with ```unit2speech_ja.py``` from this repository.

```
TTS_MODEL_PATH=<unit2speech_model_file_path>
OUT_DIR=<dir_to_dump_synthesized_audio_files>
WAVEGLOW_PATH=<path_where_you_have_downloaded_waveglow_checkpoint>

python unit2speech_ja.py \
    --tts_model_path $TTS_MODEL_PATH \
    --out_audio_dir $OUT_DIR \
    --waveglow_path  $WAVEGLOW_PATH \
```

## References
- Lakhotia, Kushal et al. On Generative Spoken Language Modeling from Raw Audio. Transactions of the Association for Computational Linguistics, 9:1336–1354, 2021. 
- Ott, Myle et al. fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 48–53, 2019.
