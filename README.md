# Сжатие изображений при помощи нейронных сетей

Задания:

1. Написать кодер и декодер, которые работают как отдельные программы

- Кодер: https://github.com/nikitaromanoov/image_compression/blob/main/encoder.py
- Декодер: https://github.com/nikitaromanoov/image_compression/blob/main/decoder.py

Примеры запусков:

`python image_compression/encoder.py --path_encoder "/kaggle/input/enciderdecoder-2/2b_encoder.pth"  --path_image "/kaggle/input/datasettest/peppers.png" --path_result  "2b_peppers.json"`


`python image_compression/decoder.py --path_decoder "/kaggle/input/enciderdecoder-2/2b_decoder.pth"  --path_compressed "2b_peppers.json" --path_result "2b_peppers.png"`

3. На вход кодера подается изображение в  raw-формата (png, bpm и т.д.), а также режим сжатия (слабее-сильнее). На выходе кодер генерирует файл со сжатым изображением.

- режим задается параметром B и соотвчетвующей моделью. Реализовано два режима: B=2 и B=8

4. На вход декордера подается сжатый файл, на выходе декодер генерирует изображение в  raw-формате

6. Алгоритм сжатия должен включаться в себя
* вычисление признаков - https://github.com/nikitaromanoov/image_compression/blob/69a7109f0e6ea79ae8ee08fd6a9aaa22a44e9942/encoder.py#L77
* квантование признаков - https://github.com/nikitaromanoov/image_compression/blob/69a7109f0e6ea79ae8ee08fd6a9aaa22a44e9942/encoder.py#L87
* сжатие квантованных признаков без потерь - https://github.com/nikitaromanoov/image_compression/blob/69a7109f0e6ea79ae8ee08fd6a9aaa22a44e9942/encoder.py#L88

График сравнения предложенного алгоритма с  JPEG в виде кривых  PSI/BPP, где BPP(bits per pixel) - количество бит, затрачиваемое в среднем на один пиксель изображения для текстовых изображений.

<img width="329" alt="image" src="https://github.com/nikitaromanoov/image_compression/assets/91135334/a344787a-9b1a-487f-8c75-3ffe1f0f0fe1">
<img width="332" alt="image" src="https://github.com/nikitaromanoov/image_compression/assets/91135334/94698aeb-1cce-424d-9f30-1273a5b7d26c">
<img width="314" alt="image" src="https://github.com/nikitaromanoov/image_compression/assets/91135334/cfbfb859-838a-4f58-a7cb-5074b78c164c">

