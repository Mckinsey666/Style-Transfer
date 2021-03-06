# Style-Transfer
Simple implementation of the classic paper by Gatys et al. Please refer to the original paper for more detail: https://arxiv.org/abs/1508.0657.
## Usage 
```
>> python3 style_transfer.py -h
usage: style_transfer.py [-h] [-d {cpu,cuda}] [-i ITERATIONS] [-c CONTENT]
                         [-s STYLE] [-o OUTPUT] [--output_dir OUTPUT_DIR]
                         [--content_dir CONTENT_DIR] [--style_dir STYLE_DIR]
                         [--style_weight STYLE_WEIGHT]
                         [--content_weight CONTENT_WEIGHT]
                         [--image_size IMAGE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device to run style transfer on.
  -i ITERATIONS, --iterations ITERATIONS
                        Number of style transfer iterations
  -c CONTENT, --content CONTENT
                        Content image
  -s STYLE, --style STYLE
                        Style image
  -o OUTPUT, --output OUTPUT
                        Output image
  --output_dir OUTPUT_DIR, --od OUTPUT_DIR
                        Where your output images are stored
  --content_dir CONTENT_DIR, --cd CONTENT_DIR
                        Where your content images are stored
  --style_dir STYLE_DIR, --sd STYLE_DIR
                        Where your style images are stored
  --style_weight STYLE_WEIGHT, --sw STYLE_WEIGHT
                        Weight of style image
  --content_weight CONTENT_WEIGHT, --cw CONTENT_WEIGHT
                        Weight of content image
  --image_size IMAGE_SIZE, --sz IMAGE_SIZE
                        Weight of content image
```
## Overview
![VGG19](./vgg19.png)

We extract `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1` in VGG19 as the style layers, and `conv4_2` in VGG19 as the content layer. Our goal is to minimize the style and content losses of the generated image.

## Results
**Original image**
![original image](./images/contents/city.png)
***
**Van gough style**
![city + van gough](./images/samples/city+van_gough.png)
***
**Kanagawa style**
![city + kanagawa](./images/samples/city_kanagawa.png)
***
**Picasso style**
![city + picasso](./images/samples/city+picasso.png)
***
**Mondrian style**
![city + mondrian](./images/samples/city_mondrian.png)