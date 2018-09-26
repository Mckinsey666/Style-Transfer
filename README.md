# Style-Transfer
Simple implementation of the classic paper by Gatys et al. Please refer to the original paper: https://arxiv.org/abs/1508.0657.
# Usage 
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
## Results
