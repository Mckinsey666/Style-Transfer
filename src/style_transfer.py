import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from argparse import ArgumentParser
import os
from utils import style_transfer, to_tensor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', help = 'Device to run style transfer on.', choices = ['cpu', 'cuda'], type = str)
    parser.add_argument('-i', '--iterations', help = 'Number of style transfer iterations', type = int)
    parser.add_argument('-c', '--content', help = 'Content image', type = str)
    parser.add_argument('-s', '--style', help = 'Style image', type = str)
    parser.add_argument('-o', '--output', help = 'Output image', type = str)
    parser.add_argument('--output_dir', '--od', help = 'Where your output images are stored', type = str)
    parser.add_argument('--content_dir', '--cd', help = 'Where your content images are stored', type = str)
    parser.add_argument('--style_dir', '--sd', help = 'Where your style images are stored', type = str)
    parser.add_argument('--style_weight', '--sw', help = 'Weight of style image', type = int)
    parser.add_argument('--content_weight', '--cw', help = 'Weight of content image', type = int)
    parser.add_argument('--image_size', '--sz', help = 'Weight of content image', type = int)
    args = parser.parse_args()
    
    device = args.device if args.device else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is currenly not enabled on your device')
        exit()
    content_dir = args.content_dir if args.content_dir else '../images/contents'
    style_dir = args.style_dir if args.style_dir else '../images/styles'
    output_dir = args.output_dir if args.output_dir else '../images/output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    iterations = args.iterations if args.iterations else 500
    content_weight = args.content_weight if args.content_weight else 1
    style_weight = args.style_weight if args.style_weight else 1000000
    image_size = args.image_size if args.image_size else 512
    content = args.content if args.content else 'city.png'
    style = args.style if args.style else 'van_gough.png'
    output = args.output if args.output else '{}+{}.png'.format(os.path.splitext(content)[0], os.path.splitext(style)[0])
 #--------------------------------------------------------------------------------------------------------------------------
    
    # vgg19 network as used in the paper
    vgg19 = models.vgg19(pretrained = True).features.to(device).eval()
    # Default vgg19 network normalization 
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    
    # Several transformations to perform on the input image
    transform = transforms.Compose([transforms.ToTensor()])
    # Get images for style transfer
    style_img = to_tensor(os.path.join(style_dir, style), image_size, transform).to(device)
    content_img = to_tensor(os.path.join(content_dir, content), image_size, transform).to(device)
    input_img = to_tensor(os.path.join(content_dir, content), image_size, transform).to(device)
    
    # Layers to extract content and style information
    content_layers = ['conv_4_2']
    style_layers = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
    
    output_img = style_transfer(orig_model = vgg19,
                                mean = vgg_mean, 
                                std = vgg_std,
                                input_img = input_img,
                                content_img = content_img,
                                style_img = style_img,
                                content_layers = content_layers,
                                style_layers = style_layers,
                                iterations = iterations,
                                content_weight = content_weight,
                                style_weight = style_weight)
    
    # Saves tensor as image [(0,1)->(0,255)]
    save_image(output_img, os.path.join(output_dir, output))