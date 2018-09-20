import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

def to_tensor(image_name, imsize, transform):
    
    '''
    Transforms an np.array object (image) into a tensor
    '''
    
    image = cv2.imread(image_name)
    h, w, _ = image.shape # height, width and color channels. Note cv2 reads in
                          # color channels as BGR (not RGB)
                          
    center_h, center_w = int(0.5 * h), int(0.5 * w)
    crop = int(0.5 * min(h, w)) # get crop width
    
    image = image[center_h - crop : center_h + crop, 
                  center_w - crop : center_w + crop, 
                  (2, 1, 0)] # Center crop and swap BGR channels -> RGB
    
    resized = cv2.resize(image, (imsize, imsize), interpolation = cv2.INTER_CUBIC)
    image = transform(resized).unsqueeze(0) # unsqueeze: add a batch channel
                                            # np.array: H * W * C  ---->
                                            # ----> pytorch tensor 1 * C * H * W
    return image

class Normalize(nn.Module):
    '''
    Normalization module to conform with vgg network's default normalization
    '''
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # mean, std are both 3-channels tensors (RGB)
        self.mean = mean
        self.std = std

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    '''
    Module to compute content loss
    '''
    def __init__(self, target_content):
        super(ContentLoss, self).__init__()
        self.target_content = target_content.detach() # target_content is fixed value. No need to compute grad.

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target_content)
        return input

def gram_matrix(input):
    '''
    Compute gram matrix of an image (style representation)
    input: a tensor of size 1 * C * H * W
    '''
    n, c, h, w = input.shape
    F = input.view(n * c, h * w) # feature vector definition
    G = torch.mm(F, F.t()) # matrix multiplication 
    return G.div(n * c * h * w) # Normalize

class StyleLoss(nn.Module):
    '''
    Module to compute style loss
    '''
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_style = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target_style)
        return input

class StyleTransferNet(nn.Module):
    def __init__(self, orig_model, mean, std, 
                 content_img, style_img, 
                 content_layers, style_layers):
        
        super(StyleTransferNet, self).__init__()
        
        self.orig_model = orig_model
        
        self.content_img = content_img
        self.content_layers = content_layers
        self.content_losses = [] # Stores the content layers
        
        self.style_img = style_img
        self.style_layers = style_layers
        self.style_losses = [] # Stores the style layers
        
        self.mean = mean
        self.std = std
        
        self.net = nn.Sequential()
        self.construct_network()
        
    def construct_network(self):
        self.net.add_module('normalization', Normalize(self.mean, self.std))
        
        # Extracting layers from the original network (eg. vgg19)
        name = None
        conv_block, block_layer = 1, 1
        for layer in self.orig_model.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(conv_block, block_layer)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(conv_block, block_layer)
                layer = nn.ReLU(inplace = False)
                block_layer += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(conv_block)
                conv_block += 1
                block_layer = 1
    
            self.net.add_module(name, layer)
    
            if name in self.content_layers:
                target_content = self.net(self.content_img).detach()
                content_loss = ContentLoss(target_content)
                self.net.add_module("content_loss_{}".format(conv_block), content_loss)
                self.content_losses.append(content_loss)
    
            if name in self.style_layers:
                target_feature = self.net(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.net.add_module("style_loss_{}".format(conv_block), style_loss)
                self.style_losses.append(style_loss)
                
    def forward(self, input):
        return self.net(input)

def style_transfer(orig_model, mean, std, 
                   input_img, content_img, style_img, 
                   content_layers, style_layers, 
                   iterations, content_weight, style_weight):
    
    net = StyleTransferNet(orig_model, mean, std, 
                           content_img, style_img,
                           content_layers, style_layers)
    
    input_img.requires_grad = True
    optimizer = optim.LBFGS([input_img])
    
    it = 0
    while it <= iterations:
        def closure():
            nonlocal it # closure variable
            optimizer.zero_grad() # Clear the previous gradients
            
            input_img.data.clamp_(0, 1)
            net(input_img)
            
            content_loss, style_loss = 0, 0
            
            for content_layer in net.content_losses:
                content_loss += content_layer.loss
            content_loss /= len(net.content_losses) # Each content layer contributes
                                                    # equally to the content loss
            
            for style_layer in net.style_losses:
                style_loss += style_layer.loss
            style_loss /= len(net.style_losses) # Each style layer contributes
                                                # equally to the style loss
                                                
            content_loss *= content_weight
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()

            it += 1
            if it % 50 == 0:
                print('Iteration {} --> Content Loss : {:4f}, Style Loss: {:4f}'.format(
                    it, content_loss.item(), style_loss.item()))
            return total_loss
        optimizer.step(closure)
        
    input_img.data.clamp_(0, 1)
    return input_img
