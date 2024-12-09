import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
)
def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

class PoseNet(nn.Module):
    """
    Pose estimation network to predict 6-DoF poses for source images relative to the target.

    Args:
        input_channels: Number of input channels (target image + source image stack).
    Returns:
        pose_final: Predicted 6-DoF poses for source images relative to the target.
                    Shape: [batch_size, num_source, 6]
    """
    def __init__(self, input_channels):
        super(PoseNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.pose_pred = nn.Conv2d(256, 6, kernel_size=1, stride=1)

        self.num_source = None  # Will be set based on input dimensions
        

    def forward(self, tgt_image, src_image_stack):
        # Concatenate target and source images along the channel axis
        inputs = torch.cat((tgt_image, src_image_stack), dim=1)
        # self.num_source = src_image_stack.shape[1] // (3 * tgt_image.shape[1])

        # Forward pass through convolutional layers
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Predict 6-DoF poses
        pose_pred = self.pose_pred(x)  # Shape: [batch_size, 6 * num_source, H, W]

        # Average spatial dimensions and reshape
        pose_avg = torch.mean(pose_pred, dim=(2, 3))  # Shape: [batch_size, 6 * num_source]
        # print(pose_avg.shape)
        # print(pose_avg)
        pose_final = 0.01 * pose_avg #.view(-1, self.num_source, 6)  # Shape: [batch_size, num_source, 6]

        return pose_final
    

#jiwoo
import torch.nn as nn
class DepthDecoder(nn.Module):
    def __init__(self, input_dim=384, output_size=(224, 224)):
        super(DepthDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
        self.upsample = nn.Upsample(size=output_size, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.decoder(x)
        depth_map = self.upsample(x)
        # depth_map.squeeze_(dim=1)
        return depth_map
    
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = DepthDecoder(input_dim=384, output_size=(224, 224))
        self.posenet = PoseNet(6)
        
    def forward(self, sample):
        pose_final = self.posenet(sample['image_t']['processed_image'], 
                                  sample['image_t1']['processed_image'])
        # print(sample['image_t']['feature_embedding'].shape)
        depth_map = self.decoder(sample['image_t']['feature_embedding'])
        depth_map = [depth_map]
        return pose_final, depth_map
    
class BiggerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = DecoderBigNet()
        self.posenet = PoseNet(6)
        
    def forward(self, sample):
        pose_final = self.posenet(sample['image_t']['processed_image'], 
                                  sample['image_t1']['processed_image'])
        # print(sample['image_t']['feature_embedding'].shape)
        depth_map = self.decoder(sample['image_t']['feature_embedding'],
                                 sample['image_t']['processed_image'])
        depth_map = [depth_map]
        return pose_final, depth_map
    
class BigModelUnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = UNet()
        self.posenet = PoseNet(6)
        
    def forward(self, sample):
        pose_final = self.posenet(sample['image_t']['processed_image'], 
                                  sample['image_t1']['processed_image'])
        # print(sample['image_t']['feature_embedding'].shape)
        # depth_map = self.decoder(sample['image_t']['feature_embedding'],
        #                          sample['image_t']['processed_image'])
        depth_map = self.decoder(sample['image_t']['processed_image'])
        depth_map = [depth_map]
        return pose_final, depth_map
    
#Panwa
class DecoderBigNet(nn.Module):
    def __init__(self):
        super(DecoderBigNet, self).__init__()
        
        # Define the upsampling and convolutional stages
        self.upsample1 = upsample(384, 192)
        self.conv1 = add_conv_stage(192, 96)
        self.upsample2 = upsample(96, 48)
        self.conv2 = add_conv_stage(48, 24)
        self.upsample3 = upsample(24, 12)
        self.conv3 = add_conv_stage(12, 6)
        self.upsample4 = upsample(6, 3)
        
        # Refinement stages
        self.conv_refine = add_conv_stage(6, 6)
        self.conv4 = add_conv_stage(6, 3)
        self.conv_final = add_conv_stage(3, 1)

    def forward(self, feature_embedding, processed_image):
        # Upsampling and convolutional stages
        out = self.upsample1(feature_embedding)  # Add batch dimension
        out = self.conv1(out)
        out = self.upsample2(out)
        out = self.conv2(out)
        out = self.upsample3(out)
        out = self.conv3(out)
        out = self.upsample4(out)
        
        # Concatenate with the image
        out = torch.cat([out, processed_image], dim=1)  # Concatenate along channels
        
        # Refinement stages
        out = self.conv_refine(out)
        out = self.conv4(out)
        out = self.conv_final(out)
        
        return out


import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Define the encoder (downsampling path)
        self.enc_conv1 = add_conv_stage(3, 64)  # Input: [B, 3, 244, 244], Output: [B, 64, 244, 244]
        self.enc_conv2 = add_conv_stage(64, 128)  # Downsample next layer
        self.enc_conv3 = add_conv_stage(128, 256)
        self.enc_conv4 = add_conv_stage(256, 512)
        self.enc_conv5 = add_conv_stage(512, 1024)
        
        # Define the decoder (upsampling path)
        self.upsample5 = upsample(1024, 512)
        self.dec_conv5 = add_conv_stage(1024, 512)  # Skip connection adds 512

        self.upsample4 = upsample(512, 256)
        self.dec_conv4 = add_conv_stage(512, 256)  # Skip connection adds 256

        self.upsample3 = upsample(256, 128)
        self.dec_conv3 = add_conv_stage(256, 128)  # Skip connection adds 128

        self.upsample2 = upsample(128, 64)
        self.dec_conv2 = add_conv_stage(128, 64)  # Skip connection adds 64

        self.dec_conv1 = add_conv_stage(64, 32)  # No skip connection here
        self.dec_conv0 = add_conv_stage(32,16)
        self.dec_conv_min1 = add_conv_stage(16,8)
        self.dec_conv_min2 = add_conv_stage(8,4)
        self.dec_conv_min3 = add_conv_stage(4,1)
        


    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)  # [B, 64, 244, 244]
        e2 = self.enc_conv2(F.max_pool2d(e1, 2))  # [B, 128, 122, 122]
        e3 = self.enc_conv3(F.max_pool2d(e2, 2))  # [B, 256, 61, 61]
        e4 = self.enc_conv4(F.max_pool2d(e3, 2))  # [B, 512, 30, 30]
        e5 = self.enc_conv5(F.max_pool2d(e4, 2))  # [B, 1024, 15, 15]

        # Decoder
        d4 = self.upsample5(e5)  # [B, 512, 30, 30]
        d4 = self.dec_conv5(torch.cat([d4, e4], dim=1))  # Concatenate encoder feature map (skip connection)

        d3 = self.upsample4(d4)  # [B, 256, 61, 61]
        d3 = self.dec_conv4(torch.cat([d3, e3], dim=1))  # Skip connection

        d2 = self.upsample3(d3)  # [B, 128, 122, 122]
        d2 = self.dec_conv3(torch.cat([d2, e2], dim=1))  # Skip connection

        d1 = self.upsample2(d2)  # [B, 64, 244, 244]
        d1 = self.dec_conv2(torch.cat([d1, e1], dim=1))  # Skip connection
        
        out = self.dec_conv1(d1)
        out = self.dec_conv0(out)
        out = self.dec_conv_min1(out)
        out = self.dec_conv_min2(out)
        out = self.dec_conv_min3(out)
        

        return out
