import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(pose_avg.shape)
        print(pose_avg)
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
        depth_map.squeeze_(dim=1)
        return depth_map
    
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = DepthDecoder()
        self.posenet = PoseNet(6)
        
    def forward(self, sample):
        pose_final = self.posenet(sample['image_t']['processed_image'], 
                                  sample['image_t1']['processed_image'])
        depth_map = self.decoder(sample['image_t']['feature_embedding'])
        return pose_final, depth_map