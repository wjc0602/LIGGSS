from jittor import nn
from models.networks.normalization import SPADE
from models.networks.spectral_norm import spectral_norm
from models.networks.vision_transformer import vit_base_patch16_224_in21k, vit_base_patch16_384


class ResidualBlock(nn.Module):
    def __init__(self, dim, ks=3):
        super(ResidualBlock, self).__init__()
        self.relu = nn.PReLU()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1)),
            nn.BatchNorm2d(dim),
            self.relu,
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1)),
            nn.BatchNorm2d(dim),
        )

    def execute(self, x):
        out = self.relu(x + self.model(x))
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def execute(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return nn.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def execute(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# # VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.model = vit_base_patch16_224_in21k()
        self.input_size = self.model.patch_embed.img_size

    def execute(self, X):
        X = nn.interpolate(X, size=self.input_size, mode='bilinear')
        out = self.model(X)
        return out

# VGG architecter, used for the perceptual loss using a pretrained VGG network

# class VGG19(nn.Module):
#     def __init__(self, requires_grad=False):
#         super().__init__()
#         vgg_pretrained_features = models.vgg19(pretrained=True).features
#         self.slice1 = nn.Sequential()
#         self.slice2 = nn.Sequential()
#         self.slice3 = nn.Sequential()
#         self.slice4 = nn.Sequential()
#         self.slice5 = nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def execute(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out
