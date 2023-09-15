import jittor as jt
from jittor import init
from jittor import nn
import jittor.models as model


def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()


def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class Unet(nn.Module):
    def __init__(self, input_nc=3, output_nc=29, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def execute(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose(inner_nc, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def execute(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return jt.contrib.concat([x, self.model(x)], 1)


if __name__ == '__main__':
    import jittor

    model = Unet(3, 29, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
    inp = jittor.ones(1, 3, 384, 512)
    out_put = model(inp)  # [1, 920, 8, 8]
    print(out_put.shape)
