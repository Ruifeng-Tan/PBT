import torch
import torch.nn as nn
from layers.nn_model import NNModel


class ConvModule(nn.Module):
    def __init__(self, din, dout, kernel_size,
                 act_fn: str = 'relu',
                 dropout: float = 0.1):
        nn.Module.__init__(self)
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(din, dout, kernel_size)
        self.pool1 = nn.AvgPool2d(kernel_size)
        self.conv2 = nn.Conv2d(dout, dout, kernel_size)
        self.pool2 = nn.AvgPool2d(kernel_size)
        self.act_fn = getattr(torch, act_fn)
        self.dropout = nn.Dropout2d(dropout)

    def output_shape(self, H, W):
        # conv1 output
        H, W = H - self.kernel_size[0] + 1, W - self.kernel_size[1] + 1
        # pool1 output
        H = int((H - self.kernel_size[0]) / self.kernel_size[0] + 1)
        W = int((W - self.kernel_size[1]) / self.kernel_size[1] + 1)
        # conv2 output
        H, W = H - self.kernel_size[0] + 1, W - self.kernel_size[1] + 1
        # pool2 output
        H = int((H - self.kernel_size[0]) / self.kernel_size[0] + 1)
        W = int((W - self.kernel_size[1]) / self.kernel_size[1] + 1)

        return H, W

    def forward(self, x):
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.pool2(x)

        return x


class Model(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 kernel_size: int = 1,
                 diff_base: int = 10,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 accumulation_steps: int = 1,
                 support_size: int = 1,
                 lr: float = 1e-3,
                 act_fn: str = 'relu',
                 filter_cycles: bool = True,
                 features_to_drop: list = None,
                 cycles_to_drop: list = None,
                 return_pointwise_predictions: bool = False,
                 seed: int = 0,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_height < kernel_size[0]:
            kernel_size = (input_height, kernel_size[1])
        if input_width < kernel_size[1]:
            kernel_size = (kernel_size[0], input_width)

        self.alpha = alpha
        self.channels = channels
        self.diff_base = diff_base
        self.train_support_size = train_support_size or support_size
        self.test_support_size = test_support_size or support_size
        self.grad_accum_steps = accumulation_steps
        self.filter_cycles = filter_cycles
        if isinstance(features_to_drop, int):
            features_to_drop = [features_to_drop]
        self.features_to_drop = features_to_drop
        if isinstance(cycles_to_drop, int):
            cycles_to_drop = [cycles_to_drop]
        self.cycles_to_drop = cycles_to_drop
        self.return_pointwise_predictions = return_pointwise_predictions

        self.ori_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.sup_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        # Shared regressor without bias
        self.fc = nn.Linear(channels, 1, bias=False)
        self.lr = lr
        self.seed = seed

    def forward(self,
               feature: torch.Tensor,
               label: torch.Tensor,
               support_feature: torch.Tensor,
               support_label: torch.Tensor,
               training: bool = False):
        B, S, C, H, W = support_feature.size()

        x_ori = self.ori_module(feature)
        x_sup = self.sup_module(support_feature.view(-1, C, H, W))

        y_ori = self.fc(x_ori.view(B, self.channels)).view(-1)
        y_sup = self.fc(x_sup.view(B, S, self.channels)).view(B, S)
        y_sup += support_label.view(B, S)

        if self.return_pointwise_predictions:
            return y_ori, y_sup

        if self.training:
            y_sup = y_sup.mean(1).view(-1)
        else:
            # We use median aggregation to minimize the influence of outliers
            y_sup = y_sup.median(1)[0].view(-1)
        
        loss = sum([(1. - self.alpha) * mse(y_ori, label), self.alpha * mse(y_sup, label)])
        outputs = (1. - self.alpha) * y_ori + self.alpha * y_sup

        return outputs, loss

def build_module(
    in_channels, channels, input_height, input_width, kernel_size, act_fn
) -> nn.Module:
    encoder = ConvModule(in_channels, channels, kernel_size, act_fn)
    H, W = encoder.output_shape(input_height, input_width)
    proj = nn.Conv2d(channels, channels, (H, W))
    return nn.Sequential(encoder, proj, nn.ReLU())


def mse(pred, label):
    return ((pred.view(-1) - label.view(-1)) ** 2).mean()
