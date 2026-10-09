import numpy as np
import torch
from torchvision import transforms
from tqdm import trange
from torch import autocast
from torch.cuda.amp import GradScaler
from pytorch_forecasting.utils import unsqueeze_like
import kornia

INF = float("inf")


class Filter(torch.nn.Module):
    def __init__(
        self,
        kernels,
        shape,
        box=(1, 1),
        sigma_color=0.1,
        norm=1,
        pad_mode="reflect",
        filter_mode=False,
        loss_factor=1,
        loss_norm=2,
    ):
        super().__init__()
        self.norm, self.sigma_color, self.pad_mode, self.box, self.filter_mode = (
            norm,
            sigma_color,
            pad_mode,
            box,
            filter_mode,
        )
        self.kernels = torch.nn.ParameterList(
            [torch.nn.Parameter(self.__get_init_w(kernel, shape)) for kernel in kernels]
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss_factor = loss_factor
        self.loss_norm = loss_norm

    def pad_w(self, w):
        return torch.nn.functional.pad(
            w, (0, w.shape[-1] - 1, 0, w.shape[-2] - 1), "reflect"
        )

    def __get_init_w(self, kernel, shape):
        repeats, _, h, w = shape
        box = self.box if self.box is not None else kernel
        boxes = [int(np.ceil(h / box[0])), int(np.ceil(w / box[1]))]
        num_boxes = boxes[0] * boxes[1]
        w = (
            kornia.filters.get_gaussian_kernel2d(kernel, torch.tensor([[0.2, 0.2]]))
            .unsqueeze(0)
            .repeat(repeats, num_boxes, 1, 1)
        )
        return (
            w[..., : int(kernel[0] // 2) + 1, : int(kernel[1] // 2) + 1]
            .clamp(1e-5, 0.999999)
            .log()
        )

    def get_dist(self, x, kernel, guidance=None, norm=None):
        norm = self.norm if norm is None else norm
        unf_inp = self.extract_patches(x, kernel)
        guidance = guidance if guidance is not None else x
        guidance = torch.nn.functional.pad(
            guidance,
            self._box_pad(guidance, kernel),
            mode=self.pad_mode,
        )

        return torch.pow(
            torch.norm(
                unf_inp
                - guidance.view(guidance.shape[0], guidance.shape[1], -1)
                .transpose(1, 2)
                .view(
                    guidance.shape[0],
                    unf_inp.shape[1],
                    unf_inp.shape[2],
                    guidance.shape[1],
                    1,
                ),
                p=norm,
                dim=-2,
                keepdim=True,
            ),
            2,
        )

    def __get_color_kernel(self, guidance, kernel):
        if self.sigma_color <= 0:
            return 1
        dist = self.get_dist(guidance.double(), kernel).float()
        ret = (
            (-0.5 / (self.sigma_color**2) * dist)
            .exp()
            .view(guidance.shape[0], dist.shape[1], dist.shape[2], -1, 1)
        )
        return torch.nan_to_num(ret, nan=0.0)

    def _box_pad(self, x, kernel):
        box = self.box if self.box is not None else kernel
        col = (
            box[1] - (x.shape[-1] - (x.shape[-1] // box[1]) * box[1]) % box[1]
        ) % box[1]
        row = (
            box[0] - (x.shape[-2] - (x.shape[-2] // box[0]) * box[0]) % box[0]
        ) % box[0]
        return [0, col, 0, row]

    def _kernel_pad(self, kernel):
        return [
            (kernel[1] - 1) // 2,
            (kernel[1] - 1) - (kernel[1] - 1) // 2,
            (kernel[0] - 1) // 2,
            (kernel[0] - 1) - (kernel[0] - 1) // 2,
        ]

    def _median_pad(self, x, kernel, stride):
        ph = (
            (kernel[0] - stride[0])
            if x.shape[-2] % stride[0] == 0
            else (kernel[0] - (x.shape[-2] % stride[0]))
        )
        pw = (
            (kernel[1] - stride[1])
            if x.shape[-1] % stride[1] == 0
            else (kernel[1] - (x.shape[-1] % stride[1]))
        )
        return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)

    def _compute_median(self, x, kernel):
        stride = kernel if not self.filter_mode else (1, 1)
        x_p = torch.nn.functional.pad(
            x, self._median_pad(x, kernel, stride), mode="reflect"
        )
        x_unf = x_p.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
        median = x_unf.contiguous().view(x_unf.size()[:4] + (-1,)).median(dim=-1)[0]
        return (
            median.unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(
                1,
                1,
                1,
                x_p.shape[-2] // median.shape[-2],
                1,
                x_p.shape[-1] // median.shape[-1],
            )
            .flatten(2, 3)
            .flatten(-2)[..., : x.shape[-2], : x.shape[-1]]
        )

    def extract_patches(self, x, kernel):
        box = self.box if self.box is not None else kernel
        kern = (box[0] + (kernel[0] - 1), box[1] + (kernel[1] - 1))
        pad = [
            b + k for b, k in zip(self._box_pad(x, kernel), self._kernel_pad(kernel))
        ]
        inp_unf = (
            torch.nn.functional.pad(x, pad, mode=self.pad_mode)
            .unfold(2, kern[0], box[0])
            .unfold(3, kern[1], box[1])
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(-1, x.shape[1], kern[0], kern[1])
        )

        return (
            inp_unf.unfold(2, kernel[0], 1)
            .unfold(3, kernel[1], 1)
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(
                x.shape[0],
                inp_unf.shape[0] // x.shape[0],
                -1,
                inp_unf.shape[1],
                kernel[0] * kernel[1],
            )
        )

    def __compute_filter_loss(self, x, kernel, norm=2):
        return self.get_dist(
            x, kernel, guidance=self._compute_median(x, kernel), norm=norm
        ).view(x.shape[0], -1).sum(-1, keepdims=True) / torch.prod(
            torch.tensor(x.shape[1:])
        )

    def __apply_filter(self, x, w, guidance=None):
        w = self.pad_w(w)
        kernel = (w.shape[-2], w.shape[-1])
        box = self.box if self.box is not None else kernel
        inp_unf = self.extract_patches(x, kernel)

        boxes = [
            int(np.ceil(x.shape[-2] / box[0])),
            int(np.ceil(x.shape[-1] / box[1])),
        ]
        color_kernel = self.__get_color_kernel(guidance, kernel)
        w = (
            self.softmax(w.view(w.shape[0], w.shape[1], -1))
            .unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(1, 1, inp_unf.shape[2], 1, 1)
            * color_kernel
        ).view(w.shape[0], w.shape[1], inp_unf.shape[2], -1, 1)
        out = inp_unf.matmul(w).transpose(2, 3).squeeze(-1) / w.squeeze(-1).sum(
            -1
        ).unsqueeze(2)
        out = (
            out.view(-1, inp_unf.shape[-2], inp_unf.shape[-3])
            .reshape(x.shape[0], -1, x.shape[1] * box[0] * box[1])
            .transpose(2, 1)
        )
        return torch.nn.functional.fold(
            out,
            (boxes[0] * box[0], boxes[1] * box[1]),
            box,
            stride=box,
        )[..., : x.shape[-2], : x.shape[-1]]

    def compute_loss(self, x):
        kernels = [(f.shape[-2] * 2 - 1, f.shape[-1] * 2 - 1) for f in self.kernels]
        if len(kernels) == 0 or self.loss_factor == 0:
            return 0
        return torch.concat(
            [
                self.__compute_filter_loss(
                    x,
                    k,
                    norm=self.loss_norm,
                )
                for k in kernels
            ],
            dim=-1,
        ).sum(-1)

    def forward(self, x, guidance):
        for filt in self.kernels:
            x = self.__apply_filter(x, filt, guidance=guidance)
        return x.float()


class StepPerformer:
    def __init__(
        self,
        comparator,
        lr,
        optimizer_class="Adam",
        tanh_space=True,
        schduler_class=None,
        regularization_type=None,
        regularization_factor=0.0,
        regularization_threshold=None,
        max_grad_l_inf=2.0,
        lr_decay_threshold=2.0,
        lr_decay_factor=0.5,
        scale_mode="fp32",
        device="cuda",
    ):
        assert isinstance(optimizer_class, str)
        assert schduler_class is None or isinstance(schduler_class, str)
        assert regularization_type is None or regularization_type in ["l1", "l2"]
        if regularization_type is not None:
            assert (
                isinstance(regularization_factor, float) and regularization_factor >= 0
            )
            assert regularization_threshold is None or isinstance(
                regularization_threshold, float
            )
        assert max_grad_l_inf is None or (
            isinstance(max_grad_l_inf, float) and max_grad_l_inf > 0
        )
        if schduler_class is not None:
            assert (
                isinstance(lr_decay_factor, float)
                and lr_decay_factor >= 0
                and lr_decay_factor <= 1
            )
            assert isinstance(lr_decay_threshold, float)
        assert scale_mode in ["fp32", "fp16"]
        optimizer = eval("torch.optim." + optimizer_class)
        scheduler = (
            eval("torch.optim.lr_scheduler." + schduler_class)
            if schduler_class is not None
            else None
        )

        self.lr = lr
        self.scale_mode = scale_mode
        self.comparator = comparator
        self.optimizer_class = optimizer
        self.schduler_class = scheduler
        self.max_grad_l_inf = max_grad_l_inf
        self.reg_factor = regularization_factor
        self.reg_norm = (
            None
            if regularization_type is None
            else (1 if regularization_type == "l1" else 2)
        )
        self.lr_decay_threshold = lr_decay_threshold
        self.lr_decay_factor = lr_decay_factor
        self.device = device
        self.tanh_space = tanh_space
        self.regularization_threshold = regularization_threshold

    def reset(self, modifier):
        self.scaler = None if self.scale_mode == "fp32" else GradScaler()
        self.optimizer = self.optimizer_class(
            [
                {"params": m, "lr": self.lr[i]} for i, m in enumerate(modifier)
            ],  # betas=(0.5, 0.5)
        )
        if self.schduler_class is not None:
            self.scheduler = self.schduler_class(
                self.optimizer,
                mode="max",
                threshold=self.lr_decay_threshold,
                factor=self.lr_decay_factor,
            )
        else:
            self.scheduler = None

    def __run(self, x, modifier, y, ox, const, filt):
        if self.scaler is not None:
            with autocast(device_type=self.device, dtype=torch.float16):
                return self.__get_outputs(x, modifier, y, ox, const, filt)
        return self.__get_outputs(x, modifier, y, ox, const, filt)

    def backward(self, loss):
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_l_inf is not None:
            for i in range(len(self.optimizer.param_groups)):
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[i]["params"], self.max_grad_l_inf
                )

    def step(self):
        if self.scaler is None:
            self.optimizer.step()
            return
        self.scaler.step(self.optimizer)

    def update(self):
        if self.scaler is not None:
            self.scaler.update()

    def __call__(self, x, modifier, y, ox, const, filt):
        self.optimizer.zero_grad()
        outs = self.__run(x, modifier, y, ox, const, filt)
        self.backward(outs[-1])
        self.step()
        self.update()
        if self.scheduler is not None:
            self.scheduler.step(outs[-4].mean())
        return outs[:-1]

    def loss_fn(self, x, y):
        ret = self.comparator.model_fn(x, y).mean(-1)
        idx = 1 - self.comparator(ret.detach(), None).float()
        return ret, idx

    def to_range(self, x):
        return (
            x * (self.comparator.clip_max - self.comparator.clip_min)
            + self.comparator.clip_min
        )

    def tanh(self, x):
        if not self.tanh_space:
            return self.to_range(x).clamp(0, 1)
        return self.to_range((torch.tanh(x) + 1) / 2)

    def norm_loss(self, x, y, norm):
        return (
            torch.pow(
                torch.norm(x - y, p=norm, dim=tuple(list(range(1, len((x).shape))))),
                1,
            )
        ).view(x.shape[0], -1)

    def __get_outputs(self, x, modifier, y, ox, const, filt):
        new_x = filt(self.tanh(modifier + x), y)
        log_loss, log_idx = self.loss_fn(new_x, y)
        l = log_loss if not self.comparator.flip else -log_loss
        filter_loss = filt.compute_loss(new_x)
        dist = self.comparator.dist_fn(new_x, ox).view(-1)
        reg = (
            self.norm_loss(new_x, y, self.reg_norm)
            if (self.reg_norm is not None and self.reg_factor > 0)
            else 0
        )
        normalized_reg = (
            reg.detach() / torch.prod(torch.tensor(x.shape[1:]))
            if (self.reg_norm is not None and self.reg_factor > 0)
            else 0
        )
        if (
            self.regularization_threshold is not None
            and self.reg_norm is not None
            and self.reg_factor > 0
        ):
            reg_idx = 1 - (normalized_reg < self.regularization_threshold).float()
            reg = reg * reg_idx
        loss = (log_idx * l * const - dist + filter_loss + reg * self.reg_factor).sum()
        detached_filter_loss = (
            filter_loss.detach() if not isinstance(filter_loss, int) else filter_loss
        )
        return (
            new_x,
            log_loss.detach(),
            dist.detach(),
            normalized_reg,
            detached_filter_loss,
            loss,
        )


class Bar:
    def __init__(
        self,
        itrs,
        binary_search_steps,
        evalu,
        with_filter_loss=True,
        with_reg=True,
        eval_interval=1,
        verbose=False,
        stage_name="high_freq",
    ):
        self.itrs = itrs
        self.pbar = trange(self.itrs) if verbose else None
        self.binary_search_steps = binary_search_steps
        self.step = 0
        self.evalu = evalu
        self.eval_interval = eval_interval
        self.itr = 0
        self.last_acc = None
        self.last_succ_rate = None
        self.with_filter_loss = with_filter_loss
        self.with_reg = with_reg
        self.stage_name = stage_name
        if self.pbar:
            self.pbar.set_description(f"UnMarker-{stage_name}")

    def reset(self):
        self.last_acc = None
        self.last_succ_rate = None
        self.itr = 0
        self.step += 1
        if self.pbar is not None:
            self.pbar.reset()

    def global_reset(self):
        self.reset()
        self.step = 0
        if self.pbar:
            self.pbar.set_description(f"UnMarker-{self.stage_name}")

    def get_eval_stats(self, x):
        if self.evalu is None or self.itr % self.eval_interval != 0:
            return None, None

        accs = self.evalu(x)
        mean_acc = accs.mean().round(decimals=6).item()
        succ_rate = (
            1 - self.evalu.is_detected(accs).astype(np.float32).round(decimals=3).mean()
        )
        return mean_acc, succ_rate

    def update(self, bestl2, x, logits, dist, filter_loss, reg_loss):
        self.itr += 1
        if self.pbar is None:
            return
        bestl2 = torch.round(bestl2.detach().mean(), decimals=6).detach().item()
        dist = torch.round(dist.detach().mean(), decimals=6).detach().item()
        loss = torch.round(logits.mean(), decimals=6).detach().item()

        Message = f"UnMarker-{self.stage_name} - Step {self.step}/{self.binary_search_steps}, best loss: {round(bestl2, 6)}, curr loss: {round(dist, 6)}, dist: {round(loss, 6)}"
        if self.with_reg:
            reg_loss = (
                torch.round(reg_loss.mean(), decimals=6).item()
                if not isinstance(reg_loss, int)
                else reg_loss
            )
            Message = Message + f", reg_loss: {round(reg_loss, 6)}"
        if self.with_filter_loss:
            filter_loss = (
                torch.round(filter_loss.mean(), decimals=6).item()
                if not isinstance(filter_loss, int)
                else filter_loss
            )
            Message = Message + f", filter_loss: {round(filter_loss, 6)}"
        mean_acc, succ_rate = self.get_eval_stats(x)
        if mean_acc is not None:
            self.last_acc = mean_acc
        if succ_rate is not None:
            self.last_succ_rate = succ_rate
        if mean_acc is None:
            mean_acc = self.last_acc
        if succ_rate is None:
            succ_rate = self.last_succ_rate
        Message = (
            (
                Message
                + f", detection acc: {round(mean_acc, 6)}, attack_success: {round(succ_rate, 6)}"
            )
            if mean_acc is not None
            else Message
        )
        self.pbar.set_description(Message)
        self.pbar.update(1)


class Comp:
    def __init__(
        self,
        model_fn,
        dist_fn,
        clip_min=0.0,
        clip_max=1.0,
    ):
        self.model_fn = model_fn
        self.dist_fn = dist_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def reset(self, thresh=1e-3, flip=False):
        self.thresh = thresh
        self.flip = flip

    def __call__(self, pred, label):
        return (pred < self.thresh) if not self.flip else (pred >= self.thresh)


class SpecialCWCoordinate:
    def __init__(
        self,
        est,
        dist_fn,
        evalu,
        optimizer_args,
        filter_args=None,
        modifier_type="RGB",
        max_iterations=1000,
        binary_search_steps=20,
        initial_const=1e-2,
        clip_min=0.0,
        clip_max=1.0,
        tanh_space=True,
        progress_bar_args=None,
        device="cuda",
    ):
        filter_args = {} if filter_args is None else filter_args
        assert isinstance(filter_args, dict)
        kernels = filter_args.get("kernels")
        box = filter_args.get("box", (1, 1))
        assert (
            optimizer_args.get("learning_rate") is not None
            and isinstance(optimizer_args["learning_rate"], dict)
            and optimizer_args["learning_rate"].get("values") is not None
        )
        learning_rates = optimizer_args["learning_rate"]["values"]
        assert modifier_type in ["RGB", "LA"]
        self.rgb = True if modifier_type == "RGB" else False
        self.tanh_space = tanh_space

        self.kernels = [] if kernels is None else kernels
        self.__check_params(self.kernels, box, learning_rates)
        if "kernels" in list(filter_args.keys()):
            del filter_args["kernels"]
        self.initial_const = float(initial_const)
        self.filter_args = filter_args
        scheduler_args = optimizer_args["learning_rate"].get("scheduler", {})
        scheduler_args = {} if scheduler_args is None else scheduler_args
        reg_args = optimizer_args.get("regularization", {})
        reg_args = {} if reg_args is None else reg_args
        assert isinstance(reg_args, dict)

        optimizer_args = {
            "optimizer_class": optimizer_args.get("type", "Adam"),
            "schduler_class": scheduler_args.get("type"),
            "regularization_type": reg_args.get("type", None),
            "regularization_factor": reg_args.get("factor", 0.0),
            "regularization_threshold": reg_args.get("thresh"),
            "max_grad_l_inf": optimizer_args.get("max_grad_l_inf", 2.0),
            "lr_decay_threshold": scheduler_args.get("args", {}).get(
                "decay_threshold", 0.0
            ),
            "lr_decay_factor": scheduler_args.get("args", {}).get("decay_factor", 1.0),
            "scale_mode": optimizer_args.get("scale_mode", "fp32"),
        }

        self.comparator = Comp(
            est,
            dist_fn,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.loss_fn = StepPerformer(
            self.comparator,
            learning_rates,
            tanh_space=tanh_space,
            device=device,
            **optimizer_args,
        )
        progress_bar_args = {} if progress_bar_args is None else progress_bar_args
        assert isinstance(progress_bar_args, dict)
        # assert (
        #    isinstance(progress_bar_args, dict)
        #    and "eval_interval" in list(progress_bar_args.keys())
        #    and "verbose" in list(progress_bar_args.keys())
        #    and len(list(progress_bar_args.keys())) == 2
        # )
        with_filter_loss = (
            False
            if len(self.kernels) == 0 or filter_args.get("loss_factor", 1) == 0
            else True
        )
        with_reg = reg_args.get("type") is not None and reg_args.get("factor") > 0
        self.pbar = Bar(
            max_iterations,
            binary_search_steps,
            evalu,
            with_filter_loss=with_filter_loss,
            with_reg=with_reg,
            **progress_bar_args,
        )

    def __check_params(self, kernels, box, learning_rates):
        assert isinstance(kernels, list)
        for k in kernels:
            assert isinstance(k, tuple) and len(k) == 2
            for kx in k:
                assert isinstance(kx, int) and kx % 2 == 1
        if len(kernels) > 0:
            assert box is None or isinstance(box, tuple)
            if isinstance(box, tuple):
                assert len(box) == 2
                for bx in box:
                    assert isinstance(bx, int)
        assert (
            isinstance(learning_rates, list) and len(learning_rates) == len(kernels) + 1
        )
        for lr in learning_rates:
            assert isinstance(lr, float)

    def setup(self, x, y, ox=None, flip=False, thresh=1e-3, initial_const=None):
        self.orig_size = x.shape[-1]
        self.pbar.global_reset()
        self.comparator.reset(thresh=thresh, flip=flip)
        initial_const = self.initial_const if initial_const is None else initial_const
        self.lower_bound = torch.zeros((len(x))).float().to(x.device)
        self.upper_bound = torch.ones((len(x))).float().to(x.device) * 1e10
        self.const = x.new_ones(len(x)) * initial_const
        o_bestl2 = torch.ones((len(x))).float().to(x.device) * (-INF)
        x = torch.clamp(x, self.comparator.clip_min, self.comparator.clip_max)
        ox = x.clone().detach() if ox is None else ox.clone().detach()
        o_bestattack = y.clone().detach()
        x = (x - self.comparator.clip_min) / (
            self.comparator.clip_max - self.comparator.clip_min
        )
        filt = Filter(
            self.kernels,
            y.shape,
            **self.filter_args,
        ).to(x.device)
        if self.tanh_space:
            x = (
                torch.arctanh((torch.clamp(x, 0, 1) * 2 - 1) * 0.999999)
                .detach()
                .clone()
            )
            modifier = torch.zeros(
                x.shape if self.rgb else (x.shape[0], 1, x.shape[2], x.shape[3]),
                requires_grad=True,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            modifier = torch.rand(
                x.shape if self.rgb else (x.shape[0], 1, x.shape[2], x.shape[3]),
                requires_grad=True,
                device=x.device,
                dtype=x.dtype,
            )
            modifier = modifier.detach() * 2 - 1
            modifier = modifier * 0.001
            modifier.requires_grad = True

        self.loss_fn.reset(list(filt.parameters()) + [modifier])
        return o_bestl2, x, ox, o_bestattack, modifier, filt

    def update_const(self, bestscore):
        u_cond = torch.all(bestscore.view(bestscore.shape[0], -1) != -1, -1).float()
        self.upper_bound = torch.minimum(
            self.upper_bound, self.const
        ) * u_cond + self.upper_bound * (1 - u_cond)
        self.lower_bound = (
            torch.maximum(self.lower_bound, self.const) * (1 - u_cond)
            + self.lower_bound * u_cond
        )
        const_cond = (self.upper_bound < 1e9).float()
        self.const = (
            (self.lower_bound + self.upper_bound) / 2
        ) * const_cond + self.const * (
            10 * (1 - u_cond) * (1 - const_cond) + (1 - const_cond) * u_cond
        )

    def compare_dist(self, x, y):
        return x > y

    def __call__(self, x, y, ox=None, flip=False, thresh=1e-3, initial_const=None):
        o_bestl2, x, ox, o_bestattack, modifier, filt = self.setup(
            x, y, ox=ox, flip=flip, thresh=thresh, initial_const=initial_const
        )

        for outer_step in range(self.pbar.binary_search_steps):
            self.pbar.reset()
            bestl2 = torch.ones((len(x))).float().to(x.device) * (-INF)
            bestscore = torch.ones(len(x)).to(torch.float).to(x.device) * (-1.0)

            for i in range(self.pbar.itrs):
                new_x, pred, dist, reg_loss, filter_loss = self.loss_fn(
                    x, modifier, y, ox, self.const, filt
                )
                succeeded = self.comparator(pred, y)
                cond = torch.logical_and(self.compare_dist(dist, o_bestl2), succeeded)
                n_cond, cond = (
                    torch.logical_or(
                        cond,
                        torch.logical_and(self.compare_dist(dist, bestl2), succeeded),
                    ).float(),
                    cond.float(),
                )
                o_bestl2 = dist * cond + torch.nan_to_num(
                    o_bestl2 * (1 - cond), nan=0.0
                )
                o_bestattack = new_x * unsqueeze_like(cond, new_x) + o_bestattack * (
                    1 - unsqueeze_like(cond, new_x)
                )
                bestl2 = dist * n_cond + torch.nan_to_num(
                    bestl2 * (1 - n_cond), nan=0.0
                )
                bestscore = pred * n_cond + bestscore * (1 - n_cond)

                self.pbar.update(
                    bestl2, new_x.detach(), pred, dist, filter_loss, reg_loss
                )

            self.update_const(bestscore)

        return o_bestattack.detach()
