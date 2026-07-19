---
type: implementation
title: "EnerVerse-AC Code Walkthrough"
repository_url: "https://github.com/AgibotTech/EnerVerse-AC"
tested_commit: null
status: implemented
updated: 2026-07-19
---

# EnerVerse-AC 源码深读

> **版本边界：**原阅读记录未保存 commit SHA。本文保留对推理数据流和核心模块的详细解释；函数名或实现细节若与上游当前版本不同，应以上游固定 commit 为准。

## 阅读路线

```Plain Text
README
→ config.yaml
→ main/generate_video_acwm.py
→ ACWMLatentDiffusion.inference
→ ACWMLatentDiffusion.get_batch_input
→ ACWMLatentDiffusion.sample_log
→ DDIMSampler.sample / ddim_sampling
→ DiffusionWrapper.forward
→ UNet
→ attention / resampler
→ VAE / action / ray / traj
```

来组织。

README 和 config 只作为入口参考，正文从推理脚本开始。

## `generate_video_acwm.py`

这个文件主要由以下函数组成：

```Plain Text
def get_image(img_path, n)
def get_action_bias_std(domain_name)
def get_action_npy(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
)
def get_action_h5(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
)
def get_caminfo_npy(extrinsic_path, intrinsic_path, n)
def get_caminfo_json(extrinsic_path, intrinsic_path, n)
def main(args)
```

## def  get\_image:

该函数主要用于处理数据集里面的图片，代码块如下所示：

```Python
def get_image(img_path, n):
    img = np.array(Image.open(img_path))
    img = torch.from_numpy(img).float().permute(2,0,1)/255.0#这啥意思？/255.0是将像素值归一化到0-1范围，图片的形状是[H,W,C]，permute(2,0,1)将其转换为[C,H,W]，然后再转换为torch张量
    #加n是什么意思？n是视频帧数
    img = img.unsqueeze(1).repeat(1,n,1,1)
    return img
```

先导入图片，之后将图片转换成为\[H,n,W,C\]的形状张量。同时会进行视频帧数的repeat。

## def get\_action\_npy:

这个函数主要是获得action和delta\_action,如下所示：

```Python
def get_action_npy(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
):
    abs_act = np.load(action_path)

    if n_chunk > 0:
        if (abs_act.shape[0] < n_chunk * chunk + n_previous):
            raise ValueError(f"Num of Action Timestep {abs_act.shape[0]} smaller than {n_previous}+{n_chunk}*{n_chunk}")
        assert(abs_act.shape[1] == 16)
        abs_act = abs_act[:n_chunk*chunk+n_previous, :]#abs_act是一个二维数组，形状为[T, 16]，其中T是时间步数，16表示每个时间步的动作维度。这里的代码意思是，如果指定了n_chunk大于0，那么就检查abs_act的时间步数是否足够满足n_chunk * chunk + n_previous的要求。如果不满足，就抛出一个ValueError异常。然后，使用assert语句确保abs_act的第二维大小为16，即每个时间步的动作维度必须为16。最后，将abs_act截取为前n_chunk * chunk + n_previous个时间步的数据，以确保只保留所需的动作数据。

    action, delta_action = get_actions(#这是在干什么？get_actions函数的作用是从绝对动作数据中提取出动作和增量动作。它将绝对动作数据abs_act作为输入，并根据指定的参数进行处理，返回两个结果：action和delta_action。action表示原始的绝对动作数据，而delta_action表示相对于前一个时间步的增量动作数据。这些数据将用于后续的视频生成过程。
        gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
        all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
        all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),#这是什么语法？np.stack函数用于在指定轴上堆叠数组。这里将abs_act[:, 3:7]和abs_act[:, 11:15]在第1个轴上堆叠，形成一个新的数组。
        slices=None,
        delta_act_sidx=n_previous,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)

    delta_action[:, :6] = (delta_action[:, :6] - sep*delta_act_meanv[:, :6]) / (sep*delta_act_stdv[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep*delta_act_meanv[:, 6:]) / (sep*delta_act_stdv[:, 6:])
    return action, delta_action

```

首先确保我们的帧数要小于我们提供的action的数量，之后是调包提取动作和变化动作。

## def get\_action\_h5:

这个函数用于从H5文件中读取动作数据，并进行处理以获取动作和增量动作。它接受H5文件路径、块数、每块的大小、前一个时间步的数量、分隔符和域名作为参数。根据这些参数，它会解析H5文件中的动作数据，转换为PyTorch张量，并对增量动作进行归一化处理，最终返回处理后的动作和增量动作张量。和上一个函数的逻辑是一致的，不多赘述。

这里主要和npy的区别在于两者数据格式不一样，所以要用两个不同函数。

## def get\_caminfo\_npy和def get\_caminfo\_json:



## def main:

main函数主要是调用上述函数，其中顺序与上述一致，这里有几点需要注意，第一个函数n，即视频帧数，就是我们提到的动作的数量，action\.shape\[0\]\.最后加载模型，再调用model\.inference\.



## `ddpm3d`

这个文件包括了我们前面刚提到的众多模块，这里不按文件顺序看，按函数来看，详细说一下：

## `inference`

这个函数很长，按函数组织顺序来叙述一下功能：

首先进行图像预处理：

```Plain Text
sample_size = tuple(config.data.params.train.params.sample_size)
trans_resize = transforms.Compose([
    transforms.Resize(sample_size),
])
trans_norm = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])
```

这个函数主要是把图像处理到训练尺寸，接着归一化到\[\-1,1\]的区间上面。

这样之后会进行相机参数的预处理，然后重点在于截取动作和轨迹条件：

```Plain Text
action = action[:n_previous+num_chunk*chunk, :]
delta_action = delta_action[:num_chunk*chunk, :]
```

截取推理需要的动作长度：历史 `n_previous` 加未来 `num_chunk*chunk`。
`delta_action` 只取未来 chunk 对应长度。

之后对于trajectory有着相同的处理方式，trajectory是用于模型的控制条件之一。

然后进入核心循环：每个chunk的自回归生成：

```Plain Text
if i_chunk == 0:
    video = torch.cat((
        ori_video[:,:,:,:n_previous,:,:],
        ori_video[:,:,:,n_previous-1:n_previous].repeat(1,1,1,chunk,1,1)
    ), dim=3)

else:
    n_history = all_x_samples.shape[3]
    idx_history = [n_history*_//(n_previous-1) for _ in range(n_previous-1)]
    video = torch.cat((
        all_x_samples[:,:,:,idx_history],
        (x_samples[:,:,:,-1:]).repeat(1,1,1,chunk+1,1,1)
    ), dim=3)
```

这一段是比较好理解的，这就是采帧的逻辑：

第一个 chunk：
前 `n_previous` 帧是真实历史帧。
未来 `chunk` 帧先用最后一帧重复填充，占位给模型采样。
最后得到长度 `n_previous + chunk` 的输入视频。

后续 chunk：
从已生成历史 `all_x_samples` 中均匀抽 `n_previous-1` 帧作为长期历史。
再把上一个 chunk 最后一帧重复 `chunk+1` 次，作为当前条件末尾和未来占位。
这样模型每次都能看到一些历史帧，同时继续往后滚动。

这样之后我们要获得提示的轨迹，动作，相机：

```Plain Text
if i_chunk == 0:
    traj = ori_traj[:,:,:,i_chunk*chunk:(i_chunk+1)*chunk+n_previous]
    i_delta_action = ori_delta_action[:,i_chunk*chunk:(i_chunk+1)*chunk]
    i_c2w_list = c2w_list[:,:,i_chunk*chunk:(i_chunk+1)*chunk+n_previous]
else:
    traj = torch.cat((
        all_trajs[:,:,:,idx_history,:,:],
        ori_traj[:,:,:,i_chunk*chunk+n_previous-1:(i_chunk+1)*chunk+n_previous,:,:],
    ), dim=3)
```

之后会经过Classifier\-Free Guidance 的 unconditional 条件，如果 guidance scale 不等于 1，就需要构造 unconditional conditioning：

```Plain Text
img = torch.zeros_like(
    video[:,:,:,0,:,:]).view(-1, c, h, w).float().to(z.device)
uc_img_emb = self.embedder(img)
uc_img_emb = self.image_proj_model(uc_img_emb.to(dtype=z.dtype, device=z.device))
uc = torch.cat([
    uc_img_emb,
    c_emb[:, uc_img_emb.shape[1]:],
], dim=1)
```

先构造全0图像作为条件，然后和原来的文本/domain 等后半部分 embedding 拼起来。

启用concat条件：

```Plain Text
u_c_cat =  deepcopy(c_cat)
if self.use_cat_mask:
    u_c_cat[1][:,:,-self.chunk:] = torch.zeros_like(u_c_cat[1][:,:,-self.chunk:]).to(inference_dtype)
    u_c_cat[0] = u_c_cat[0]*u_c_cat[1].to(inference_dtype)
uc = {"c_concat": u_c_cat, "c_crossattn": [uc, ]}
```

之后会进行ddim采样，解码和积累：

```Python
samples, _ = self.sample_log(
    cond=cond, batch_size=N, ddim=True,
    ddim_steps=ddim_steps, causal=True, eta=ddim_eta,
    unconditional_guidance_scale=unconditional_guidance_scale,
    unconditional_conditioning=uc, x0=z.to(inference_dtype), chunk=self.chunk,
    cat_mask=self.use_cat_mask, sparse=self.sparse_memory,
    traj=False, ddim_dtype=torch.float16, **kwargs
)
x_samples = self.decode_first_stage(samples.to(z.device)).data.cpu()
x_samples = rearrange(x_samples, "(b v) c t h w -> b c v t h w", v=self.n_view)
```

这就是推理的大致过程。

## def get\_batch\_input:

这个函数负责把 dataloader 返回的batch转换成模型训练需要的：z, cond,其中:z是视频 latent,cond是条件字典\.

首先读取视频：

```Plain Text
x = super().get_input(batch, self.first_stage_key[0])
```

接着变换x的形状，合并batch和view，从\[b, c, v, t, h, w\]变成\[b\*v, c, t, h, w\]，后面再把视频编码成latent：

```Plain Text
if pre_z is not None:
    z = pre_z
else:
    z = self.encode_first_stage(x)
```

再来处理轨迹条件：

同样也是从batch里面取出轨迹，和上述过程中一样处理。

这样之后是构造 classifier\-free guidance 掩码：

1. 生成随机数：

```Plain Text
if random_uncond:
    random_num = torch.rand(b, device=x.device)
else:
    random_num = torch.ones(b, device=x.device)
```

训练时 `random_uncond=True`，为每个 batch 样本生成一个 `[0, 1)` 的随机数。

推理时通常为 `False`，于是所有值都是 `1`，表示不随机删除条件。

2. 再对文本条件进行mask：

```Plain Text
prompt_mask = 1 - rearrange(
    (random_num < 2 * self.uncond_prob).float(),
    "n -> n 1 1"
)
```

3. 图像输入 mask

```Plain Text
input_mask = 1 - rearrange(
    (random_num >= self.uncond_prob).float()
    * (random_num < 3 * self.uncond_prob).float(),
    "n -> n 1 1 1"
)
```

这段代码利用两个布尔条件相乘，判断随机数是否落在某个区间。

这样之后还会有轨迹/视频的拼接条件的mask。

5. 提取图像条件：

```Plain Text
img_ = rearrange(torch.stack(img, dim=0), 'b c v h w -> (b v) c h w')
x = rearrange(x ,'b c v t h w -> (b v) c t h w')
img_ = rearrange(
    torch.stack(img, dim=0),
    'b c v h w -> (b v) c h w'
)
```

这是取出条件帧，并且合并batch和view，接着运用掩码：

```Plain Text
input_mask = rearrange(
    input_mask,
    'b v c h w -> (b v) c h w'
)
img = input_mask * img_
```

6. 图像编码并且处理文本条件：

```Plain Text
img_emb = self.embedder(img)
img_emb = self.image_proj_model(img_emb)
cond_input = prompt_mask * cond_input
cond_emb = self.cond_stage_model(
    cond_input.to(dtype=x.dtype, device=self.device)
)
```

7. 构造视频拼接条件：

```Plain Text
cat_mask = cat_mask.unsqueeze(1).repeat(1, v, 1, t, 1, 1)
cat_mask = rearrange(
    cat_mask,
    'b v c t h w -> (b v) c t h w'
)
```

把 mask 扩展成视频 latent 的形状：

```Plain Text
[b*v, 1, t, h, w]
```

```Plain Text
cat_mask[:, :, :-self.chunk] = 1.
```

前面的历史帧强制保留。

也就是说，mask 主要控制最后chunk帧

轨迹 mask 同理：

```Plain Text
cat_traj_mask[:, :, :-self.chunk] = 0.
```

8. 构造图像拼接 latent

```Plain Text
img_cat_cond = z.clone()
```

复制一份视频 latent。

```Plain Text
img_cat_cond[:, :, -self.chunk:] =
    img_cat_cond[:, :, -(self.chunk+1):-self.chunk]
    .repeat(1, 1, self.chunk, 1, 1)
```

取最后预测区域前面的那一帧，然后重复chunk次

9. 之后保存到 `c_concat`

```Plain Text
cond["c_concat"] = [img_cat_cond]
```

`c_concat` 是通过通道维拼接的条件。

如果启用 mask：

```Plain Text
cond['c_concat'].append(cat_mask)
```

再添加轨迹：

```Plain Text
cond['c_concat'].append(traj)
```

因此得到：

```Plain Text
cond["c_concat"] = [
    img_cat_cond,
    cat_mask,
    traj
]
```

9. 构造 cross\-attention 条件

```Plain Text
cond_emb = cond_emb.unsqueeze(dim=1).repeat(1, v, 1, 1)
```

这里增加视角维度并复制到每个视角。

```Plain Text
cond_emb = rearrange(
    cond_emb,
    "b v c t -> (b v) c t"
)
```

变成：

```Plain Text
[b*v, sequence_length, embedding_dim]
```

然后：

```Plain Text
cond["c_crossattn"] = [
    torch.cat([img_emb, cond_emb], dim=1)
]
```

把图像 embedding 和文本 embedding 在第 1 维拼接。

`c_crossattn` 会被 Transformer 或 cross\-attention 模块使用。

10. 可选 ray map

```Plain Text
if self.use_raymap_dir or self.use_raymap_origin:
```

如果启用几何条件，就读取：

```Plain Text
intrinsic = batch["intrinsic"]
extrinsic = batch["extrinsic"]
```

- `intrinsic`：相机内参

- `extrinsic`：相机外参

然后：

```Plain Text
intrinsic_transform_batch(...)
```

把相机内参缩放到 latent 分辨率。

```Plain Text
gen_batch_ray_parellel(...)
```

生成每个像素对应的光线：

- `batch_raymap_o`：光线起点

- `batch_raymap_d`：光线方向

最后把它们追加到：

```Plain Text
cond["c_concat"]
```



## Def sample\_log:

这个函数的主要作用就是加载样本和一些中间态：

```Python
@torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, causal=False, chunk=4,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, causal=causal,chunk=chunk,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True,causal=causal,chunk=chunk,**kwargs)

        return samples, intermediates

```

## def p\_losses:

这个函数负责计算扩散模型训练损失。

整体流程：

```Plain Text
原始 latent x_start
    ↓ 加噪
带噪 latent x_noisy
    ↓ UNet 预测
model_output
    ↓ 和目标比较
loss
```

1. 生成噪声

```Plain Text
if self.noise_strength > 0:
```

如果启用了 offset noise：

```Plain Text
b, c, f, _, _ = x_start.shape
```

读取形状。

```Plain Text
offset_noise = torch.randn(
    b, c, f, 1, 1,
    device=x_start.device
)
```

生成每个 batch、通道、帧独立的噪声。

```Plain Text
noise = default(
    noise,
    lambda: torch.randn_like(x_start)
    + self.noise_strength * offset_noise
)
```

如果调用者没有传入 `noise`，就生成随机噪声。

2. 给视频加噪

```Plain Text
x_noisy = self.q_sample(
    x_start=x_start,
    t=t,
    noise=noise
)
```

`q_sample` 实现正向扩散：

```Plain Text
x_noisy =
sqrt(alpha_bar_t) * x_start
+
sqrt(1 - alpha_bar_t) * noise
```

其中我们保留历史帧不加噪

```Plain Text
x_noisy[:, :, :-self.chunk] = x_start[:, :, :-self.chunk]
```

3. 调用扩散模型

```Plain Text
model_output = self.apply_model(
    x_noisy,
    t,
    cond,
    **kwargs
)
```

`apply_model` 会把条件整理成：

```Plain Text
{
    "c_concat": ...,
    "c_crossattn": ...
}
```

然后调用：

```Plain Text
self.model(...)
```

而 `self.model` 就是 `DiffusionWrapper`。

4. 选择训练目标

```Plain Text
if self.parameterization == "x0":
    target = x_start
```

模型直接预测原始干净 latent。

```Plain Text
elif self.parameterization == "eps":
    target = noise
```

模型预测加入的噪声。

5. 计算逐元素损失

```Plain Text
loss_simple_map = self.get_loss(
    model_output,
    target,
    mean=False,
    last_only=self.chunk
)
```

`mean=False` 表示先不求平均，保留每个元素的损失。

`last_only=self.chunk` 会只取最后 `chunk` 帧：

```Plain Text
target = target[:, :, -last_only:]
pred = pred[:, :, -last_only:]
```

因此损失只计算生成区域，不计算历史区域。

对于 5 维视频张量，损失形状是：

```Plain Text
[b, c, chunk, h, w]
```

```Plain Text
loss = loss_simple / torch.exp(logvar_t) + logvar_t
```

这是扩散模型中用于调整不同时间步损失权重的公式。

最后的到主要训练损失：

```Plain Text
loss = self.l_simple_weight * loss.mean()
```

6. VLB 损失：

再引入扩散模型/vae里面常用的损失优化：

```Plain Text
loss_vlb = loss_simple
```

这里没有重新计算完整的 VLB，而是直接复用 `loss_simple`。

```Plain Text
loss_vlb = (
    self.lvlb_weights[t] * loss_vlb
).mean()
```

根据时间步权重调整。

```Plain Text
loss += self.original_elbo_weight * loss_vlb
```

如果：

```Plain Text
original_elbo_weight == 0
```

那么 VLB 不会影响最终损失。

最后：

```Plain Text
loss_dict.update({
    f'{prefix}/loss': loss
})
return loss, loss_dict
```

返回：

1. 用于反向传播的标量 `loss`

2. 用于日志记录的字典 `loss_dict`





## Def forward:

forward的代码也很少：

```Python
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, s=None, mask=None, **kwargs):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        else:
            raise NotImplementedError()

        return out
```

主要是根据不同的conditioning\_key来决定我们的x要被送进哪个模块里面。

主要参数：

- `x`：带噪 latent

- `t`：扩散时间步

- `c_concat`：需要在通道维拼接的条件

- `c_crossattn`：送给 cross\-attention 的条件

- `**kwargs`：额外参数

里面最关键的一点是：

**hybrid 模式**

ACWM 使用的就是这个模式：

```Plain Text
elif self.conditioning_key == 'hybrid':
```

同时使用两类条件。

1. 拼接类条件

```Plain Text
xc = torch.cat([x] + c_concat, dim=1)
```

把：

```Plain Text
带噪 latent
图像 latent
轨迹 latent
mask
ray map
```

沿通道维拼起来。

2. 注意力类条件

```Plain Text
cc = torch.cat(c_crossattn, 1)
```

把：

```Plain Text
图像 embedding
文本 embedding
```

拼接成 attention context。

3. 调用 UNet

```Plain Text
out = self.diffusion_model(
    xc,
    t,
    context=cc,
    **kwargs
)
```

因此 hybrid 模式同时完成：

```Plain Text
输入：xc
条件：cc
时间步：t
```

## DDIM

这里说一下ddim，正好这个仓库的采样是用ddim的。

## def sample:

首先准备采样表：

```Plain Text
self.make_schedule(
    ddim_num_steps=S,
    ddim_discretize=timestep_spacing,
    ddim_eta=eta,
    verbose=schedule_verbose,
    ddpm_from=ddpm_from
)
```

然后回获取我们的batch的形状，之后调用采样函数：

```Plain Text
outputs = self.ddim_sampling_causal(conditioning, size,
                                                callback=callback,
                                                img_callback=img_callback,
                                                quantize_denoised=quantize_x0,
                                                mask=mask, x0=x0,
                                                ddim_use_original_steps=False,
                                                noise_dropout=noise_dropout,
                                                temperature=temperature,
                                                score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                log_every_t=log_every_t,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning,
                                                verbose=verbose,
                                                precision=precision,
                                                fs=fs,
                                                guidance_rescale=guidance_rescale,
                                                chunk=chunk,
                                                cat_mask=cat_mask,
                                                sparse=sparse,
                                                **kwargs)

        return outputs
```

然后来到真正的核心函数：

## def ddim\_sampling\_causal:

从 `c_concat` 取初始视频

```Plain Text
img = cond['c_concat'][0].clone()
```

前面我们说过：

```Plain Text
c_concat[0] = img_cat_cond
```

它包含：

```Plain Text
历史帧 latent + 最后 chunk 帧的条件 latent
```

接着生成噪声：

```Plain Text
if predefined_noise is None:
    img[:, :, -chunk:] = torch.randn_like(img[:, :, -chunk:])
```

选择时间步：

```Plain Text
if timesteps is None:
    timesteps = (
        self.ddpm_num_timesteps
        if ddim_use_original_steps
        else self.ddim_timesteps
    )
```

构建倒序去噪循环

```Plain Text
time_range = np.flip(timesteps)
```

`np.flip` 会反转数组。

接着循环生成：

```Plain Text
for fstep in tqdm(range(total_frames // chunk)):

            b,c,f,h,w = img.shape


            img_ = img.clone()

            for i, step in enumerate(iterator):


                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
                if mask is not None:
                    assert x0 is not None
                    if clean_cond:
                        img_orig = x0
                    else:
                        img_orig = self.model.q_sample(x0, ts)
                    img = img_orig * mask + (1. - mask) * img # keep original & modify use img

                outs = self.p_sample_ddim(img_, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                        **kwargs)

                img_, pred_x0 = outs

                img_[:,:,:-chunk] = img[:,:,:-chunk]


                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if return_intermediates and (index % log_every_t == 0 or index == total_steps - 1):
                    intermediates['x_inter'].append(img_[:,:,-chunk:])
                    intermediates['pred_x0'].append(pred_x0[:,:,-chunk:])

                torch.cuda.empty_cache()
```

**执行一次 DDIM 去噪**

```Plain Text
outs = self.p_sample_ddim(img_, cond, ts, index=index, ...)
img_, pred_x0 = outs
```

`p_sample_ddim` 做一次：

```Plain Text
x_t → x_(t-1)
```

它内部会：

1. 调用 UNet，预测噪声或 `v`；

2. 根据预测结果推回 `pred_x0`（估计的干净 latent）；

3. 计算前一个时间步的 `x_prev`。

因此返回：

```Plain Text
img_     # x_(t-1)，用于下一轮去噪
pred_x0  # 模型认为的干净视频 latent
```

```Plain Text
x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
```

---

**强制历史帧不被修改**

```Plain Text
img_[:, :, :-chunk] = img[:, :, :-chunk]
```

这是因果生成的关键。

无论 DDIM 在 `p_sample_ddim` 中如何处理，前面的历史帧都被覆盖回原样。

**更新 ****`c_concat[0]`**

```Plain Text
cond['c_concat'][0] = torch.cat([
    cond['c_concat'][0][:, :, chunk:-chunk],
    img_[:, :, -chunk:],
    img_[:, :, -1:].repeat(1, 1, chunk, 1, 1)
], dim=2)
```

**收集并返回所有生成帧**

```Plain Text
img_out.append(img_[:, :, -chunk:].data.cpu())
```

每次生成完一个 chunk，就把它放到输出列表。

```Plain Text
img_out = torch.cat(img_out, dim=2)
```

沿时间维拼接：

```Plain Text
[历史帧] + [第 1 个生成 chunk] + [第 2 个生成 chunk] + ...
```

最后：

```Plain Text
return img_out, intermediates
```

- `img_out`：最终视频 latent；

- `intermediates`：DDIM 去噪中的中间 latent，用于可视化。

最终 `img_out` 还需要通过 VAE decoder：

```Plain Text
decode_first_stage(img_out)
```

才能还原成真正的视频帧。

## UNet 路径

## Attention / Resampler 路径

## VAE / Action / Ray / Trajectory 路径

这三部分在本项目中不是平行独立的，而是一条数据流：

```Plain Text
视频 / 轨迹图 ──VAE──> latent ─┐
ray map ──────────────────────┼─> c_concat ─┐
条件图像 ─CLIP + Resampler──> ─┤             │
action ─────Resampler────────> └─> context ─┼─> UNet ─> 预测 v
                                             │
带噪 latent x ───────────────────────────────┘
```

配置里已经明确了输入通道组成：

```Plain Text
img(4) + mask(1) + cond img(4) + traj(4) + ray(6) = 19
```

见 \[config\.yaml \(line 58\)\]\(/C:/Users/Arison/Desktop/EnerVerse\-AC\-main/configs/agibotworld/config\.yaml:58\)。

---

## UNet

当前配置：

```Plain Text
输入： [B*V, 19, 20, 40, 64]
输出： [B*V,  4, 20, 40, 64]
```

- 19：上面说的 `x + c_concat`

- 20：4 帧历史 \+ 16 帧待生成

- 4：VAE latent 的通道数

- 模型使用 `parameterization: "v"`，所以输出是预测的 `v`，不是最终视频。

### 初始化中最关键的部分

```Plain Text
self.time_embed = nn.Sequential(
    linear(model_channels, time_embed_dim),
    nn.SiLU(),
    linear(time_embed_dim, time_embed_dim),
)
```

这是“扩散时间步 embedding”。

例如当前噪声处于第 700 步，整数 `700` 会先变成一个向量，再经过两层全连接层，成为 `emb`。UNet 中每个 ResBlock 都能知道：“现在噪声有多重”。

```Plain Text
self.input_blocks = nn.ModuleList([
    TimestepEmbedSequential(
        conv_nd(dims, in_channels, model_channels, 3, padding=1)
    )
])
```

第一层卷积，把输入：

```Plain Text
19 通道 → 320 通道
```

`TimestepEmbedSequential` 是特殊的 Sequential：普通层只接收 `x`，但 ResBlock、Transformer 还可以接收时间 embedding `emb`、条件 `context`。

```Plain Text
for level, mult in enumerate(channel_mult):
    for _ in range(num_res_blocks):
```

这是 U\-Net 的下采样编码器部分。

当前：

```Plain Text
channel_mult: [1, 2, 4, 4]
num_res_blocks: 2
```

即有四个尺度，每个尺度两个 ResBlock。通道数大致为：

```Plain Text
320 → 640 → 1280 → 1280
```

```Plain Text
layers = [
    ResBlock(
        ch, time_embed_dim, dropout,
        out_channels=mult * model_channels,
        ...
    )
]
```

每层先经过 ResBlock。ResBlock 的重点是：

```Plain Text
输入特征 + 时间步 emb
       ↓
卷积、归一化、激活
       ↓
残差连接
```

`temporal_conv=True` 时，ResBlock 内还有时间维卷积，因此不仅看单帧，也会看邻近帧。

```Plain Text
if ds in attention_resolutions:
```

`ds` 是当前下采样倍数。配置里：

```Plain Text
attention_resolutions: [4, 2, 1]
```

所以多个分辨率都会插入 attention。

```Plain Text
layers.append(
    S2MVTransformer(..., context_dim=context_dim, ...)
)
```

这是空间/多视角 Transformer。它会让每个空间位置读取 `context`，也就是图像 token 和 action token。

```Plain Text
layers.append(
    TemporalTransformer(...)
)
```

这是时间 Transformer，用来处理视频中帧与帧之间的关系。

```Plain Text
self.input_blocks.append(
    TimestepEmbedSequential(
        Downsample(...)
    )
)
ds *= 2
```

每个尺度结束时下采样：

```Plain Text
H, W 变小
感受野变大
```

例如：

```Plain Text
40×64 → 20×32 → 10×16 → 5×8
```

中间层 `middle_block` 是 U\-Net 最深处：

```Plain Text
self.middle_block = TimestepEmbedSequential(*layers)
```

之后 `output_blocks` 做上采样解码。关键是跳跃连接：

```Plain Text
h = torch.cat([h, hs.pop()], dim=1)
```

`hs` 存着编码器每一层的结果。`pop()` 取出最后一个。

这就是 U\-Net 的 “U”：

```Plain Text
编码器：逐渐缩小，理解整体内容
      ↓
中间层
      ↓
解码器：逐渐放大，恢复细节
      ↑
跳跃连接：把编码器的细节直接送回来
```

---

### `UNetModel.forward` 逐段解释

```Plain Text
x = rearrange(x, '(b v) c t h w -> b c v t h w', v=self.n_view)
b, _, v, t, _, _ = x.shape
```

之前 batch 和 view 被合并为 `B*V`，这里拆回去：

```Plain Text
[B*V, C, T, H, W]
→
[B, C, V, T, H, W]
```

```Plain Text
t_emb = timestep_embedding(
    timesteps, self.model_channels, repeat_only=False
).type(x.dtype)

emb = self.time_embed(t_emb)
```

将扩散时间步 `timesteps` 转成向量 `emb`。

```Plain Text
t0_emb = timestep_embedding(torch.zeros_like(timesteps)[:1], ...)
emb0 = self.time_embed(t0_emb)
emb0 = emb0.repeat(b*v, 1)
```

构造时间步为 0 的 embedding。

原因是：前面历史帧是干净的，没有加噪，因此给它们 `t=0`；最后待生成的 `chunk` 帧才使用真实扩散时间步。

```Plain Text
emb = emb.unsqueeze(1).repeat(1, self.chunk, 1)
emb0 = emb0.unsqueeze(1).repeat(1, t-self.chunk, 1)
emb = torch.cat([emb0, emb], dim=1)
```

以当前配置为例：

```Plain Text
t = 20
chunk = 16
```

最终每帧对应的扩散时间：

```Plain Text
历史 4 帧：t = 0
最后 16 帧：t = 当前 DDIM / DDPM 时间步
```

```Plain Text
emb = rearrange(emb, '(b v) t c -> (b v t) c', b=b, v=v)
x = rearrange(x, 'b c v t h w -> (b v t) c h w')
```

把每一帧摊平为类似图片 batch：

```Plain Text
[B, C, V, T, H, W]
→
[B*V*T, C, H, W]
```

这样空间卷积可以像处理图片一样处理每一帧。时间关系则由 TemporalTransformer 和 temporal conv 处理。

```Plain Text
if self.fs_condition:
```

把 fps/frame stride 转为 embedding 后加进 `emb`：

```Plain Text
emb = emb + fs_embed
```

同样，`domain_id` 也可变为 embedding：

```Plain Text
emb = emb + domain_emb
```



```Plain Text
h = x.type(self.dtype)
hs = []
for idx, module in enumerate(self.input_blocks):
    h = module(h, emb, context=context, batch_size=b, n_view=v)
    hs.append(h)
```

逐层执行编码器，并保存每层输出到 `hs`，供之后跳跃连接使用。

```Plain Text
h = self.middle_block(h, emb, context=context, batch_size=b, n_view=v)
```

通过最深层。

```Plain Text
for module in self.output_blocks:
    h = torch.cat([h, hs.pop()], dim=1)
    h = module(h, emb, context=context, batch_size=b, n_view=v)
```

解码阶段：拼接对应编码器特征后继续处理。

```Plain Text
y = self.out(h)
y = rearrange(y, '(b v t) c h w -> (b v) c t h w', b=b, v=v)
return y
```

输出从：

```Plain Text
[B*V*T, 4, H, W]
```

恢复成：

```Plain Text
[B*V, 4, T, H, W]
```

这就是扩散模型需要的预测结果。

---

## attention 和 Resampler：

### Resampler：

再把很多输入特征压缩成固定数量 token

项目有两个 Resampler：

```Plain Text
delta_action → 16 个 action token
CLIP 图像特征 → 16×16=256 个 image token
```

```Plain Text
self.latents = nn.Parameter(
    torch.randn(1, num_queries, dim) / dim**0.5
)
```

`latents` 是可训练的“查询 token”。

```Plain Text
self.proj_in = nn.Linear(embedding_dim, dim)
self.proj_out = nn.Linear(dim, output_dim)
self.norm_out = nn.LayerNorm(output_dim)
```

- `proj_in`：将输入特征维度统一成内部维度；

- `proj_out`：投影成 UNet context 维度 `1024`；

- `LayerNorm`：稳定数值范围。

```Plain Text
latents = self.latents.repeat(x.size(0), 1, 1)
```

原本：

```Plain Text
[1, token数, 1024]
```

复制给 batch 内每个样本：

```Plain Text
[B, token数, 1024]
```

```Plain Text
x = self.proj_in(x)
```

例如 action 原始每一帧是 14 维：

```Plain Text
[B, 16, 14] → [B, 16, 1024]
```

```Plain Text
for attn, ff in self.layers:
    latents = attn(x, latents) + latents
    latents = ff(latents) + latents
```

每一层都做：

```Plain Text
可学习 query token
      ↓ 读取 x
PerceiverAttention
      ↓
残差相加
      ↓
前馈网络 FFN
```

- `latents` 是残差连接，避免深层网络破坏旧信息。

```Plain Text
latents = self.proj_out(latents)
latents = self.norm_out(latents)
return latents
```

返回固定数量、固定维度的 token。

---

### `PerceiverAttention`：

Resampler 内部怎样“读取”输入？

```Plain Text
q = self.to_q(latents)
kv_input = torch.cat((x, latents), dim=-2)
k, v = self.to_kv(kv_input).chunk(2, dim=-1)
```

- `q`：来自可学习 query token；

- `k, v`：来自输入特征 `x`，也包含 query token 自己；

- `dim=-2`：倒数第二维，即 token 数量维；

- `.chunk(2, dim=-1)`：在最后一维均分成两半，分别当作 K 和 V。

```Plain Text
weight = (q * scale) @ (k * scale).transpose(-2, -1)
weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
out = weight @ v
```

---

### CrossAttention：

```Plain Text
self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
```

- Query `Q`：来自当前 UNet 视频特征；

- Key `K`、Value `V`：来自条件 token。

当前配置：

```Plain Text
图像 token：256 个
action token：16 个
context 总长：272 个
```

```Plain Text
if self.traj_cross_attention:
    self.text_context_len = 0
    self.to_k_tp = nn.Linear(context_dim, inner_dim, bias=False)
    self.to_v_tp = nn.Linear(context_dim, inner_dim, bias=False)
```

注意这里变量名 `traj` 容易让人迷惑。

在当前模型中，attention 的最后 16 个 token 实际来自：

```Python
cond_stage_key: delta_action
```

也就是 action Resampler 的输出；代码把这一类额外 token 叫作 `traj` attention 分支。

而真正画出来的轨迹图 `traj` 则走的是 VAE → `c_concat`，不是这 16 个 token。

```Plain Text
q = self.to_q(x)
context = default(context, x)
```

- 若传入 `context`：cross\-attention；

- 若没传入：`context=x`，就变成 self\-attention。

```Plain Text
context_image = ...
context_traj = ...
k_ip = self.to_k_ip(context_image)
v_ip = self.to_v_ip(context_image)
k_tp = self.to_k_tp(context_traj)
v_tp = self.to_v_tp(context_traj)
```

把 context 按 token 段拆开：

```Plain Text
前 256 个：条件图像 token
后 16 个：action token
```

然后分别生成它们的 K、V。

```Plain Text
out_ip = xformers_attn(q, k_ip, v_ip)
out_tp = xformers_attn(q, k_tp, v_tp)
```

在高效实现中，分别计算：

```Plain Text
视频特征读取图像条件的结果
视频特征读取动作条件的结果
```

最后相加：

```Plain Text
out = out + self.image_cross_attention_scale * out_ip
out = out + self.traj_cross_attention_scale * out_tp
```

所以 UNet 最终同时受到：

```Plain Text
条件图像：场景现在长什么样
action：机器人接下来如何移动
```

的影响。

---

## VAE / action / ray / traj：

### VAE：

图像和轨迹图变成 latent

```Plain Text
h = self.encoder(x)
moments = self.quant_conv(h)
posterior = DiagonalGaussianDistribution(moments)
return posterior
```

- `encoder(x)`：把 RGB 图像压缩成特征；

- `quant_conv`：将特征转成高斯分布参数；

- `DiagonalGaussianDistribution`：表示 latent 不是一个固定值，而是一个分布。

```Plain Text
z = posterior.sample()
```

训练或编码时，从这个分布采样 latent `z`。

当前配置：

```Plain Text
原图： [B, 3, 256, 256]
latent：[B, 4,  32,  32]
```

因此扩散模型不是直接生成 256×256 RGB，而是生成更小的 4 通道 latent。

解码反过来：

```Plain Text
z = self.post_quant_conv(z)
dec = self.decoder(z)
return dec
```



```Plain Text
latent z → Decoder → RGB 图像
```

在 ACWM 中：

```Plain Text
video → VAE → x_start
traj 图 → VAE → c_concat 中的 traj latent
```

---

### action：

配置中：

```Plain Text
cond_stage_key: delta_action
embedding_dim: 14
```

左臂：

```Plain Text
all_delta_actions[..., 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
all_delta_actions[..., 6] = gripper[...] / 120.0
```

即：

```Plain Text
左手：位置变化 xyz（3）
    + 姿态变化 rpy（3）
    + 夹爪开合（1）
= 7
```

右臂：

```Plain Text
all_delta_actions[..., 7:13] = ...
all_delta_actions[..., 13] = ...
```

```Plain Text
右手同样 7 维
总计 14 维
```

这些 `[时间, 14]` 的动作数据进入 action Resampler，变成 16 个 action token，进入 `c_crossattn`。

---

### ray：

```Plain Text
fx, fy, cx, cy = intrinsic[:, 0, 0], ...
```

从相机内参矩阵取出：

- `fx, fy`：焦距；

- `cx, cy`：主点坐标。

```Plain Text
i, j = torch.meshgrid(...)
```

构造每个像素的位置网格。

```Plain Text
dirs = torch.stack([
    (i-cx)/fx,
    (j-cy)/fy,
    torch.ones_like(i)
], -1)
```

根据针孔相机模型，算出像素在相机坐标系中的方向：

```Plain Text
[(u-cx)/fx, (v-cy)/fy, 1]
```

```Plain Text
rays_d = torch.sum(
    dirs[..., np.newaxis, :] * c2w[..., :3, :3],
    -1
)
```

用相机旋转矩阵把方向从“相机坐标系”转换到“世界坐标系”。

```Plain Text
rays_o = c2w[:, :3, -1]...
viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
```

- `rays_o`：每条光线起点，即相机位置；

- `viewdir`：归一化后的光线方向。

因此 ray condition 共 6 通道：

```Plain Text
origin xyz：3
direction xyz：3
```

它们被加入 `c_concat`，让模型知道相机几何关系。

---

### traj：

后把机械臂末端轨迹画成一张图

```Plain Text
ee_key_pts = torch.tensor(EndEffectorPts, ...).view(1,1,4,4).permute(0,1,3,2)
```

`EndEffectorPts` 是夹爪上的 4 个固定关键点，以末端执行器自身坐标系表示。

```Plain Text
pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7])
pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15])
```

将左右手的：

```Plain Text
xyz + 四元数 quaternion
```

变成 4×4 位姿变换矩阵。

```Plain Text
ee2cam_l = torch.matmul(w2c, pose_l_mat)
ee2cam_r = torch.matmul(w2c, pose_r_mat)
```

将夹爪关键点从世界/机器人坐标转换到相机坐标。

```Plain Text
pts_l = torch.matmul(ee2cam_l, ee_key_pts)
pts_r = torch.matmul(ee2cam_r, ee_key_pts)
```

得到每个夹爪关键点在相机坐标中的 3D 坐标。

```Plain Text
uvs_l = torch.matmul(intrinsic, pts_l[:, :, :3, :])
uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :]
```

使用相机内参投影到二维图像：

```Plain Text
3D 点 (X, Y, Z)
→
像素点 (u, v)
```

除以 `Z` 是透视投影。

```Plain Text
img = np.zeros((h, w, 3), dtype=np.uint8) + 50
```

创建深灰色背景轨迹图。

```Plain Text
cv2.circle(img, tuple(point), radius, color, -1)
```

画夹爪基点圆。

```Plain Text
cv2.line(img, tuple(base), tuple(point), colors[i-1], 8)
```

画基点到其他夹爪关键点的线。

最后：

```Plain Text
all_img_list = rearrange(
    torch.tensor(all_img_list),
    "v t h w c -> c v t h w"
).float()
```

把轨迹图整理成：

```Plain Text
[C, V, T, H, W]
```

之后它和视频一样经过 VAE，成为 4 通道轨迹 latent，再放进 `c_concat`。

最容易混淆的一点是：

```Plain Text
action：14 个数 → Resampler → c_crossattn
traj 图：彩色轨迹图 → VAE → c_concat
ray：相机 origin + direction → c_concat
```

这三者都和“动作”有关，但进入 UNet 的路线完全不同。

## 关联笔记

- [EnerVerse-AC 仓库概览](overview.md)
- [Action conditioning 专题](../../topics/action-conditioning.md)
- [Action controllability 评估设计](../../research/action-controllability-evaluation.md)
