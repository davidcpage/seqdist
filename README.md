# MCTC
> Matrix-valued and banded extensions of the CTC loss in pytorch and cupy.


## Install

`pip install your_project_name`

## How to use

```
sample_inputs = logits, targets, input_lengths, target_lengths = generate_sample_inputs(T_min=450, T_max=500, N=128, C=20, L_min=80, L_max=100)
ctc_loss_pytorch(*sample_inputs).item(), ctc_loss_cupy(*sample_inputs).item()
```




    2



```
report(benchmark_fwd_bwd(ctc_loss_pytorch, *sample_inputs))
```

```
report(benchmark_fwd_bwd(ctc_loss_cupy, *sample_inputs))
```

```
betas = [0.1, 1.0, 10.]
alignments = {f'beta={beta:.1f}': to_np(soft_alignments(*sample_inputs, beta=beta)) for beta in betas}
alignments['viterbi'] = to_np(viterbi_alignments(*sample_inputs)
fig, axs = plt.subplots(2, 2, figsize=(15, 8))
for (ax, (title, data)) in zip(np.array(axs).flatten(), alignments.items()):
    ax.imshow(data[:, 0].T, vmax=0.05);
    ax.set_title(title)  
```
