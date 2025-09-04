This is a gan (generative adversarial network) with a generator and discriminator that I trained on 2048 x 1024 images and had the generator produce 512 x 256 images.

## Loss Functions

This GAN uses the **hinge adversarial loss** with **spectral normalization** on the discriminator.

### Discriminator loss
\[
\mathcal{L}_D = \mathbb{E}_{x \sim p_\text{real}} \big[ \max(0, 1 - D(x)) \big] + \mathbb{E}_{\hat{x} \sim p_G} \big[ \max(0, 1 + D(\hat{x})) \big]
\]

### Generator loss
\[
\mathcal{L}_G = - \mathbb{E}_{\hat{x} \sim p_G} [ D(\hat{x}) ]
\]

This encourages the discriminator to output **positive values for real samples** and **negative values for fake samples**, while the generator learns to maximize the discriminator’s output on generated samples.

---

## Additional Losses

Alongside the adversarial objective, the generator is trained with:

- **L1 loss** between generated and target images  
- **Perceptual loss** (VGG-based feature similarity)  
- **Feature matching loss** (using intermediate discriminator features)  

The final generator objective is a weighted sum:
\[
\mathcal{L}_G^\text{total} = \lambda_\text{adv} \, \mathcal{L}_G + \lambda_{L1} \, \mathcal{L}_{L1} + \lambda_\text{perc} \, \mathcal{L}_\text{perc} + \lambda_\text{fm} \, \mathcal{L}_\text{fm}
\]

### Default weights
- `advWeight = 10`  
- `l1Weight = 1`  
- `percWeight = 0.2`  
- `fmWeight = 10`  

---

## Notes
- The discriminator outputs raw logits (no sigmoid).  
- **Spectral normalization** is applied to the discriminator for stability.  
