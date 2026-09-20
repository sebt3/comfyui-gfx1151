# comfyui-gfx1151

Image ComfyUI pour AMD Strix Halo (**gfx1151 / RDNA 3.5**), poussée sur
Docker Hub + ghcr via GitHub Actions (`sebt3/comfyui-gfx1151`).

Sœur de [`sebt3/vllm-gfx1151`](https://github.com/sebt3/vllm-gfx1151) : même
assemblage ROCm 7.14 + wheels `wheels.vllm.ai/rocm/` (torch/torchvision/
torchaudio/triton, gfx1151-capable, mutuellement ABI-pinées), mais sans rien
de spécifique à l'inférence LLM — pas de wheel vllm, pas d'amd-aiter, pas de
flash-attn, aucun des patches site-packages FLA/GDN/MoE de vllm-gfx1151. Sur
cette base : [ComfyUI](https://github.com/comfyanonymous/ComfyUI) officiel
(GPL-3.0), tag pinné, installé via son propre `requirements.txt` contraint
pour ne pas laisser pip remplacer notre torch gfx1151 par une wheel PyPI
CUDA/CPU générique.

## Pourquoi réutiliser l'assemblage vllm-gfx1151

ComfyUI est une appli PyTorch comme une autre : pas besoin de reconstruire
une chaîne ROCm+torch gfx1151 qui fonctionne, celle de vllm-gfx1151 est déjà
validée sur silicium réel (qualification 16 tests gfx1151 via
`lemonade-sdk/vllm-rocm`, qui consomme la même source de wheels). Ce repo
ne fait que retirer les briques LLM-only et ajouter ComfyUI par-dessus.

## Attention backend

Défaut `--use-pytorch-cross-attention` (torch SDPA, backend AOTriton ROCm
via `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). AITER et flash-attn ne
sont **pas** embarqués : ils sont tunés pour les patterns d'attention LLM
(decode long-contexte), jamais mesurés sur les blocs UNet/DiT de ComfyUI —
à revisiter seulement après une comparaison A/B sur le matériel réel.

## ⚠️ Non validé sur silicium réel

Ce Dockerfile n'a pas encore bouté sur le nœud Strix Halo au moment du
premier commit. Voir `think/apps/comfyui/DEBUG.md` dans `kydah/home` pour
le statut de validation avant de considérer quoi que ce soit ci-dessus
comme un fait mesuré plutôt qu'une hypothèse de conception.
