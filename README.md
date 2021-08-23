# X-SPELLS-V2 - Explaining Sentiment Classification with Diverse Synthetic Exemplars and Counter-Exemplars

**Orestis Lampridis**, Aristotle University of Thessaloniki, Greece, lorestis@csd.auth.gr \
**Laura State**, University of Pisa, Italy and Scuola Normale Superiore, Pisa, Italy, laura.state@di.unipi.it \
**Riccardo Guidotti**, University of Pisa, Italy, riccardo.guidotti@unipi.it \
**Salvatore Ruggieri**, University of Pisa, Italy, salvatore.ruggieri@unipi.it 

We present XSPELLS, a model-agnostic local approach for explaining the decisions of black box models for sentiment classification of short texts. The explanations provided consist of a set of exemplar sentences and a set of counter-exemplar sentences. The former are examples classified by the black box with the same label as the text to explain. The latter are examples classified with a different label (a form of counter-factuals). Both are close in meaning to the text to explain, and both are meaningful sentences – albeit they are synthetically generated. XSPELLS generates neighbors of the text to explain in a latent space using Variational Autoencoders for encoding text and decoding latent instances. A decision tree is learned from randomly generated neighbors, and used to drive the selection of the exemplars and counter-exemplars. Moreover, diversity of counter-exemplars is modeled as an optimization problem, solved by a greedy algorithm with theoretical guarantee. We report experiments on three datasets showing that XSPELLS outperforms the well-known LIME method in terms of quality of explanations, fidelity, diversity, and usefulness, and that is comparable to it in terms of stability.

## Comments

The paper is a follow-up of "Explaining Sentiment Classification with Synthetic Exemplars and Counter-Exemplars" (XSPELLS) that can be accessed here: https://doi.org/10.1007/978-3-030-61527-7_24

The code of XSPELLS is available under this link: https://github.com/orestislampridis/X-SPELLS

Here, we uploaded basic data as well as source code files for the SVAE and BVAE. Computations for the OVAE are the same
but require a few substitutions for data loading. We also uploaded the version of lime we used.

### OVAE: Optimus Variational Autoencoder

We used the framework introduced in the paper "Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space"
that can be found here: https://github.com/ChunyuanLI/Optimus

The corresponding code and pretrained models are available here: https://github.com/ChunyuanLI/Optimus

To get the code running, it is necesary to pull a docker container and link it to the code. Inside of the container, the
fine-tuning and generation of sentences can be done.

## Instructions

To install the conda environment necessary to try out xspells run the following command:

```
conda env create -f environment.yml
```

Afterwards

## Licenses

- MIT License for the source code in the lstm_vae directory.
- BSD 2-Clause "Simplified" License for the source code in the lime directory.
- Apache-2.0 License for the rest of the project.

## References

O. Lampridis, R. Guidotti, S. Ruggieri. Explaining Sentiment Classification with Synthetic Exemplars and
Counter-Exemplars. Discovery Science (DS 2020). 357-373. Vol. 12323 of LNCS, Springer, September 2020

Li, Chunyuan and Gao, Xiang and Li, Yuan and Li, Xiujun and Peng, Baolin and Zhang, Yizhe and Gao, Jianfeng, Optimus:
Organizing Sentences via Pre-trained Modeling of a Latent Space, EMNLP, 2020
