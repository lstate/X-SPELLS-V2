# X-SPELLS-V2 - Explaining Sentiment Classification with Diverse Synthetic Exemplars and Counter-Exemplars

**Orestis Lampridis**, Aristotle University of Thessaloniki, Greece, lorestis@csd.auth.gr \
**Laura State**, University of Pisa, Italy and Scuola Normale Superiore, Pisa, Italy, laura.state@di.unipi.it \
**Riccardo Guidotti**, University of Pisa, Italy, riccardo.guidotti@unipi.it \
**Salvatore Ruggieri**, University of Pisa, Italy, salvatore.ruggieri@unipi.it 

We present XSPELLS, a model-agnostic local approach for explain-ing the decisions black box models for sentiment classification of short texts. The explanations provided consist of a set of exemplar sentences and a set of counter-exemplar sentences. The former are examples classified by the blackbox with the same label as the text to explain. The latter are examples classified with a different label (a form of counter-factuals). Both are close in meaning to the text to explain, and both are meaningful sentences – albeit they aresynthetically generated. XSPELLS generates neighbors of the text to explain ina latent space using Variational Autoencoders for encoding text and decodinglatent instances. A decision tree is learned from randomly generated neighbors,and used to drive the selection of the exemplars and counter-exemplars. Moreover, diversity of counter-exemplars is modeled as an optimization problem,solved by a greedy algorithm with theoretical  guarantee. We report experiments on three datasets showing that XSPELLS outperforms the well-known LIME method in terms of quality of explanations, fidelity, diversity, and use-fulness, and that is comparable to it in terms of stability.

## Comments

The paper is a follow-up of "Explaining Sentiment Classification with Synthetic Exemplars and Counter-Exemplars" that can be accessed here: https://doi.org/10.1007/978-3-030-61527-7_24

The source code, which is the basis to XSPELLS2, is available under this link: https://github.com/orestislampridis/X-SPELLS

Here, we uploaded basic data as well as source code files for the SVAE and BVAE. Computations for the OVAE are the same but require a few substitutions for data loading. 

### OVAE: Optimus Variational Autoencoder

We used the framwork introduced in the paper "Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space" that can be found here: https://github.com/ChunyuanLI/Optimus

The corresponding code and pretrained models are available here: https://github.com/ChunyuanLI/Optimus

To get the code running, it is necesary to pull a docker container and link it to the code. Inside of the container, the fine-tuning and generation of sentences can be done.


## References

O. Lampridis, R. Guidotti, S. Ruggieri. Explaining Sentiment Classification with Synthetic Exemplars and Counter-Exemplars. Discovery Science (DS 2020). 357-373. Vol. 12323 of LNCS, Springer, September 202

Li, Chunyuan and Gao, Xiang and Li, Yuan and Li, Xiujun and Peng, Baolin and Zhang, Yizhe and Gao, Jianfeng, Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space, EMNLP, 2020
