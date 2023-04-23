# Anime-faces-generation-using-SAGAN
## For train were used DCGAN with Self Attention, <a href='https://github.com/mit-han-lab/data-efficient-gans'>Differentiable Augmentation. </a></h3>
### Used dataset https://www.kaggle.com/datasets/tianbaiyutoby/animegirl-faces
<br>
<p align="center">
<img src='https://github.com/IOBananaOI/Anime-faces-generation-using-SAGAN/blob/main/gen_outputs/gif.gif'>
</p>
<p align="center">
<h4 align="center">Generator's outputs during the training</h4>
</p>






# Preliminary results

### Models were trained for 400 epochs, with Adam optimizers, label smoothing and Spectral normalization.
<img src='https://github.com/IOBananaOI/Anime-faces-generation-using-SAGAN/blob/main/plot.png'>

## There are generator's output on the last epoch
<img src='https://github.com/IOBananaOI/Anime-faces-generation-using-SAGAN/blob/main/gen_outputs/output.png'>

## As we can see, model has trained to draw eyes, hair and face features, but still can't place them in the right place.
