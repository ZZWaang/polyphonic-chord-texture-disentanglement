This is the demo folder of the paper 
"LEARNING INTERPRETABLE REPRESENTATION FOR CONTROLLABLE POLYPHONIC MUSIC GENERATION"
Submitted to ISMIR2020.

* The demo folder contains 3 subfolders, corresponding to the 3 tasks of controllable generation discussed in Section 4.
* For each task, we present 1) the MIDI files corresponding to the figures in the paper, and 2)more examples
* Below is the description of each sub-folder

* 1_compositional_style_transfer
    * `16bar_style_transfer` corresponds to Fig. 2, 16-bar style transfer.
        * 'all.mid' contains all four scores. **Please open this file using DAW!**
        * We present more examples in `more_examples`.
    * `2bar_style_transfer` presents k^2 recombination of k z_chd's and k z_txt's from k 2-bar pieces.
        * `explain_swap.png` explains how it is done when k = 4. 
        * `swap_x_y.mid` indicates the x-th row, y-th column of the piano-roll in `explain_swap.png`. 
        * `swap_all.mid` contains multiple tracks of all 16 piano-rolls. **Please open this file using DAW!**
        * In `more examples` folder, we present k = 31 2-bar style transfer. **Please open this file using DAW!**
* 2_texture_variation
    * `posterior_sampling` corresponds to Fig. 3(a).
        * `original.mid` is the original piece. 
        * `post_sample_on_fig.mid` is the posterior sample of `original.mid` (Fig. 3(a)).
        * We provide 2 more posterior samples of `original.mid`.
        * We provide more examples in `more_examples`.
    * `prior_sampling` corresponds to Fig. 3(b).
        * `C-Am-F-G.mid` corresponds to the score in Fig. 3(b).
* `acc_arrangement`
    * `fig_generation_given2bar.mid` corresponds to the score in Fig. 4.  
    * We provide more examples in `more_examples`. 
    * In the `more_examples/long_generation` subfolders, we iteratively call the transformer model to generate long accompaniment (>= 16 bars). 
    * In all arrangement examples, the whole melody, the whole chord progression, and the first two/four bars of the accompaniment are given.


* We also provide examples of chord progression interpolation by interpolating the latent chord representation. (This part is not discussed in the paper.)