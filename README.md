# Personalized LexRank

This repository provides a reference implementation of *Personalized LexRank* as described in the paper:<br>
> Learning to Generate Personalized Product Descriptions.<br>
> Guy Elad, Ido Guy, Slava Novgorodov, Benny Kimelfeld and Kira Radinsky.<br>
> ACM International Conference on Information and Knowledge Management, 2019.<br>
> http://kiraradinsky.com/files/Learning_to_Generate_Personalized_Product_Descriptions.pdf<Insert paper link>

In the *Personalized LexRank* algorithm, we apply extractive summarization over a given product description. The algorithm constructs a sentence graph, and applies random walks thereof to produce the summarization. The graph is built to adapt to the user personality, and it incorporates the personality language model of the user for creating the summary.

### Requirements
 - python>=3.6
 - pandas
 - numpy
 - scipy
 - sklearn
 - colour


### Basic Usage


### Citing
If you find *Personalized LexRank* useful for your research, please consider citing the following paper:

@article{elad2019learning,
  title={Learning to Generate Personalized Product Descriptions},
  author={Elad, Guy and Guy, Ido and Novgorodov, Slava and Kimelfeld, Benny and Radinsky, Kira},
  year={2019}
}

Contact: <sguyelad@cs.technion.ac.il>.