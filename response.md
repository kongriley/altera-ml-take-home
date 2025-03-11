1. What should you do if the two models have different tokenizers?

We would have to employ a method to convert one tokenizer to another. There are some approaches, but they are not perfect and have their own drawbacks.

One approach would be to perform subword composition by breaking up the tokens for a model $m_x$ into subwords, which exist in the vocabulary of the other model $m_y$. Then the distribution $\log p_{m_x}(w)$ can be computed by summing over the probabilities of the subwords in $w$. Unfortunately, this approach fails for completely novel tokens that do not have any subwords in the vocabulary of $m_y$.

Another approach would be to use a projection-based method. We could project the tokens of $m_x$ into the vocabulary of $m_y$ by training on a model $W$ such that $W \cdot \text{token}_x = \text{token}_y$. Unfortunately, this approach requires training $W$, which might involve a large amount of data and be infeasible.

1. Do you think contrastive decoding is used in practice?

No, I don't think contrastive decoding is used in practice. It requires two models (amateur and expert) to be served at the same time, which introduces computational overhead. Other strategies like prompt/fine-tuning or better sampling strategies on a single model are more practical.