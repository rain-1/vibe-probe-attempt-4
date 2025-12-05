Now we want to be able to train a logistic probe that should be a simple linear layer that attaches to the hidden layer in question and has gelu on the end.
So we want to be able to take our data set and train the probe on it in isolation from the LLM, since the hidden activations are there for us to use.
We could either train on the binary classification problem or we could train on the probability logit that might have a better stronger signal for us.
We might want to use Ridge regularization to make the training converge well.
We should have progress bars with tqbn and we should want wandb log as much detail as we can about this It would be fantastic to be logging the loss the accuracy on a held out validation data set and also positive accuracy and the negative accuracy separately so that we can understand what's happening at a finer level
when training a probe include metadata in the file to explain what the attachment site is. (which layer we trained it on)
