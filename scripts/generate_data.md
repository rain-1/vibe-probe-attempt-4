
Load up unsloth/gemma-3-270m
generate 100 tokens with temp 1 then continue to generate 1000 tokens with temp 0, then truncate to 1000 tokens (removing the initial 100 high temp stuff)
repeat this process n times, producing a jsonl file containing n entries.

next take our target token 'T' and pick out all the strings that end with 'T' to create our positive data set. Suppose we get K of these.
pick out all the strings that do not end with 'T' to create our negative data set, but there will be many more of those. so stop as soon as you have K of these.

Now we process these strings in our LLM, with the final token removed (which will be T in positive set and not T in negative set). We need to run inference on all of these strings to get the hidden layer activations and store those as our probe target columns. Name them 'layer-0-hidden', 'layer-1-hidden', etc.

I'd like to be able to inspect the dataset. So let's save it as a parquet file.
