this is the repo about verl. 

verl supports two inference frameworks. 1/ sglang. 2/ vllm. 

more info about verl can be found in readme at the basic level. most of the usable code is in the verl folder. entry points are in scripts folder. i want to be able to do the most basic thing here, which is create a base inference class. 

to help claude understand why - think of RL as four modular elements. 1/inference 2/ environment 3/ training 4/ algorithm. algorithm for me orchestrates env and inference. env's role is to give out a batch, inference is to use the batch and give completitions. algo gets the batch, passes to inference, gets completition passes to env which does either next turn or ends. now you are at training where policy gradient happens. THIS IS A BACKGROUND YOU DO NOT HAVE TO DO ALL THIS. THIS IS JUST SO you understand. 

verl's code is written where a lot of things are extra that i do not need in my little setup. i want more control on environment and training and want to use verl's setup and libraries for inference. later maybe algo, but for now strictly inference. 

i want you to build out a simple inference class using verl's setup. The inference class should take in a batch, tokenize it / apply chat template or whatever, and then run rollouts and generate outputs. Advantage of using verl is that i do not need to worry about ddp or multi gpu setup. I like that part. verl already handled that. i need the wrapper for inference. based on my choice of inference framework and config. 