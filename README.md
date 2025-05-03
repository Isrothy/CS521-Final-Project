# CS 521 Final Project

Code is based on the repository:
   [https://github.com/ithemal/Ithemal](https://github.com/ithemal/Ithemal)
   [https://github.com/ithemal/Ithemal-models](https://github.com/ithemal/Ithemal-models)
   [https://github.com/ithemal/bhive](https://github.com/ithemal/bhive)

## Building the Docker Image

You first need to install docker and docker-compose.

It is easiest to run Ithemal within the provided docker environment. To build the docker environment, run docker/docker_build.sh. No user interaction is required during the build, despite the various prompts seemingly asking for input.

Once the docker environment is built, connect to it with docker/docker_connect.sh. This will drop you into a tmux shell in the container.

## Running the Benchmark

Here is an example of how to run the benchmark using the Ithemal model for Haswell processors:

```bash
python learning/pytorch/ithemal/evaluate.py --model Ithemal-models/paper/haswell/predictor.dump --model-data Ithemal-models/paper/haswell/trained.mdl --input-file bhive/benchmark/throughput/hsw
```

It took a long time to run the benchmark on the full set of instructions, so we recommend running it on a smaller subset of instructions first to ensure everything is working as expected.
