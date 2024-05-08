ARNES_USER := $(shell ssh arnes whoami)

build-docker:
	docker build -t nvidia_container .

save_docker_image:
	mkdir -p ./singularity
	docker save nvidia_container:latest > ./singularity/nvidia_container.tar 

build-singularity:
	rm -rf ./singularity/nvidia_container.sif
	singularity build ./singularity/nvidia_container.sif  docker-archive://./singularity/nvidia_container.tar
	rm -rf singularity/nvidia_container.tar

setup-singularity-container: build-docker save_docker_image build-singularity

push-to-arnes:
	rsync -avz --progress ./*  arnes:/d/hpc/home/$(ARNES_USER)/nlp
