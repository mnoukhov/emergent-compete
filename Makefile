# Borgy Image
USERNAME := $(shell whoami 2> /dev/null)
# Use a default value of `nobody` if variable is empty
USERNAME := $(or $(USERNAME),$(USERNAME),nobody)

IMAGE_VERSION=latest
REGISTRY_URL=volatile-images.borgy.elementai.net
IMAGE_NAME=emergent
IMAGE_NAME_AND_TAG=${REGISTRY_URL}/${USERNAME}/${IMAGE_NAME}:${IMAGE_VERSION}

build:
	@echo "Building image: ${IMAGE_NAME_AND_TAG}"
	#echo "export PYPI_ACCESS_KEY=${PYPI_ACCESS_KEY}" >> .dockerenv
	#echo "export PYPI_SECRET_KEY=${PYPI_SECRET_KEY}" >> .dockerenv
	# Enable docker buildkit for passing secrets
	#DOCKER_BUILDKIT=1
		#--secret id=env,src=.dockerenv
	docker build -f Dockerfile --tag $(IMAGE_NAME_AND_TAG) .

push:
	docker push ${IMAGE_NAME_AND_TAG}
