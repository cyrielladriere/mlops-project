REPO_NAME="mlops-project-image-repository"
PROJECT_ID="mlops-project-431716"

IMAGE_NAME="mlops-image"
REPO_URL="europe-west1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
DOCKERFILE_DIR="training_pipeline/src/Dockerfile"

docker build -t ${IMAGE_NAME} -f ${DOCKERFILE_DIR} .
docker tag ${IMAGE_NAME} "${REPO_URL}/${IMAGE_NAME}:latest"

# gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://eu-west1-docker.pkg.dev

docker push "${REPO_URL}/${IMAGE_NAME}:latest"