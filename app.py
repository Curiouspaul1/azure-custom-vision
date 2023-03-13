from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials 
import numpy as np
import os

ENDPOINT = "https://birdvision64.cognitiveservices.azure.com/"

# Replace with a valid key
training_key = "a5c1550ecadb4f7e908f84386e151a8f"
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
publish_iteration_name = "classifyBirdModel"

trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Create a new project
print ("Creating project...")
# project = trainer.create_project("Bird Classification")
# print(project.id)

print("Project created!")

# Create a tag list from folders in bird directory
path = os.path.abspath('./training-data')
tags = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
print(tags)


def createImageList(tag, tag_id):
    # Set directory to current tag.
    tag_id = tag
    base_image_url = f"{path}/{tag}/"
    print(base_image_url)
    photo_name_list = os.listdir(base_image_url)
    image_list = []
    for file_name in photo_name_list:
        with open(base_image_url+file_name, "rb") as image_contents:
            image_list.append(ImageFileCreateEntry(name=base_image_url+file_name, contents=image_contents.read(), tag_ids=[tag_id]))
    return image_list


def uploadImageList(image_list):
    upload_result = trainer.create_images_from_files(project_id="38b1f962-a7dc-4c64-9ccb-d378e873c7d3", batch=image_list)
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)
        exit(-1)


for tag in tags:
    tag_id = tag
    image_list = createImageList(tag, tag_id)
    print("image_list created with length " + str(len(image_list)))

    # Break list into lists of 25 and upload in batches
    for i in range(0, len(image_list), 25):
        batch = ImageFileCreateBatch(images=image_list[i:i + 25])
        print(f'Upload started for batch {i} total items {len(image_list)} for tag {tag}...')
        uploadImageList(batch)
        print(f"Batch {i} Image upload completed. Total uploaded {len(image_list)} for tag {tag}")
