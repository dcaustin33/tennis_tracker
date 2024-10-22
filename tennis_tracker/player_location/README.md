To get the model weights, follow this link https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?pli=1

Remember the actual keypoint model is run at half dimension so you need to multiply the dimensions by 2. For example, if the original dimensions are 360, 640 and you want to get the middle point on the transformed court the point you want to pass is [360, 640] as that is exactly in the middle of the larger image.

Still have to iterate through all of the output bounding boxes 