from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

ocr = PaddleOCR(lang='en', use_textline_orientation=False, 
                use_doc_orientation_classify=False, use_doc_unwarping=False)

image_path = 'temp/preprocessed_for_ocr.png'
# output_dir = 'ocr_output_predict'
# os.makedirs(output_dir, exist_ok=True)

# Perform OCR
prediction_results = ocr.predict(image_path)
# for line in prediction_results:
#     print(line)
results_dict = prediction_results[0]  

# -------------------------------------------

# outputing the text, confidence score
# texts = results_dict.get('rec_texts', [])
# scores = results_dict.get('rec_scores', [])

# for text, score in zip(texts, scores):
#     print(f"Text: {text}, Confidence: {score:.3f}")

# -------------------------------------------
# getting bounding boxes and texts in cli
# boxes = results_dict.get('rec_polys', [])

# for text, score, box in zip(texts, scores, boxes):
#     print(f"Text: {text}, Score: {score:.3f}, Box: {box.tolist()}")

# -------------------------------------------

# Draw bounding boxes on the image
img = cv2.imread(image_path)
hImg, wImg = img.shape[:2]

texts = results_dict.get('rec_texts', [])
scores = results_dict.get('rec_scores', [])
boxes = results_dict.get('rec_polys', [])

# Draw each bounding box with its text
# for text, score, box in zip(texts, scores, boxes):
#     pts = box.astype(int)  # ensure integers
#     # Draw the polygon
#     cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#     # Put the recognized text near the first corner
#     x, y = pts[0]
#     cv2.putText(
#         img, f"{text}", (x, y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#         (0, 0, 255), 1, cv2.LINE_AA
#     )

# # Show the image (or save)
# cv2.imshow("OCR Results", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------

# only printing the texts with score greater than a threshold like 0.5 and above
threshold = 0.7
if scores is not None and texts is not None:
    for text, score, box in zip(texts, scores, boxes):
        if score >= threshold:
            print(f"Text: {text}, Score: {score:.3f}")
            pts = box.astype(int)  # ensure integers
            # Draw the polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Put the recognized text near the first corner
            x, y = pts[0]
            cv2.putText(
                img, f"{text}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA
            )

    # Show the image (or save)
    cv2.imshow("OCR Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------






                    
# output_img = os.path.join(output_dir, "ocr_bboxes_from_rec_boxes.png")
# cv2.imwrite(output_img, img)
# print(f"Image with bounding boxes saved: {output_img}")
