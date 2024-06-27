import cv2
import os 
from google.cloud import vision_v1
from google.cloud.vision_v1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "data\plasma-buckeye-427410-t8-6c2611019c9d.json"
def draw_boxes(image_path, grouped_results):
    image = cv2.imread(image_path)
    for grouped_result in grouped_results:
        bounding_poly = grouped_result.bounding_poly
        x_min = bounding_poly.normalized_vertices[0].x * image.shape[1]
        y_min = bounding_poly.normalized_vertices[0].y * image.shape[0]
        x_max = bounding_poly.normalized_vertices[2].x * image.shape[1]
        y_max = bounding_poly.normalized_vertices[2].y * image.shape[0]
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # 取得產品名稱和分數
        for result in grouped_result.results:
            product = result.product
            product_display_name = product.display_name
            product_score = result.score
            
            # 框線上顯示名稱和分數
            #!  文字跟分數 改成results = response.product_search_results.results
    
            text = f"{product_display_name} ({product_score:.2f})"
            cv2.putText(image, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image with Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def product_search(image_path):
    image_annotator_client = vision_v1.ImageAnnotatorClient()
    project_id = 'plasma-buckeye-427410-t8'
    location = 'asia-east1'
    product_set_id = '5e22bdcb197e2a53'
    product_set_path = f'projects/{project_id}/locations/{location}/productSets/{product_set_id}'

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    request = types.AnnotateImageRequest(
        image=image,
        features=[types.Feature(type=vision_v1.Feature.Type.PRODUCT_SEARCH)],
        image_context=types.ImageContext(
            product_search_params=types.ProductSearchParams(
                product_set=product_set_path,
                product_categories=['homegoods-v2']
            )
        )
    )

    response = image_annotator_client.annotate_image(request)
    grouped_results = response.product_search_results.product_grouped_results
    # boxes = []
    # print("response=",response)
    # for result in response.product_search_results.product_grouped_results:
    #     boxes.append(result.bounding_poly)

    return grouped_results

if __name__ == "__main__":
    image_path = "image\/test\/S__24051720_0.jpg"
    boxes = product_search(image_path)
    draw_boxes(image_path, boxes)
