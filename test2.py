from google.cloud import vision
import cv2
import numpy as np
import os 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "data\plasma-buckeye-427410-t8-6c2611019c9d.json"

def get_similar_products_file(
   project_id = 'plasma-buckeye-427410-t8',
    location = 'asia-east1',
    product_set_id = '5e22bdcb197e2a53',
    product_category= 'homegoods-v2',
    file_path= "image\/test\/S__24059916_0.jpg",
    filter= 'type=key' ,
    max_results= 5,
):
    

    # product_search_client is needed only for its helper methods.
    product_search_client = vision.ProductSearchClient()
    image_annotator_client = vision.ImageAnnotatorClient()

    # Read the image as a stream of bytes.
    with open(file_path, "rb") as image_file:
        content = image_file.read()

    # Create annotate image request along with product search feature.
    image = vision.Image(content=content)

    # product search specific parameters
    product_set_path = product_search_client.product_set_path(
        project=project_id, location=location, product_set=product_set_id
    )
    product_search_params = vision.ProductSearchParams(
        product_set=product_set_path,
        product_categories=[product_category],
        filter=filter,
    )
    image_context = vision.ImageContext(product_search_params=product_search_params)

    # Search products similar to the image.
    response = image_annotator_client.product_search(
        image, image_context=image_context, max_results=max_results
    )

    index_time = response.product_search_results.index_time
    print("Product set index time: ")
    print(index_time)

    all_results = response.product_search_results.results
    
    print("Search results:")
    for result in all_results:
        product = result.product

        print(f"Score(Confidence): {result.score}")
        print(f"Image name: {result.image}")

        print(f"Product name: {product.name}")
        print("Product display name: {}".format(product.display_name))
        print(f"Product description: {product.description}\n")
        print(f"Product labels: {product.product_labels}\n")
        print('-----------------------------')
        
    results = response.product_search_results.results[0]
    grouped_results = response.product_search_results.product_grouped_results
    
    # 呈現圖片以及繪製框線
    image = cv2.imread(file_path)
    for grouped_result in grouped_results:
        bounding_poly = grouped_result.bounding_poly
        
        x_min = bounding_poly.normalized_vertices[0].x * image.shape[1]
        y_min = bounding_poly.normalized_vertices[0].y * image.shape[0]
        x_max = bounding_poly.normalized_vertices[2].x * image.shape[1]
        y_max = bounding_poly.normalized_vertices[2].y * image.shape[0]
        
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        # 使用 response.product_search_results.results 中的 display_name 和 score
        product = results.product
        cv2.putText(image, f"{product.display_name} ({results.score:.2f})", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.namedWindow('Search Results', cv2.WINDOW_NORMAL)
    cv2.imshow('Search Results', image)
    cv2.resizeWindow('Search Results', 800, 600)  # 設定顯示視窗的寬度和高度
    cv2.waitKey(0)
    cv2.destroyAllWindows()
get_similar_products_file()